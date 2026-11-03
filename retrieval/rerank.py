"""Cross-encoder 两段式重排:在粗排候选集上逐对打分,重新排序。

为什么要有这一层
----------------
双塔(embedding)把 query 和 doc 分别编码成两个向量,相似度是点积。这个结构
决定了它只能看"话题像不像",看不到两者之间的交互。

    "Transformer 一共有多少参数"
    vs. 一段讲 Transformer 结构的文字

向量距离很近(都在讲 Transformer),但那段文字并没有回答参数数量。
这个失败模式在 P1 的阈值标定里已经量化过:5 条伪相关 OOD 埋在正样本的
p25~p75 区间里,双塔分不开。

cross-encoder 把 [query, passage] 拼成一个序列一起喂进 Transformer,
每一层 attention 都能让 query 的 token 直接看过 passage 的 token。
它对"这段文字回答了这个问题吗"比双塔敏锐得多。

代价:它没有可以预先计算的表示。每个 (query, doc) 对都要跑一次完整前向,
N 个候选就是 N 次。所以只能用在粗排之后的小候选集上 —— 这就是"两段式"
(coarse-to-fine)的全部理由:
    召回   用便宜的双塔 + BM25 拿 20 个(RRF,只看名次)
    排序   用贵的 cross-encoder 精修这 20 个
反过来(先重排全库 1210 块)单查询就是 1210 次前向,不可接受。

输入必须是裸对
-------------
bge 的指令前缀只属于 embedding 模型的查询侧。reranker 的训练分布是
(query, passage) 裸对,加前缀等于把输入推出分布外,会掉分。
也不要拼对话历史 —— 历史改写是检索侧的事,不是重排的事。

不要改变闸门的输入
----------------
拒答阈值 0.90 标定在"候选里最小的 vector_distance"上。重排只改顺序,
不改候选集合,所以闸门读到的数不变。
一旦这里开始截断候选(比如粗排 20 却只返回 5),被截掉的块不再参与 min,
闸门口径就变了。所以 retrieve 必须返回 min(k, len(candidates)) 个,
截断交给调用方(rag.py 按 FINAL_K)去做。

模型加载
--------
1.1 GB 的模型不能放在 import 路径上。懒加载 + 模块级单例:第一次 retrieve
才构造,之后复用。Streamlit 每次交互重跑脚本,但 `import rag` 走 sys.modules
缓存,所以 pipeline 对象和这里的模型都只构造一次,不需要 st.cache_resource。
"""

import numpy as np
from sentence_transformers import CrossEncoder

from config import BASE_DIR, COARSE_K, RERANKER_MODEL
from retrieval.base import Retriever, ScoredChunk

_model:CrossEncoder | None = None

def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        # max_length=512 与 CHUNK_SIZE=300 是配套的:中文 1 字约 1~1.5 token,
        # 300 字的块加上 query 仍在 512 以内,不会被截断。
        # 改了分块大小就要重新核对这里,否则块尾会被静默切掉。
        _model = CrossEncoder(str(BASE_DIR / "models" / RERANKER_MODEL), max_length=512)
    return _model

class RerankPipeline:
    def __init__(self, coarse: Retriever, coarse_k: int = COARSE_K):
        self.coarse = coarse
        self.coarse_k = coarse_k

    def retrieve(self, query: str, k: int = COARSE_K) -> list[ScoredChunk]:
        candidates = self.coarse.retrieve(query, k = self.coarse_k)
        if not candidates :
            return []

        model = get_reranker()
        # 裸 (query, passage) 对,不带指令前缀、不拼历史(见模块 docstring)
        pairs = [(query,c.text) for c in candidates]
        # reshape(-1):不同版本的 CrossEncoder 可能返回 (N,) 或 (N,1)
        scores = np.asarray(
            model.predict(pairs,batch_size=32,show_progress_bar=False)
        ).reshape(-1)
        for c,s in zip(candidates, scores):
            c.rerank_score = float(s)

        # list.sort 是稳定的:同分时保留粗排的相对顺序,结果可复现。
        # 别用 torch.topk 或 np.argsort 默认的 quicksort —— 它们在并列处不稳定。
        candidates.sort(key=lambda c:c.rerank_score,reverse=True)
        for i,c in enumerate(candidates,1):
            c.rank = i
        return candidates[:min(k,len(candidates))]
