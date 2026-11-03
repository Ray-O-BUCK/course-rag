"""手写 BM25 —— 与 rank_bm25.BM25Okapi 严格对齐,可对拍到 1e-9。

为什么手写
----------
1. 这个项目的定位是"搞懂底层":BM25 是检索岗的必考点,调包等于放弃这块。
2. 手写 + 与参考实现对拍,比"用了 rank_bm25"有说服力得多 —— 前者证明理解了公式,
   后者只证明会 pip install。

为什么能对拍到 1e-9
-------------------
同语料、同分词器、同 k1/b/epsilon 下,两边算的是同一个闭式公式,没有近似。
唯一的"实现自由度"是 IDF 的负数处理(见 _build_idf),两边必须选同一种约定。
rank_bm25 的默认是 epsilon=0.25,这里照抄。

K1/B 的直觉(面试会问)
--------------------
    score = Σ_t IDF(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1-b + b*|d|/avgdl))

    IDF(t)   词的稀有度。这是 BM25 打赢向量检索的地方:精确的词面证据权重自动放大。
    k1       词频饱和速度。f 不是线性计入,而是 f/(f+K) 的形式 —— 出现 10 次和 100 次
             的相关性没有 10 倍差别,线性计入会让复读机块霸榜。k1 越大越接近线性。
    b        长度归一强度。长块天生更容易命中查询词(容错高),要惩罚。
             b=0 不惩罚,b=1 完全按 |d|/avgdl 归一。
    k1=1.5 / b=0.75 是 TREC 时代的经验值,几乎所有实现都用它俩。
"""

import math
import  re
from collections import Counter
import jieba

from db import collection
from retrieval.base import ScoredChunk

K1 = 1.5
B = 0.75
EPSILON = 0.25

# --- 文本归一化 ------------------------------------------------------------
# 语料体检结果(1210 块):URL 148 块 / markdown 链接 173 块 / LaTeX 命令 196 块。
# 不清理的话,"https://huggingface.co/datasets/billsum" 会把 https / huggingface /
# co / datasets 全塞进词表。实测(未清洗)这四个 token 分别出现在 147 / 110 / 91 / 58
# 块里,接近停用词(df 越大 IDF 越低);清洗后降到 0 / 18 / 7 / 39,才恢复成有区分度的词。
# 链接文字要保留:[BillSum](https://...) 里的 billsum 是数据集名,是内容;URL 不是。
_URL = re.compile(r"https?://\S+")
# [文字](链接) -> 文字。链接文字是内容(BillSum 是数据集名),URL 不是。
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# 其余非"字母/数字/下划线/中文"的字符一律当分隔符。
_NON_WORD = re.compile(r"[^\w一-鿿]+")
# 过滤掉纯符号 token(如 jieba 可能吐出的 "^" "_" "/")
_HAS_CONTENT = re.compile(r"[0-9A-Za-z一-鿿]")

def normalize(text: str) -> str:
    """把 markdown/LaTex 骨架清掉,只留可检索的内容."""
    text = _URL.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _NON_WORD.sub(" ", text)
    text = text.strip()
    return text

def tokenize(text: str) -> list[str]:
    """切词。

       用 cut_for_search 而不是 lcut:前者对长词会同时吐出长词和它的子词
       ("初始化" 同时产出 "初始" 和 "初始化"),召回更高 —— 查询 "Xavier 初始化"
       能命中只写了 "初始化方案" 的块。代价是块长被略微高估,可接受。

       不加自定义词典:实测 jieba 把连续的 ASCII 字母当整词处理,RoPE / WordPiece /
       DistilBERT / Xavier / LSTM 都完整保留,不需要干预。
       """
    return [t.lower() for t in jieba.cut_for_search(normalize((text))) if _HAS_CONTENT.search(t)]

class BM25Index:
    """倒排统计 + 打分。索引在内存里,几百块毫秒级重建。"""
    def __init__(self, docs:list[str], tokenizer=tokenize, k1:float=K1, b:float=B, epsilon:float=EPSILON):
        self.k1,self.b,self.epsilon = k1, b, epsilon
        self.corpus: list[list[str]] = [tokenizer(doc) for doc in docs]             # 每篇文档的 token 列表
        self.n = len(self.corpus)                                                   # 文档总数
        # Counter 而不是 list:打分时要频繁查"词 t 在这块里出现几次"
        self.doc_freqs = [Counter(doc) for doc in self.corpus]                      # 每篇文档的词频 Counter
        self.doc_len = [len(doc) for doc in self.corpus]                          # 每篇文档的长度
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0   # 平均文档长度
        self.idf: dict[str, float] = {}
        self._build_idf()

    def _build_idf(self) -> None:
        """IDF 用的是 BM25 原始形式,允许为负,再把负值统一压到 epsilon*avg_idf。

        为什么会有负数:log((N-df+0.5)/(df+0.5)) 在 df > N/2 时变负 —— 也就是出现在
        超过一半文档里的词贡献负分。直觉上这合理(词太常见=没区分度=该惩罚),但会让
        "查一个高频词"的结果很反直觉,所以各实现都做了修正。rank_bm25 的做法是把负值
        换成 epsilon * 平均 IDF,这里照抄 —— 不照抄就对拍不上。
        """
        df:Counter = Counter()
        for doc_freq in self.doc_freqs:
            df.update(doc_freq.keys())      # 每个词在每块里只计一次

        total = 0.0
        negative = []
        for term, d in df.items():
            v = math.log(self.n - d + 0.5) - math.log(d + 0.5)
            self.idf[term] = v
            total += v
            if v < 0:
                negative.append(term)

        # 存成属性,不是为了后面要用 —— 是给 scripts/compare_bm25.py 比结构量用。
        # 它必须是**修正前**的平均值:负值被替换成 eps 后 sum(idf.values()) 会变大,
        # 拿替换后的均值去比 rank_bm25.average_idf 会差 4e-4,那是比错了量。
        self.average_idf = total / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * self.average_idf
        for term in negative:
            self.idf[term] = eps

    def get_scores(self, query: str) -> list[float]:
        """返回全库分数(不截断)。检索用不到,只为与 rank_bm25 对拍。"""
        scores = [0.0] * self.n
        if not self.avgdl:
            return scores
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i,freqs in enumerate(self.doc_freqs):
                f = freqs.get(term)
                if not f:
                    continue
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
        return scores

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """返回 (文档下标, 分数) 的 top-k,只保留分数 > 0 的。

        全 0 的块说明一个查询词都没命中,塞进 top-k 只会挤掉真正有信号的结果。
        """
        scores = self.get_scores(query)
        order = sorted(range(self.n), key = scores.__getitem__, reverse = True)[:k]
        return [(i, scores[i]) for i in order if scores[i] > 0]

class BM25Retriever:
    """从 Chroma 现拉文本建索引,不落盘 —— 永远与向量库一致。

        几百块重建索引是毫秒级,不值得引入"索引文件和库不同步"这类新故障。
        块数上万时这里要改成落盘索引,README 里记了这个可扩展边界。
        """

    def __init__(self):
        got = collection.get(include = ["documents","metadatas"])
        # 标识用 (doc, chunk_id) 拼出的 key,而不是 get() 返回的下标 ——
        # 下标只在本次调用内有意义,跨检索器/跨次调用会漂。
        self.items = [
            (f"{m['doc']}#{m['chunk_id']}",m["doc"],m["chunk_id"],text)
            for text,m in zip(got["documents"],got["metadatas"])
        ]
        self.index = BM25Index([it[3] for it in self.items])

    def retrieve(self,query:str,k:int=20) -> list[ScoredChunk]:
        out = []
        for rank, (i, score) in enumerate(self.index.search(query, k), 1):
            key, doc, chunk_id, text = self.items[i]
            out.append(ScoredChunk(key=key, doc=doc, chunk_id=chunk_id,
                                   text=text, bm25_score=score, rank=rank))
        return out
