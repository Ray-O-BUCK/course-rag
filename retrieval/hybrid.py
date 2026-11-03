"""混合检索 = 向量路 + BM25 路,RRF 融合。

两路互补在哪里(实测,81 条)
------------------------
top-20 重叠率仅 32.0%(平均 6.4/20 条相同),说明两路在找**不同的东西** ——
这是融合有收益的前提,重叠 90% 的融合最多新增 10% 候选,收益封顶。
分类别看:
    multi-chunk     53.0%   都在找同一个"主题"段落 —— 重叠最高,融合增益也最小
                            (R@10:vector 0.700 → hybrid 0.800,而 term-match 是 0.760 → 0.960)
    paraphrase      38.1%
    term-match      31.8%
    ood             5.0%    两路都找不到,一致地失败
    ood-adjacent    8.8%
    ood-pseudo      35.0%   词面重叠的伪相关题上两路**一致地错**(n=3,只当信号)
                            —— 这是拒答闸门挡不住伪相关题的结构性原因之一

工程注意
-------
BM25Retriever 在 __init__ 里建索引(1210 块约 1s)。Streamlit 里必须用
st.cache_resource 缓存这个对象,否则每次 rerun 都重建。
"""
from config import COARSE_K, RRF_K
from retrieval.base import ScoredChunk
from retrieval.bm25 import BM25Retriever
from retrieval.fusion import rrf_fuse
from retrieval.vector import VectorRetriever

class HybridRetriever:
    def __init__(self,coarse_k:int = COARSE_K,rrf_k:int = RRF_K,weights:tuple[float,float] = (1.0,1.0)):
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()
        self.coarse_k,self.rrf_k = coarse_k,rrf_k
        self.weights = list(weights)

    def retrieve(self,query:str,k:int = COARSE_K) -> list[ScoredChunk]:
        vec = self.vector.retrieve(query, k=self.coarse_k)
        lex = self.bm25.retrieve(query, k=self.coarse_k)
        # 两路各取 coarse_k,融合后的并集可能到 2*coarse_k,再截到 k
        return rrf_fuse([vec, lex], k=self.rrf_k, weights=self.weights)[:k]