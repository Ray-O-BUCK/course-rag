"""按 config.RETRIEVER 装配检索管线的工厂。"""
from config import COARSE_K, RETRIEVER, RRF_K, RRF_WEIGHTS
from retrieval.base import Retriever
from retrieval.vector import VectorRetriever

def build_retriever(name: str = RETRIEVER, coarse_k: int = COARSE_K) -> Retriever:
    if name == "vector":
        return VectorRetriever()
    if name == "bm25":
        # 单路 BM25。消融必须有这一组:否则"hybrid 比 vector 好"证明不了
        # 融合本身有价值 —— 可能只是 BM25 比向量强,融合反而稀释了它。
        from retrieval.bm25 import BM25Retriever
        return BM25Retriever()
    if name == "hybrid":
        from retrieval.hybrid import HybridRetriever
        return HybridRetriever(coarse_k=coarse_k, weights=RRF_WEIGHTS)
    if name in ("hybrid_rerank", "hybrid_rerank_rewrite"):
        from retrieval.hybrid import HybridRetriever
        from retrieval.rerank import RerankPipeline
        base = HybridRetriever(coarse_k=coarse_k)
        if name == "hybrid_rerank_rewrite":
            from retrieval.rewrite import QueryRewriteRetriever
            base = QueryRewriteRetriever(base, k_rrf = RRF_K)
        return RerankPipeline(base, coarse_k=coarse_k)
    raise ValueError(f"未知的 RETRIEVER: {name!r}(可选 vector / bm25 / hybrid / hybrid_rerank / hybrid_rerank_rewrite)")