"""按 config.RETRIEVER 装配检索管线的工厂。"""
from config import COARSE_K, FINAL_K, RETRIEVER
from retrieval.base import Retriever
from retrieval.vector import VectorRetriever

def build_retriever(name: str = RETRIEVER, coarse_k: int = COARSE_K) -> Retriever:
    if name == "vector":
        return VectorRetriever()
    if name == "hybrid":
        from retrieval.hybrid import HybridRetriever
        return HybridRetriever(coarse_k=coarse_k)
    if name == "hybrid_rerank":
        from retrieval.hybrid import HybridRetriever
        from retrieval.rerank import RerankPipeline
        return RerankPipeline(HybridRetriever(coarse_k=coarse_k),
                              coarse_k=coarse_k, top_n=FINAL_K)
    raise ValueError(f"未知的 RETRIEVER: {name!r}(可选 vector / hybrid / hybrid_rerank)")