"""检索器的统一接口与数据结构。

所有检索器(向量/BM25/混合/重排)都返回 list[ScoredChunk];
各路的分数写在自己的字段里、互不覆盖 —— 融合与评估都靠这个约定。
"""
from dataclasses import dataclass
from typing import Protocol

@dataclass
class ScoredChunk:
    key: str                                # 全局唯一键 "doc#chunk_id",各路共用
    doc: str
    chunk_id: int
    text: str
    vector_distance: float | None = None    # l2 距离,越小越相关
    bm25_score: float | None=  None         # BM25 分,越大越相关
    fusion_score: float | None = None       # RRF 融合分,越大越相关
    rerank_score: float | None = None       # cross-encoder sigmoid 分 0~1
    rank: int = 0                           # 在本路结果中的名次,从 1 开始

class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 20) -> list[ScoredChunk]:
        ...