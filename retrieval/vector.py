"""向量检索:在本进程内做精确 l2 检索,不用 Chroma 的 HNSW 近似查询。

为什么不调 collection.query
------------------------
Chroma 1.5.9 的 HNSW 查询跨进程不可复现。实测:同一份脚本、同一份语料、同一份
索引文件连跑三次,top-20 的集合在 23/81 条题上不同,top-1 是 0/81 —— 典型的
近似检索长尾误差,随 k 增大而放大。对"用数据说话"的消融实验这是致命的:
两次运行的指标差分不清是配置变了还是索引抖了。

1210 块 x 512 维的暴力检索是毫秒级,这点规模没有理由拿可复现性去换延迟。
块数上万后再换回 ANN,那时用召回换延迟才是划算的。

距离口径
-------
对外报的是**平方 l2**,不是 l2。Chroma 的 "l2" 空间返回的就是平方 l2:
实测 mlp.md#27 返回 0.8005499840,精确平方 l2 是 0.8005499325,而开方是
0.8947346 —— 对不上的是开方那个。config.DISTANCE_THRESHOLD(0.90)是在
Chroma 的返回值上标定的,所以这里必须沿用平方 l2,否则阈值整个作废。

索引新鲜度
---------
embedding 是构造时的快照。重跑 ingest.py 之后要重新构造检索器 —— 旧实现每次
query 都问 Chroma,不会有这个问题;这里是拿"永远与库同步"换"可复现"。
"""
import numpy as np

from db import collection
from embedder import embed_query
from retrieval.base import ScoredChunk

class VectorRetriever:
    def __init__(self):
        got = collection.get(include=["embeddings", "documents", "metadatas"])
        metas = got["metadatas"] or []
        self.keys = [f"{m['doc']}#{m['chunk_id']}" for m in metas]
        self.docs = [m["doc"] for m in metas]
        self.chunk_ids = [m["chunk_id"] for m in metas]
        self.texts = got["documents"] or []
        emb = got["embeddings"]
        # float64:库里的 embedding 是 float32,升到 float64 是无损的。
        # 距离累加用高精度,并列名次才不会被舍入噪声随机决定。
        self.emb = (np.asarray(emb, dtype=np.float64) if emb is not None
                    else np.empty((0, 0)))

    def retrieve(self, query: str, k: int = 20) -> list[ScoredChunk]:
        n = len(self.keys)
        if n == 0:
            return []
        q = np.asarray(embed_query(query), dtype=np.float64)
        # 平方 l2,与 Chroma 的 "l2" 空间同口径(见模块 docstring)。
        # 只排序的话平方就够(单调),开方只是为了对上阈值的量纲 —— 这里不开。
        dist = ((self.emb - q) ** 2).sum(axis=1)

        # kind="stable":并列时按库内顺序,跨进程可复现。
        # np.argsort 默认 quicksort 不稳定,同一份输入在并列处可能给出不同名次。
        order = np.argsort(dist, kind="stable")[:min(k, n)]
        return [
            ScoredChunk(
                key=self.keys[i],
                doc=self.docs[i],
                chunk_id=self.chunk_ids[i],
                text=self.texts[i],
                vector_distance=float(dist[i]),
                rank=r + 1,
            )
            for r, i in enumerate(order)
        ]
