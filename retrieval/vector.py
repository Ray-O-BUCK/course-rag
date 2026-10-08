"""向量检索:套上统一接口。"""
from db import collection
from embedder import embed_query
from retrieval.base import ScoredChunk

class VectorRetriever:
    def retrieve(self, query: str, k: int = 20 ) -> list[ScoredChunk]:
        n = collection.count()
        if n == 0:
            return []
        # n_results 大于库容时 Chroma 会静默少返回,先钳一下更明确
        res = collection.query(
            query_embeddings= [embed_query(query)],
            n_results= min(n, k),
            include = ["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(res["ids"][0])):
            meta = res["metadatas"][0][i]
            chunks.append(ScoredChunk(
                key = f"{meta['doc']}#{meta['chunk_id']}",
                doc = meta["doc"],
                chunk_id = meta["chunk_id"],
                text = res["documents"][0][i],
                vector_distance = res["distances"][0][i],
                rank = i + 1
            ))
        return chunks
