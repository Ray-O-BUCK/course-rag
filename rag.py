from embedder import embed_query
from ingest import collection
from llm import chat
from config import TOP_K,DISTANCE_THRESHOLD

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    q_emb = embed_query(query)
    res = collection.query(query_embeddings= [q_emb], n_results= top_k)
    docs = []
    for i in range(len(res["ids"][0])):
        docs.append({
            "id": res["ids"][0][i], # 入库时唯一id
            "text": res["documents"][0][i], # 块内容
            "doc": res["metadatas"][0][i]["doc"], # 来源
            "chunk_id": res["metadatas"][0][i]["chunk_id"], # 块id
            "distance": res["distances"][0][i], # 距离,越小越相似
        })
    return docs

SYSTEM = """你是课程资料问答助手。根据【参考资料】回答用户问题。
要求：
1. 只使用参考资料中的信息，不要编造。
2. 每条结论后用 [doc:文件名#chunk_id] 标注来源。
3. 如果资料无法回答，直接回答“根据现有资料无法回答该问题”，不要猜测。"""

def answer(query: str, top_k: int = TOP_K) -> dict:
    docs = retrieve(query, top_k)

    if not docs or docs[0]["distance"] > DISTANCE_THRESHOLD:
        return {"answer": "根据现有资料无法回答该问题.", "sources": []}

    context = "\n\n".join(f"[doc:{d['doc']}#{d['chunk_id']}]\n{d['text']}"for d in docs)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"参考资料: \n{context}\n\n问题: {query}"},
    ]
    res = chat(messages)
    return {"answer": res, "sources": docs}