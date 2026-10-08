from config import COARSE_K, CONTEXT_MAX_CHARS, DISTANCE_THRESHOLD, FINAL_K
from llm import chat
from retrieval.base import ScoredChunk
from retrieval.pipeline import build_retriever

SYSTEM = """你是课程资料问答助手。根据【参考资料】回答用户问题。
要求：
1. 只使用参考资料中的信息，不要编造。
2. 每条结论后用 [doc:文件名#chunk_id] 标注来源。
3. 如果资料无法回答, 直接回答"根据现有资料无法回答该问题",不要猜测。"""

REFUSAL = "根据现有资料无法回答该问题."

# 管线在模块加载时装配一次;模型/索引都是懒加载,这里只是组装对象
pipeline = build_retriever()

def build_context(chunks: list[ScoredChunk], max_chars: int = CONTEXT_MAX_CHARS) -> str:
    """按顺序拼参考资料, 超出字符预算就截断 -- 不在无脑塞满 top_k。"""
    parts, total = [], 0
    for c in chunks:
        block = f"[doc:{c.doc}#{c.chunk_id}]\n{c.text}"
        if parts and total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)

def answer(query: str, top_k: int = COARSE_K) -> dict:
    candidates = pipeline.retrieve(query, k = top_k)

    if not candidates or candidates[0].vector_distance > DISTANCE_THRESHOLD:
        return {"answer": REFUSAL, "sources": []}

    # 进上下文的块按"名次"选,不再按绝对距离过滤:
    # 阈值是给拒答门用的,拿它筛上下文会把 rank 7~9 的正确块一起筛掉
    kept = candidates[:FINAL_K]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"参考资料:\n{build_context(kept)}\n\n问题: {query}"},
    ]
    return {"answer": chat(messages), "sources": kept}
