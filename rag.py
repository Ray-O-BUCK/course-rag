import re
from config import (COARSE_K, CONTEXT_MAX_CHARS, CONTEXT_MIN_RERANK,
                    DISTANCE_THRESHOLD, FINAL_K, GATE_SIGNAL,
                    RERANK_THRESHOLD, RETRIEVER,STRICT_CITATION)
from llm import chat
from retrieval.base import ScoredChunk
from retrieval.pipeline import build_retriever

SYSTEM = """你是课程资料问答助手。根据【参考资料】回答用户问题。
要求：
1. 只使用参考资料中的信息，不要编造。
2. 每条结论后用 [doc:文件名#chunk_id] 标注来源。
3. 如果资料无法回答, 直接回答"根据现有资料无法回答该问题",不要猜测。"""

REFUSAL = "根据现有资料无法回答该问题."
# LLM 按 prompt 约定输出 [doc:文件名#chunk_id]。这里只提取 doc:...#...，
# 不依赖方括号形状，模型写成 [doc:...]、［doc:...］ 都能抓到。
_CITATION_RE = re.compile(r"doc[:：]\s*([^\[\]\s#]+?)\s*#\s*(\d+)")

def validate_citations(text: str, chunks: list[ScoredChunk]) -> dict:
    """校验回答里的引用是否真来自本次送进 LLM 的上下文。

        只验证"引用可落地"，不验证"这句话是否被该块支持"——后者是语义忠实度，
        需要 LLM judge 或人工标注。这里能确定性回答的是：
        引用的 doc#chunk_id 是否在 kept 块集合里。
        """
    allowed = {f"{c.doc}#{c.chunk_id}" for c in chunks}
    cited: list[str] = []
    for m in _CITATION_RE.finditer(text):
        key = f"{m.group(1).strip()}#{m.group(2)}"
        if key not in cited:
            cited.append(key)
    return {
        "cited": cited,
        "valid": [k for k in cited if k in allowed],
        "invalid": [k for k in cited if k not in allowed],
        "no_citation": not cited,
    }

# 管线在模块加载时装配一次;模型/索引都是懒加载,这里只是组装对象
pipeline = build_retriever()

def build_context(chunks: list[ScoredChunk], max_chars: int = CONTEXT_MAX_CHARS) -> str:
    """按顺序拼参考资料, 超出字符预算就截断 -- 不在无脑塞满 top_k。

    预算是硬上限,块之间的 "\\n\\n" 也计入;否则实际 prompt 会比 max_chars 长,
    预算就只是个近似值。
    """
    parts, total = [], 0
    for c in chunks:
        block = f"[doc:{c.doc}#{c.chunk_id}]\n{c.text}"
        extra = len(block) + (2 if parts else 0)   # 2 = 块间分隔 "\n\n"
        if parts and total + extra > max_chars:
            break
        parts.append(block)
        total += extra
    return "\n\n".join(parts)

def _should_refuse(candidates:list[ScoredChunk]) -> bool:
    """拒答闸门。信号由 config.GATE_SIGNAL 选择,语义见 config.py 的注释。

        信号缺失时抛错,不静默退回另一种闸门 —— 换口径会让标定数字和线上行为
        对不上,而这是这个项目最不能接受的一类静默错误。
        """
    if GATE_SIGNAL == "rerank":
        rrs = [c.rerank_score for c in candidates if c.rerank_score is not None]
        if not rrs:
            raise RuntimeError(f"GATE_SIGNAL={GATE_SIGNAL!r} 但检索器 {RETRIEVER!r} 没有产生 rerank 分数")
        return max(rrs) < RERANK_THRESHOLD

    # 向量信号:候选里最小的向量距离 = 向量路自己的 top-1。
    # 融合后 candidates[0] 可能来自 BM25、没有向量分数(None),不能取 [0]。
    # 向量 rank-1 一定在融合并集里(至少拿到 1/61 分),所以 min 就等于它。
    # 空列表同样抛错:bm25 检索器配 GATE_SIGNAL="vector" 时所有候选都没有
    # vector_distance,静默拒答等于让线上行为悄悄变成"什么都不答"。
    vds = [c.vector_distance for c in candidates if c.vector_distance is not None]
    if not vds:
        raise RuntimeError(f"GATE_SIGNAL={GATE_SIGNAL!r} 但检索器 {RETRIEVER!r} 没有产生 vector 距离")
    return min(vds) > DISTANCE_THRESHOLD

def _keep_for_context(c: ScoredChunk) -> bool:
    """逐块过滤:只把 rerank 分数过绝对下限的块送进上下文。

        为什么不用 RERANK_THRESHOLD:那是"候选里最好的证据"的闸门线。
        拿 0.16 筛每一块会丢掉 c46(0.041)、c71(0.119) 这类有 gold 的块,
        等于把 rank 7~9 的正确证据又筛掉了 —— 这正是 P2 时犯过的错。
        """
    return c.rerank_score is not None and c.rerank_score >= CONTEXT_MIN_RERANK

def answer(query: str, top_k: int = COARSE_K) -> dict:
    candidates = pipeline.retrieve(query, k = top_k)
    if _should_refuse(candidates):
        return {"answer": REFUSAL, "sources": []}

    # 逐块过滤只在 rerank 信号下启用:向量信号没有逐块相关性标定,
    # 按距离筛会丢掉 BM25 单独召回的块(没有 vector_distance)。
    if GATE_SIGNAL == 'rerank':
        kept = [c for c in candidates if _keep_for_context(c)][:FINAL_K]
    else:
        kept = candidates[:FINAL_K]

    # 闸门保证 max(rerank_score) >= 0.16 > 0.02,理论上不可能为空;
    # 真出现就是两个阈值口径漂了,宁可报错也不把空上下文送进 LLM。
    if not kept:
        raise RuntimeError("闸门放行但逐块过滤后没有候选,两个阈值口径不一致")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"参考资料:\n{build_context(kept)}\n\n问题: {query}"},
    ]
    text = chat(messages)
    check = validate_citations(text, kept)
    if STRICT_CITATION and not check["valid"]:
        return {"answer": REFUSAL, "sources": [], "citation_check": check, "strict_citation": True}
    return {
        "answer": text,
        "sources": kept,
        "citation_check": check
    }
