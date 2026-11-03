"""分块策略。

fixed:     定长窗口 + overlap，作为消融基线。
structure: 按 Markdown 标题切段；长段再按段落/句子/字符拆；
           短段并入相邻块。标题路径（breadcrumb）会写进块文本。
"""
import re

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(?:```|~~~)")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    """按标题切段，返回 [(breadcrumb, body)]，body 保留自己的标题行。

    两个关键点：
    1. 用 in_fence 跟踪 ``` / ~~~ 围栏。d2l 的 #@save、#@tab 以及代码块里的
       Python 注释都以 # 开头，不跟踪围栏就会被误当成标题。
    2. stack 保存标题层级，生成 "文档标题 > 小节 > 子小节" 的路径。
    """
    out = []
    stack = []          # [(level, title)]
    body = []
    prefix = ""
    in_fence = False
    starts_with_heading = False

    def flush():
        nonlocal body, starts_with_heading
        chunk = "\n".join(body).strip()
        # 只有标题、没有正文的段不单独成块；标题已经进了 breadcrumb
        content = body[1:] if starts_with_heading else body
        if chunk and "\n".join(content).strip():
            out.append((prefix, chunk))
        body = []
        starts_with_heading = False

    for line in text.splitlines():
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence             # 遇到 ``` 或 ~~~ 就翻转状态

        m = None if in_fence else _HEADING_RE.match(line)       # 在围栏内不识别标题
        if m is None:
            body.append(line)                   # 普通行 → 累进 body
            continue

        flush()                         # 遇到标题行就结算上一个块
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()                     # 弹出所有 >= 当前层级的（回退）
        stack.append((level, title))        # 压入当前标题
        prefix = " > ".join(t for _, t in stack)
        body = [line]   # 标题行留在正文里，块内仍能看出层次
        starts_with_heading = True

    flush()
    return out


def _split_long(text: str, budget: int, overlap: int) -> list[str]:
    """单段超过预算时：先按句号切；句子仍超长再按字符窗口切。"""
    sents = [s for s in re.split(r"(?<=[。！？!?；;])", text) if s.strip()]
    pieces, cur = [], ""
    for s in sents:
        if len(s) <= budget:
            if cur and len(cur) + len(s) > budget:
                pieces.append(cur)
                cur = s
            else:
                cur += s
            continue
        # 句子本身 > budget：先收掉 cur，再对这句按字符窗口切
        if cur:                         # 句子本身 > budget：先收掉 cur，再对这句按字符窗口切
            pieces.append(cur)
            cur = ""
        step = max(1, budget - min(overlap, budget - 1))  # 窗口步长
        step = max(1, budget - min(overlap, budget - 1))
        for i in range(0, len(s), step):  # 按窗口切句
            piece = s[i:i + budget].strip()
            if piece:
                pieces.append(piece)
            if i + budget >= len(s):
                break
    if cur:
        pieces.append(cur)
    return pieces


def _pack(text: str, budget: int, overlap: int) -> list[str]:
    """按空行分段装包；遇到超长段交给 _split_long。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, cur = [], ""
    for p in paras:
        if len(p) > budget:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_split_long(p, budget, overlap))
            continue
        if cur and len(cur) + 2 + len(p) > budget:
            pieces.append(cur)
            cur = p
        else:
            cur = p if not cur else cur + "\n\n" + p
    if cur:
        pieces.append(cur)
    return pieces


def chunk_struct(text: str, max_chars: int = 500, overlap: int = 80,
                 min_chars: int = 100) -> list[str]:
    chunks = []
    for prefix, body in _sections(text):
        head = prefix + "\n" if prefix else ""
        budget = max(1, max_chars - len(head))
        pieces = [body] if len(body) <= budget else _pack(body, budget, overlap)
        for p in pieces:
            c = (head + p).strip()
            if c:
                chunks.append(c)

    # 短块并入前一块，避免标题下只有一两句话时单独成块
    merged = []
    for c in chunks:
        if merged and len(c) < min_chars and len(merged[-1]) + 2 + len(c) <= max_chars:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged
