"""扩评测集时的锚点辅助工具。

锚点有两个硬约束,靠人眼在 1210 块里核是不现实的:
  1. 必须在**全库唯一** —— 否则 gold 不确定,整份评测数字作废
  2. 最好落在目标块的**偏移安全区** —— 落在 overlap 区的句子天然会被相邻块
     也包含,直接违反第 1 条

安全区的边界不用手算:工具对每个候选句都拿全库实测一遍,唯一就是唯一。
理论安全区 [overlap, chunk_size-overlap) 只是"往哪儿找"的提示。

用法
----
    # 按关键词找块(先定题,再找锚点)
    python scripts/suggest_anchors.py --grep "批归一化"

    # 看某个块,并列出它内部可用的锚点
    python scripts/suggest_anchors.py --chunk d2l-zh__chapter_multilayer-perceptrons__mlp.md#3

    # 检查某句话是否全库唯一
    python scripts/suggest_anchors.py --check "ReLU 计算简单、缓解梯度消失"
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anchors import load_store  # noqa: E402

# 太短的锚点没有区分度,而且容易在讲同一主题的地方重复出现
MIN_ANCHOR_LEN = 10


def hits(store, anchor: str) -> list:
    return [c for c in store if anchor in c.text]


def unique(store, anchor: str) -> tuple[bool, str]:
    h = hits(store, anchor)
    if not h:
        return False, "0 命中(不在语料里)"
    if len(h) > 1:
        spans = [f"{c.key}[{c.text.find(anchor)}]" for c in h]
        return False, f"{len(h)} 命中: " + ", ".join(spans)
    off = h[0].text.find(anchor)
    return True, f"{h[0].key} offset={off}"


def minimal_unique(store, key: str, anchor: str) -> tuple[str, int] | None:
    """把撞车的锚点扩成"刚好唯一"的最短片段,以 key 指定的那块为基准。

    锚点横跨 overlap 边界时会同时出现在块 i 和 i+1 里 —— 这是最常见的失败。
    往左扩(伸到块 i+1 起点之前)或往右扩(伸进块 i+1 之外)都能解决,
    取最短的:锚点越短,越不容易在别的文档里也撞上。

    注意基准块必须由调用方指定。撞车的定义就是"命中不止一块",
    所以"唯一命中一块"不能拿来当守卫 —— 那等于要求它别撞车才肯帮忙修撞车。
    """
    c = next((c for c in store if c.key == key), None)
    if c is None:
        return None
    i = c.text.find(anchor)
    if i < 0:
        return None
    j = i + len(anchor)
    for grow in range(1, 61):
        found = []
        for lo, hi in ((i - grow, j), (i, j + grow), (i - grow, j + grow)):
            lo, hi = max(0, lo), min(len(c.text), hi)
            s = c.text[lo:hi]
            if len([x for x in store if s in x.text]) == 1:
                found.append((len(s), s, lo))
        if found:
            _, s, lo = min(found)
            return s, lo
    return None


def candidates(text: str) -> list[str]:
    """把块切成候选句。

    按中文句读和换行切 —— 锚点不能含换行(评测集是 JSONL,含换行会破坏格式,
    而且跨行的片段在重分块时更容易被切开)。
    """
    out, buf = [], ""
    for ch in text:
        if ch == "\n":
            if buf.strip():
                out.append(buf.strip())
            buf = ""
            continue
        buf += ch
        if ch in "。!?;!?；":
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    # 长句优先:短句在 1210 块里撞车概率高得多
    return sorted({s for s in out if len(s) >= MIN_ANCHOR_LEN}, key=len, reverse=True)


def show_chunk(store, key: str) -> None:
    target = next((c for c in store if c.key == key), None)
    if target is None:
        near = [c.key for c in store if key.split("#")[0] in c.key][:8]
        print(f"没有 {key!r}。相近的:{near}")
        return

    print(f"=== {target.key} ===  ({len(target.text)} 字符)\n")
    print(target.text)
    print(f"\n--- 可用锚点(全库唯一,按长度降序)---")
    ok = 0
    for s in candidates(target.text):
        good, why = unique(store, s)
        if good:
            ok += 1
            print(f"  [{why}]  {s}")
    if not ok:
        print("  这个块里没有唯一句 —— 换一块,或人工挑更长的片段")
    print(f"\n共 {ok} 个可用。挑 1~2 句写进 anchors。")


def main() -> None:
    ap = argparse.ArgumentParser(description="锚点辅助工具")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--grep", help="按关键词在全库找块")
    g.add_argument("--chunk", help="查看某个块(doc#chunk_id)")
    g.add_argument("--check", help="检查语句是否全库唯一")
    g.add_argument("--fix", help="把撞车的锚点扩成唯一: --fix '<锚点>' --key <doc#chunk_id>")
    ap.add_argument("--key", help="--fix 用:以哪一块为基准扩(即你希望 gold 落在哪块)")
    ap.add_argument("-n", type=int, default=15, help="--grep 最多显示多少块")
    args = ap.parse_args()

    from db import collection

    store = load_store(collection)
    print(f"库内 {len(store)} 块\n")

    if args.fix:
        if not args.key:
            print("--fix 需要同时给 --key(doc#chunk_id)")
            return
        got = minimal_unique(store, args.key, args.fix)
        if got is None:
            print(f"扩不出来:{args.key} 里找不到 {args.fix[:30]!r},或左右扩 60 字仍撞车")
            hits = [c.key for c in store if args.fix in c.text]
            print(f"  原始片段命中 {len(hits)} 块: {hits[:6]}")
        else:
            s, lo = got
            print(f"唯一化后 offset={lo}:\n  {s}")

    elif args.chunk:
        show_chunk(store, args.chunk)

    elif args.check:
        good, why = unique(store, args.check)
        print(("唯一 ✓  " if good else "不可用 ✗  ") + why)

    else:
        kw = args.grep
        found = [c for c in store if kw in c.text]
        print(f"含 {kw!r} 的块:{len(found)} 个\n")
        for c in found[:args.n]:
            i = c.text.find(kw)
            lo = max(0, i - 40)
            print(f"  {c.key}")
            print(f"      ...{c.text[lo:i + len(kw) + 40]}...")
        if len(found) > args.n:
            print(f"\n  ...还有 {len(found) - args.n} 个,用 -n 调大")


if __name__ == "__main__":
    main()
