"""评测集的加载、锚点解析与校验。

为什么不直接标 chunk_id:重切分块会让所有 chunk_id 漂移,标注全部作废。
锚点存的是"目标块里的一个唯一短句片段" —— 只要那句话还在语料里,就能重新
定位到当前分块方案下的块。换 chunker、调 chunk_size 都不用重标。

代价:锚点必须在全库唯一。分块带 overlap 时,落在重叠区的句子会同时出现在
相邻两个块里,于是"命中多个块"而无法确定 gold。所以改完评测集一定要跑一遍:

    python eval/anchors.py
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# 无论从哪个目录运行,都能导入项目根目录和本目录的模块
ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EVAL_SET_PATH = EVAL_DIR / "eval_set_chunks.jsonl"


class AnchorError(ValueError):
    """锚点无法唯一解析。"""


@dataclass
class ChunkRef:
    key: str
    doc: str
    chunk_id: int
    text: str


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def load_store(collection) -> list[ChunkRef]:
    """把全库的块读进内存,供锚点反查。

    数百块规模直接全量读;块数上万时这里要换成只拉 anchor 命中的块。
    """
    got = collection.get(include=["documents", "metadatas"])
    refs = []
    for i, _id in enumerate(got["ids"]):
        meta = got["metadatas"][i]
        refs.append(ChunkRef(
            key=f"{meta['doc']}#{meta['chunk_id']}",
            doc=meta["doc"],
            chunk_id=meta["chunk_id"],
            text=got["documents"][i],
        ))
    return refs


def resolve_anchors(item: dict, store: list[ChunkRef]) -> list[str]:
    """把一个 item 的 anchors 解析成命中的块对象。

    返回 ChunkRef 而不是 key 字符串:调用方既要 key(做 gold 集合),
    也要 doc(反推 relevant_docs)。返回对象则两者都在,不必再
    split("#") 反解 key —— 那等于把 key 的格式约定复制了第二份,
    而且文档名里一旦出现 '#' 就会解析错。

    任一锚点命中 0 个或 >=2 个块都抛 AnchorError —— 宁可报错也不要一个
    不确定的 gold,否则整份评测数字都不可信。
    """
    anchors = item.get("anchors") or []
    gold: list[ChunkRef] = []
    problems: list[str] = []

    for anchor in anchors:
        hits = [c for c in store if anchor in c.text]
        if not hits:
            problems.append(f"找不到锚点 {anchor!r}(语料被改过?或跨块被切断了?)")
        elif len(hits) > 1:
            spans = []
            for h in hits:
                i = h.text.find(anchor)
                spans.append(f"{h.key}[{i},{i + len(anchor)}]")
            problems.append(f"锚点不唯一 {anchor!r} -> 命中 {len(hits)} 块: " + ", ".join(spans))
        elif all(h.key != hits[0].key for h in gold):
            gold.append(hits[0])

    if problems:
        raise AnchorError(f"[{item.get('id', '?')}] " + "; ".join(problems))
    return gold


def validate(items: list[dict], store: list[ChunkRef]) -> list[str]:
    """校验整个评测集,返回错误列表(空列表 = 全部通过)。

    两类检查:
    1. 锚点能唯一解析 —— 否则 gold 不确定,所有指标不可信
    2. 锚点解析出的文档集合 == 手写的 relevant_docs —— 否则文件级的
       run_eval.py 和块级指标各报各的数,矛盾无人发现
    """
    errors = []
    for item in items:
        qid = item.get("id","?")
        declared = set(item.get("relevant_docs") or [])

        if not item.get("anchors"):
            if declared:
                errors.append(
                    f"[{qid}] 没有锚点却标了 relevant_docs = {sorted(declared)}"
                    f" — — 拒答题不该有 gold， 标注自相矛盾"
                )
            continue

        try:
            refs = resolve_anchors(item, store)
        except AnchorError as exc:
            errors.append(str(exc))
            continue

        derived = {c.doc for c in refs}
        if declared != derived:
            errors.append(
                f"[{qid}] 锚点解析出的文档集合与 hand_label 不一致"
                f"hand_label = {sorted(declared)}"
                f"derived = {sorted(derived)}"
            )
    return errors

def safe_zone_hint() -> str:
    """固定分块下锚点安全区的提示,校验失败时打印。

    为什么是 [overlap, chunk_size - overlap):
    chunk i 覆盖 [d*i, d*i+S),d = S-O。一个区间同时被 i 和 i+1 命中,
    当且仅当它整个落在 chunk i 的最后 O 个字符里;同时被 i 和 i-1 命中,
    当且仅当整个落在最前 O 个字符里。两头挖掉即得安全区。
    首块无前驱、末块无后继,所以两端各放宽一次。
    """
    from config import CHUNK_MODE, CHUNK_SIZE, CHUNK_OVERLAP

    if CHUNK_MODE != "fixed":
        return ""  # 结构分块没有这个简单关系,不提示

    lo, hi = CHUNK_OVERLAP, CHUNK_SIZE - CHUNK_OVERLAP
    if hi <= lo:
        return (
            f"\n警告:CHUNK_SIZE={CHUNK_SIZE} 而 CHUNK_OVERLAP={CHUNK_OVERLAP},"
            f"安全区宽度 {hi - lo} —— 任何锚点都不可能唯一,必须把 overlap 调小。"
        )
    return (
        f"\n提示:CHUNK_MODE=fixed,size={CHUNK_SIZE},overlap={CHUNK_OVERLAP} 时,"
        f"锚点要完全落在目标块偏移 [{lo},{hi}) 内"
        f"(首块下界可到 0,末块上界可到 {CHUNK_SIZE})。"
        f"改法:向左或向右扩到跨出重叠区;两头都跨不过去就换同块安全区内的另一句。"
    )


def main() -> None:
    from db import collection

    if not EVAL_SET_PATH.exists():
        print(f"评测集不存在: {EVAL_SET_PATH}")
        sys.exit(1)

    items = load_eval_set()
    store = load_store(collection)
    print(f"评测集 {len(items)} 条,语料 {len(store)} 块\n")

    errors = validate(items, store)
    if errors:
        print(f"锚点校验失败 {len(errors)} 条:")
        for e in errors:
            print("  " + e)
        print(safe_zone_hint())
        sys.exit(1)

    # 顺带统计:gold 跨块的多块题有多少,以及每类的分布
    by_cat: dict[str, int] = {}
    multi = 0
    for item in items:
        by_cat[item.get("category", "?")] = by_cat.get(item.get("category", "?"), 0) + 1
        if item.get("anchors") and len(resolve_anchors(item, store)) > 1:
            multi += 1

    print("锚点校验通过")
    print("分类分布: " + ", ".join(f"{c}={n}" for c, n in sorted(by_cat.items())))
    print(f"gold 跨多块的题目: {multi}")


if __name__ == "__main__":
    main()
