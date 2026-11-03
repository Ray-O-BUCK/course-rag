"""引用校验的离线评估。

只测确定性部分：
  - citation precision：回答里引用的 key，有多少真在本次上下文里
  - citation coverage：有回答的题里，多少至少带了一个有效引用
  - no-citation / invalid-citation：无引用、含无效引用的回答占比

不测语义忠实度（"这句话是否被引用块支持"），那需要 LLM judge 或人工标注。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "eval")):
    if p not in sys.path:
        sys.path.insert(0,p)

from anchors import load_eval_set # noqa: E402
from rag import answer              # noqa: E402
def main() -> None:
    ap = argparse.ArgumentParser(description = "引用校验离线评估")
    ap.add_argument("--limit", type = int, default = 0, help = "只跑前 N 条, 调试用")
    ap.add_argument("--json", help = "逐题结果落盘")
    ap.add_argument("--eval-set", default=str(ROOT / "eval" / "eval_set_chunks.jsonl"),
                    help="评测集 JSONL(默认 dev 集)")
    args = ap.parse_args()

    items = load_eval_set(Path(args.eval_set))
    if args.limit:
        items = items[: args.limit]

    rows = []
    for i, it in enumerate(items,1):
        result = answer(it["query"])
        check = result.get("citation_check") or {
            "cited": [], "valid": [], "invalid": [], "no_citation": True,
        }
        row = {
            "id": it["id"],
            "eval_set": Path(args.eval_set).stem,
            "category": it.get("category", "?"),
            "query": it["query"],
            "refused": not result["sources"],
            **check,
        }
        rows.append(row)
        flag = "refused" if row["refused"] else (
            f"cited = {len(check['cited'])} valid = {len(check['valid'])}"
            f" invalid={len(check['invalid'])}"
        )
        print(f"[{i:2d}/{len(items)}] {it['id']} {flag}")

    answered = [r for r in rows if not r["refused"]]
    cited = sum(len(r["cited"]) for r in answered)
    valid = sum(len(r["valid"]) for r in answered)
    with_valid = sum(1 for r in answered if r["valid"])
    no_citation = sum(1 for r in answered if r["no_citation"])
    with_invalid = sum(1 for r in answered if r["invalid"])

    print("\n" + "─" * 60)
    print(f"总题数 {len(rows)}，回答 {len(answered)}，拒答 {len(rows) - len(answered)}")
    if answered:
        print(f"citation precision: {valid}/{cited} = {valid / cited:.3f}" if cited else "citation precision: 无引用")
        print(f"citation coverage : {with_valid}/{len(answered)} = {with_valid / len(answered):.3f}")
        print(f"no-citation       : {no_citation}/{len(answered)} = {no_citation / len(answered):.3f}")
        print(f"invalid-citation  : {with_invalid}/{len(answered)} = {with_invalid / len(answered):.3f}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n逐题结果 -> {out}")


if __name__ == "__main__":
    main()