"""预生成评测集的 query 改写缓存. 跑一次,后续消融只读缓存."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval.anchors import load_eval_set          # noqa: E402
from retrieval.rewrite import CACHE_PATH, rewrite_query  # noqa: E402

def main() -> None:
    items = load_eval_set()
    queries = sorted({it["query"] for it in items})
    print(f"共 {len(queries)} 条唯一 query,缓存 -> {CACHE_PATH}")
    for i, q in enumerate(queries, 1):
        rewritten = rewrite_query(q)
        print(f"[{i:2d}/{len(queries)}] {q}  ->  {rewritten}")


if __name__ == "__main__":
    main()
