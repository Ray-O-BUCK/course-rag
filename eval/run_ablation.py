"""跑检索消融,把每次检索的名次落盘成 JSONL。

为什么落盘,而不是当场算完指标打印
----------------------------------
名次序列是"检索实际发生了什么",指标是"怎么给这个结果打分"。后者会变
(加 Recall@20?改分级相关?按 category 拆?),前者不会。分开之后,重算指标
只读日志,不必重跑模型 —— vector 基线 34 条要几十秒,上了 reranker 是几分钟。

日志按行追加,多次运行共存,靠 run_id 区分。

为什么要 corpus_tag
------------------
Phase 4 会重切分块,块 id 全变。跨语料版本的指标不可比(README 踩坑 #7),
所以每行都记语料指纹,report.py 只对比 corpus_tag 相同的行。

用法
----
    python eval/run_ablation.py                    # 跑 CONFIGS 里的全部配置
    python eval/run_ablation.py --config vector    # 只跑一个
    python eval/run_ablation.py --k 20             # 覆盖 COARSE_K
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anchors import load_eval_set, load_store, resolve_anchors  # noqa: E402
from config import COARSE_K  # noqa: E402
from metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from retrieval.pipeline import build_retriever  # noqa: E402

RESULTS_PATH = EVAL_DIR / "results" / "ablation.jsonl"

# 每推进一个 Phase 往这里加一个名字。这些名字同时是报告里的列名,
# 定了就别改 —— 改了等于历史上所有日志的 config 列失效。
# bm25 是混合检索的对照组:不加它就只能证明"hybrid 比 vector 好",
# 证明不了"融合比最强的单路好"。
CONFIGS: tuple[str, ...] = ("vector", "bm25", "hybrid","hybrid_rerank")
# 实验配置不放进默认全量跑:它会调 LLM 改写,不适合每次默认执行
EXTRA_CONFIGS: tuple[str, ...] = ("hybrid_rerank_rewrite",)

# 报告要算的名次截断点。超过实跑 k 的会被丢掉并提示,不静默算出个假的 MRR@10。
REPORT_KS: tuple[int, ...] = (1, 3, 5, 10, 20)


def corpus_tag(store) -> str:
    """语料 + 分块方案的指纹。

    哈希块 key 和块内容,而不是用文件 mtime 或块数:重切分块后块数碰巧不变时,
    只有内容哈希会变。12 位十六进制足够区分本项目规模的语料版本。
    """
    h = hashlib.sha256()
    for c in sorted(store, key=lambda c: c.key):
        h.update(c.key.encode("utf-8"))
        h.update(b"\x00")
        h.update(c.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:12]


def collect(configs: list[str], k: int, eval_path: Path) -> tuple[str, str, dict, int]:
    """跑检索并落盘。返回 (run_id, corpus_tag, {config: [(item, ranked, gold)]}, 写入行数)。"""
    from db import collection  # 延迟到调用时再建 Chroma client

    items = load_eval_set(eval_path)
    eval_name = eval_path.stem
    store = load_store(collection)
    tag = corpus_tag(store)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {}
    n_lines = 0
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        for name in configs:
            retriever = build_retriever(name)
            rows = []
            for item in items:
                gold = {c.key for c in resolve_anchors(item, store)}
                chunks = retriever.retrieve(item["query"], k=k)
                rows.append((item, [c.key for c in chunks], gold))
                for c in chunks:
                    f.write(json.dumps({
                        "run_id": run_id,
                        "eval_set": eval_name,
                        "corpus_tag": tag,
                        "config": name,
                        "qid": item["id"],
                        "category": item.get("category", "?"),
                        "rank": c.rank,
                        "key": c.key,
                        # relevant 冗余存一份(能由 gold 推出来),好处是
                        # awk/grep 直接能看,不必再加载语料反查锚点
                        "relevant": c.key in gold,
                        "n_gold": len(gold),
                        "vdist": c.vector_distance,
                        "bm25": c.bm25_score,
                        "rr": c.rerank_score,
                    }, ensure_ascii=False) + "\n")
                    n_lines += 1
            results[name] = rows
    return run_id, tag, results, n_lines


def _stats(rows: list, ks: list[int]) -> dict[str, float]:
    return {
        **{f"R@{n}": mean([recall_at_k(r, g, n) for _, r, g in rows]) for n in ks},
        **({"MRR@10": mean([reciprocal_rank(r, g, k=10) for _, r, g in rows])}
           if 10 in ks else {}),
        **({"nDCG@10": mean([ndcg_at_k(r, g, 10) for _, r, g in rows])}
           if 10 in ks else {}),
    }


COL_W = 8  # 要容下最宽的列名 nDCG@10(7 字符)


def _fmt(v: float) -> str:
    return "-" if math.isnan(v) else f"{v:.3f}"


def _cells(s: dict, cols: list[str]) -> str:
    return "".join(f"{_fmt(s[c]):>{COL_W}}" for c in cols)


def print_summary(tag: str, results: dict, ks: list[int]) -> None:
    cols = [f"R@{n}" for n in ks] + (["MRR@10", "nDCG@10"] if 10 in ks else [])
    head = "".join(f"{c:>{COL_W}}" for c in cols)

    print(f"\n语料指纹 {tag}   (跨指纹的数字不可比,别把两次跑的结果并排放)")
    print(f"{'config':<16}{head}{'n':>5}")
    for name, rows in results.items():
        n = sum(1 for _, _, g in rows if g)  # 有 gold 的题才算得进指标,拒答题是 nan
        print(f"{name:<16}{_cells(_stats(rows, ks), cols)}{n:>5}")

    print("\n按类别(看清 n —— 4 条题上的 0.05 差距是噪声,不是结论)")
    cats = sorted({item.get("category", "?") for item, _, _ in next(iter(results.values()))})
    print(f"{'category':<16}{'config':<16}{head}{'n':>5}")
    for cat in cats:
        for name, rows in results.items():
            sub = [r for r in rows if r[0].get("category", "?") == cat]
            n = sum(1 for _, _, g in sub if g)
            if not n:
                continue
            print(f"{cat:<16}{name:<16}{_cells(_stats(sub, ks), cols)}{n:>5}")


def main() -> None:
    ap = argparse.ArgumentParser(description="跑检索消融并落盘名次")
    ap.add_argument("--config", action="append", choices=list(CONFIGS + EXTRA_CONFIGS),
                    help=f"只跑指定配置,可重复。默认全部:{', '.join(CONFIGS)}")
    ap.add_argument("--k", type=int, default=COARSE_K, help=f"每路取多少候选(默认 COARSE_K={COARSE_K})")
    ap.add_argument("--eval-set", default=str(EVAL_DIR / "eval_set_chunks.jsonl"),
                    help="评测集 JSONL(默认 dev 集)")
    args = ap.parse_args()

    ks = [n for n in REPORT_KS if n <= args.k]
    dropped = [n for n in REPORT_KS if n > args.k]
    if dropped:
        print(f"注意:k={args.k},以下截断点超过候选数,已从报告里去掉:{dropped}")
    if 10 not in ks:
        print("注意:k<10 时 MRR@10 / nDCG@10 无意义,本次不报这两项")

    run_id, tag, results, n_lines = collect(args.config or list(CONFIGS), args.k, Path(args.eval_set))
    print(f"eval_set {args.eval_set}")
    print(f"run_id {run_id} -> {RESULTS_PATH}(追加 {n_lines} 行)")
    print_summary(tag, results, ks)


if __name__ == "__main__":
    main()
