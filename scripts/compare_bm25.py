"""手写 BM25 vs rank_bm25.BM25Okapi 对拍 —— README 里"手写而非调包"这句话的证据。

对拍为什么是"严格相等"而不是"差不多"
----------------------------------
两边算的是同一个闭式公式,没有任何近似,所以偏差只应该是浮点累加误差(1e-15 量级)。
判据卡在 1e-9 = 留 6 个数量级余量,同时任何"理解错误级"的偏差都跑不掉 ——
IDF 的负数约定、长度归一、词频饱和,哪一处和参考实现不一致都会立刻超差。
如果放宽成"top-10 大致相同",这类错误会被排序的健壮性掩盖,对拍就退化成走过场。

对拍必须同时满足三条,少一条结论都不成立
------------------------------------
  1. 同语料   —— 喂同一个 docs 列表
  2. 同分词器 —— rank_bm25 接受**预分词**的 list[list[str]],所以要把切好的词喂给它,
                 而不是让它用自己的默认分词器。分词不一致的话,比的是分词不是打分
  3. 同参数   —— k1/b/epsilon 从 retrieval/bm25.py 里 import,不在这里写死。
                 否则改了实现忘了改这里,对拍会悄悄失去意义

语料用真实库而不是玩具数据
------------------------
玩具数据上对了不代表真实文本上对。真实语料才会暴露分词边界、空文档、
查询词不在词表、超长块、LaTeX 碎片 —— 这些是打分函数的边界条件。

用法
----
    python scripts/compare_bm25.py                    # 跑全部查询
    python scripts/compare_bm25.py --query Adam       # 只跑匹配的查询
    python scripts/compare_bm25.py --k 10             # 改名次一致性检查的 k
    python scripts/compare_bm25.py --tol 1e-6         # 放宽判据(排查时用,别当常态)

退出码:0 = 通过,1 = 有查询超差。可直接当断言用。

依赖:rank_bm25 是**仅开发用**依赖,只被这个脚本引用,不进 requirements.txt。
"""

import argparse
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 判据见模块 docstring。想改这个数之前先解释为什么 1e-9 不够用。
TOL = 1e-9
TOP_K = 20

# 这批查询覆盖的不是"常见问法",是**打分的边界条件**。每条都标了它在测什么 ——
# 少了任意一条,对应的那条代码分支就没有证据。
QUERIES: list[tuple[str, str]] = [
    ("什么是反向传播", "普通中文问句"),
    ("Xavier 初始化的方差怎么定", "英文专名混中文"),
    ("Adam 用什么估计梯度的动量", "缩写词"),
    ("为什么深层网络训练不动了", "paraphrase 换说法"),
    ("WordPiece 挑合并对和 BPE 有什么不同", "两个英文术语同现"),
    ("注意力掩码是干什么的", "高频主题词"),
    ("糖醋排骨怎么做", "OOD 期望几乎全 0"),
    ("zzzqqxx", "词完全不在词表"),
    ("", "空查询"),
    ("的", "超高频停用词"),
]


def _w(s: str) -> int:
    """终端显示宽度。中日韩全角字符占 2 列,`{s:20s}` 按码点数补齐会错位。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, width: int, right: bool = False) -> str:
    sp = " " * max(0, width - _w(s))
    return sp + s if right else s + sp


def load_impl():
    """导入被测实现。文件还没落地时给人话,而不是甩 traceback。"""
    try:
        from retrieval.bm25 import B, EPSILON, K1, BM25Index, tokenize
    except ImportError as e:
        sys.exit(f"导入 retrieval.bm25 失败:{e}\n"
                 f"这个脚本是它的验收工具,先把 retrieval/bm25.py 落地再跑。")
    return K1, B, EPSILON, BM25Index, tokenize


def load_ref():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit("未安装 rank_bm25。它是仅开发用依赖(对拍基线),不在 requirements.txt 里:\n"
                 "    pip install rank_bm25")
    return BM25Okapi


def print_struct(ours, ref, k1: float, b: float, epsilon: float) -> None:
    rows = [
        ("块数", float(ours.n), float(ref.corpus_size), "{:.0f}"),
        ("avgdl", ours.avgdl, float(ref.avgdl), "{:.4f}"),
        ("词表大小", float(len(ours.idf)), float(len(ref.idf)), "{:.0f}"),
        # avg_idf 是 IDF 负值修正的基准:eps = epsilon * avg_idf。
        # 两边都必须是修正**前**的平均值 —— 用修正后的 sum(idf.values()) 去比会差
        # 4e-4 量级(负值被换成了正数),那是拿错了量,不是实现错了。
        ("avg_idf", ours.average_idf, float(ref.average_idf), "{:.6f}"),
    ]
    print(f"\n--- 结构量 (k1={k1} b={b} epsilon={epsilon},两边同源) ---")
    print(f"  {_pad('', 12)} {_pad('手写', 14, True)} {_pad('rank_bm25', 14, True)}  "
          f"{'最大差':>10}")
    for name, a, bb, fmt in rows:
        print(f"  {_pad(name, 12)} {_pad(fmt.format(a), 14, True)} "
              f"{_pad(fmt.format(bb), 14, True)}  {abs(a - bb):10.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="手写 BM25 与 rank_bm25.BM25Okapi 对拍")
    ap.add_argument("--query", help="只跑包含该子串的查询")
    ap.add_argument("--k", type=int, default=TOP_K,
                    help=f"名次一致性检查的 k(默认 {TOP_K})")
    ap.add_argument("--tol", type=float, default=TOL,
                    help=f"最大绝对误差判据(默认 {TOL:g})")
    args = ap.parse_args()

    queries = [(q, note) for q, note in QUERIES if not args.query or args.query in q]
    if not queries:
        sys.exit(f"--query {args.query!r} 没匹配到任何查询。\n"
                 f"可用查询:{' / '.join(repr(q) for q, _ in QUERIES)}")

    k1, b, epsilon, BM25Index, tokenize = load_impl()
    BM25Okapi = load_ref()

    from anchors import load_store  # noqa: E402  延迟:import 就会建 Chroma client
    from db import collection  # noqa: E402

    store = load_store(collection)
    if not store:
        sys.exit("语料为空。先 python ingest.py。")

    docs = [c.text for c in store]
    print(f"语料 {len(docs)} 块,用项目自己的 tokenize 预分词 ...")
    tokenized = [tokenize(d) for d in docs]
    vocab = {t for d in tokenized for t in d}
    avg_len = sum(len(d) for d in tokenized) / len(tokenized)
    print(f"  词表 {len(vocab)} 个 token,平均块长 {avg_len:.1f} token")

    ours = BM25Index(docs)
    # 参数从被测实现里读 —— 两边同源,否则比的是两套参数不是两套代码
    ref = BM25Okapi(tokenized, k1=k1, b=b, epsilon=epsilon)

    print_struct(ours, ref, k1, b, epsilon)

    print(f"\n--- 逐查询({len(queries)} 条)---")
    print(f"  {_pad('查询', 38)} {_pad('测的是', 22)} "
          f"{_pad('最大绝对误差', 14, True)} {_pad('非零块', 10, True)} "
          f"{_pad(f'top-{args.k} 重合', 12, True)}")

    worst, worst_q = 0.0, ""
    failed: list[str] = []
    for q, note in queries:
        a = ours.get_scores(q)
        # 预分词后喂给参考实现 —— 保证两边分词完全一致
        ref_scores = [float(x) for x in ref.get_scores(tokenize(q))]

        diff = max((abs(x - y) for x, y in zip(a, ref_scores)), default=0.0)
        if diff > worst:
            worst, worst_q = diff, q
        if diff > args.tol:
            failed.append(q)

        # 名次一致性:ours 走 search()(含 "分数>0" 过滤 + 稳定排序),
        # 参考走 get_top_n()(np.argsort,不稳定)。分数相同时集合仍可能不同,
        # 所以这一列只报不改判 —— 真正的判据是上一条的分数差。
        mine_k = {i for i, _ in ours.search(q, k=args.k)}
        theirs_k = set(ref.get_top_n(tokenize(q), list(range(ours.n)), n=args.k))
        overlap = (len(mine_k & theirs_k) / args.k if args.k else 1.0)
        nz = sum(1 for x in a if x > 0)

        print(f"  {_pad(q if q else '\"\"', 38)} {_pad(note, 22)} "
              f"{_pad(f'{diff:.3e}', 14, True)} {_pad(f'{nz}/{len(a)}', 10, True)} "
              f"{_pad(f'{overlap:.0%}', 12, True)}")

    print()
    print(f"最大绝对误差 {worst:.3e}" + (f"(来自 {worst_q!r})" if worst_q else "")
          + f"   判据 < {args.tol:g}")
    print("注:top-k 重合率 <100% 只可能来自**并列分的排序差异**(np.argsort 不稳定,"
          "\n    且 search() 会滤掉 0 分块),不是分数算错 —— 分数那一列已经排除了这个可能。")

    if failed:
        print(f"\nFAIL —— {len(failed)} 条查询超差,实现与参考约定不一致。"
              f"\n先查 _build_idf 的 IDF 负值处理(df > N/2 时 log 变负)是否与 "
              f"rank_bm25 一样替换成 epsilon * avg_idf。")
        return 1

    print(f"\nPASS —— 手写实现与 rank_bm25.BM25Okapi 在 {len(docs)} 块真实语料、"
          f"{len(queries)} 条查询上一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
