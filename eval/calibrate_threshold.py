"""拒答阈值标定。

【要解决什么问题】
闸门决定"这个问题在不在资料射程内"。阈值必须和线上闸门同口径标定,
否则调出来的数和实际行为对不上。

【两种信号】
config.GATE_SIGNAL 决定闸门读哪个数:
    "vector"  候选里最小的 vector_distance > DISTANCE_THRESHOLD  -> 拒答
    "rerank"  候选里最大的 rerank_score < RERANK_THRESHOLD       -> 拒答
两者方向相反:距离越小越好,rerank 分数越大越好。--signal 默认跟随 config,
也可以显式传另一个值,在同一份候选上做对比(hybrid_rerank 同时提供两种信号)。

【口径 —— 本脚本最需要交代的一件事】
    正样本 = top-K 里找得到 gold 的题   —— 被拒 = 误拒(该答的答不出)
    负样本 = OOD 题                     —— 放行 = 误答(不该答的答了)
正样本判定用 chunk key 命中,**不要求** gold 块带 vector_distance:混合检索的
gold 可能是 BM25 单独召回的(vd=None),按旧写法会被误记成"没进 top-K",
正样本从 59 缩到 52,和线上行为对不上。

注意正样本量的是闸门输入,而**不是** gold 块自己的分数。因为闸门的职责只是
"这问题在不在射程内",不是"top-1 对不对" —— 就算 top-1 是个非 gold 块,
只要它信号够强,闸门就该放行,后面由排序去解决。best_gold 单独报,是因为
Phase 4 改成逐块过滤时起作用的是它。

【不做的两件事】
  - 不自动改 config.py。阈值是产品决策(宁可误拒还是宁可误答),数字摆出来让人定。
  - 不报准确率。类别不平衡(n_pos >> n_neg)时准确率恒等于多数类占比,零信息量。

【统计诚实性】
OOD 只有 17 条。若观察到 1 次误答,FPR 点估计是 5.9%,但 95% Wilson 区间约
[1.0%, 18.8%] —— 上界比点估计宽 3 倍。样本这么少,阈值只能定到"量级正确",
不要假装能定到小数点后两位。所以下面每个 FPR 都带 Wilson 区间(见 wilson 函数)。
阈值只在 dev 集上标,holdout 集不参与调参。

【用法】
    python eval/calibrate_threshold.py                          # 跟随 config 的检索器和信号
    python eval/calibrate_threshold.py --signal vector          # 同一份候选,换信号对比
    python eval/calibrate_threshold.py --retriever vector --k 20
    python eval/calibrate_threshold.py --json eval/results/threshold.json
"""


import argparse
import json
import math
import sys
from pathlib import Path

# 项目里 eval/ 和 scripts/ 是平铺导入的(anchors.py 直接 import db),不设包。
# 所以脚本必须自己把这两个目录塞进 sys.path,否则从仓库根目录跑不起来。
ROOT = Path(__file__).resolve().parent.parent          # eval/ 的上一级 = 仓库根
for _p in (str(ROOT), str(ROOT / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anchors import load_eval_set, load_store, resolve_anchors   # noqa: E402
from config import (COARSE_K, DISTANCE_THRESHOLD, RETRIEVER,
                    GATE_SIGNAL, RERANK_THRESHOLD)               # noqa: E402
from db import collection                                        # noqa: E402
from retrieval.pipeline import build_retriever                   # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """成功比例的 Wilson 置信区间。

    为什么不用中学那个 p ± z*sqrt(p(1-p)/n):
      n 小、p 又接近 0 或 1 时它会算出负数下界或超过 1 的上界。17 个样本、
      0 次误答会给出 [0, 0] —— 等于宣称"真实误答率一定是 0",这显然错了。
      Wilson 是解一个二次不等式得到的,端点天然落在 [0,1] 内,小样本下也合理。

    参数 k=观察到的成功次数, n=总次数。返回 (下界, 上界)。
    """
    if n == 0:                      # 没有样本时,唯一诚实的回答是"什么都不知道"
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n                            # 二次项系数
    c = p + z * z / (2 * n)                      # 中心
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))   # 半宽
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def describe(xs: list[float]) -> str:
    """一行打印一组距离的分布。比均值+标准差有用:距离分布是偏的,且样本少时
    分位数直接告诉你"有没有重叠" —— 正负样本分布一旦重叠,再好的阈值也分不开。
    """
    if not xs:
        return "(空)"
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]   # 下标截断,防止 f=0.75 越界
    return (f"n={len(s):3d}  min={s[0]:.3f}  p25={q(.25):.3f}  "
            f"中位={q(.50):.3f}  p75={q(.75):.3f}  max={s[-1]:.3f}")

def signal_values(chunks, signal: str) -> list[float]:
    """取候选上的闸门信号。两个方向相反,后面统一用 accepts() 判断。"""
    field = 'rerank_score' if signal == 'rerank' else 'vector_distance'
    return [getattr(c, field) for c in chunks if getattr(c,field) is not None]

def gate_value(vals: list[float], sinal: str) -> float | None:
    """闸门实际读的那个数:距离取最小,rerank 分数取最大.没有信号返回None."""
    if not vals:
        return None
    return min(vals) if sinal == 'vector' else max(vals)

def accepts(value: float, t: float, signal: str) -> bool:
    """阈值 t 下是否放行。距离越小越好,rerank 分数越大越好 —— 方向在这里统一。"""
    return value >= t if signal == 'rerank' else value <= t

def candidate_thresholds(values: list[float], current: float, signal: str) -> list[float]:
    """扫描点 = 相邻样本中点 + 现行阈值.整数刻度可能整段跨过分界,让人误以为"怎么调都一样"."""
    lo,hi = (0.0,1.0) if signal == "rerank" else (0.0,2.0)
    pts = sorted({round(x,4) for x in values})
    cands = {lo,current} | {round((a+b)/2,4) for a,b in zip (pts,pts[1:])}
    return [t for t in sorted(cands) if lo <= t <= hi]


def main() -> None:
    ap = argparse.ArgumentParser(description="拒答阈值标定")
    ap.add_argument("--retriever", default=RETRIEVER, help=f"默认 config.RETRIEVER={RETRIEVER}")
    ap.add_argument("--signal", choices=("vector", "rerank"), default=GATE_SIGNAL,help = f"闸门信号,默认 config.GATE_SIGNAL={GATE_SIGNAL}")
    ap.add_argument("--k", type=int, default=COARSE_K, help=f"默认 config.COARSE_K={COARSE_K}")
    ap.add_argument("--json", help="把原始信号值和扫描表落盘(写进报告当证据用)")
    args = ap.parse_args()

    items = load_eval_set()
    store = load_store(collection)
    retriever = build_retriever(args.retriever)

    current = RERANK_THRESHOLD if args.signal == "rerank" else DISTANCE_THRESHOLD
    rule = ">= 阈值 放行" if args.signal == "rerank" else "<= 阈值 放行"
    print(f"检索器 {args.retriever}   信号 {args.signal}({rule})   k={args.k}")
    print(f"语料 {len(store)} 块   评测集 {len(items)} 条   现行阈值 {current}\n")

    pos_top1: list[float] = []  # 正样本:闸门实际读的那个数
    pos_best_gold: list[float] = []  # gold 块自己的信号值(距离取最小,rerank 取最大)
    missed: list[str] = []            # top-K 里根本没 gold 的题 —— 与阈值无关,是排序问题
    neg: dict[str, list[float]] = {}  # 负样本按子类分开存:ood-pseudo 才是真正的压力测试

    for it in items:
        chunks = retriever.retrieve(it["query"], k=args.k)
        if not chunks:
            continue

        # 与 rag.py._should_refuse 同口径:距离取候选最小值,rerank 取最大值。
        # 不能取 chunks[0] 的分数 —— 融合后 top-1 可能来自另一路,那个字段是 None,
        # 整道题会被静默跳过,正负样本一起缩水。
        vals = signal_values(chunks, args.signal)
        d1 = gate_value(vals, args.signal)
        if d1 is None:
            print(f"  ! {it['id']} 在当前检索器上没有 {args.signal} 信号,已跳过")
            continue

        if it["category"].startswith("ood"):
            neg.setdefault(it["category"], []).append(d1)
            continue

        if not it["anchors"]:         # 非 OOD 但没锚点 = 数据有问题,不当正样本用
            continue

        # resolve_anchors 命中 0 或 >=2 块会抛 AnchorError。能走到这里说明eval/anchors.py 的校验已经过了,所以直接取。
        gold = {c.key for c in resolve_anchors(it, store)}

        # 正样本判定用 key 命中,不再要求 gold 块有 vector_distance --
        # hybrid 的 gold 可能是 BM25 单独召回的(vd=None),旧写法会把它记成"没进 top-K",正样本从 59 缩到 52,两边就对不上了。
        if any(c.key in gold for c in chunks):
            pos_top1.append(d1)
            gv = [getattr(c,"rerank_score" if args.signal == "rerank" else "vector_distance") for c in chunks if c.key in gold]
            gv = [x for x in gv if x is not None]
            if gv:
                pos_best_gold.append(max(gv) if args.signal == "rerank" else min(gv))
        else:
            # gold 连 top-K 都没进。这种题调阈值救不了,只能靠 P4 的结构分块去捞。
            # 单独列出来,免得把召回问题算进阈值账上。
            missed.append(it["id"])

    n_pos = len(pos_top1)
    n_neg = sum(len(v) for v in neg.values())
    if n_pos == 0 or n_neg == 0:
        print(f"正样本 {n_pos} 条、负样本 {n_neg} 条 —— 至少一边为空,标定无意义。")
        print("检查评测集是否加载成功,或 --k 是否太小。")
        return

    print("─" * 74)
    print(f"{args.signal} 信号分布")
    print("─" * 74)
    print(f"  正样本 闸门输入      {describe(pos_top1)}")
    print(f"  正样本 best_gold  {describe(pos_best_gold)}")
    for cat in sorted(neg):
        print(f"  负样本 {cat:14s} {describe(neg[cat])}")
    if missed:
        print(f"\n  ! {len(missed)} 题在 top-{args.k} 里没有 gold(与阈值无关):{missed}")
    print()
    print("  看点:正样本 max 和负样本 min 之间如果贴着,说明这两个集合在当前检索器下")
    print("  本来就分不开,阈值只能取折中;如果离得远,中间随便挑一个都行。\n")

    # 扫描候选阈值。取相邻信号值的中点,而不是只试 0.1/0.2 这种整数刻度 ——
    # 整数刻度可能整段跨过正负样本的分界,让人误以为"怎么调都一样"。
    all_d = pos_top1 + [d for v in neg.values() for d in v]
    cands = candidate_thresholds(all_d, current, args.signal)

    print("─" * 74)
    print("阈值扫描")
    print("─" * 74)
    print(f"  误拒 = 该答的题被卡掉   误答 = 不该答的题混进来   ({rule})")
    print(f"  {'阈值':>7} {'TPR':>6} {'误拒':>4} {'FPR':>6} {'误答':>4} {'J':>7}   FPR 95% CI")
    best = None
    for t in cands:
        tp = sum(1 for d in pos_top1 if accepts(d, t, args.signal))
        fp = sum(1 for v in neg.values() for d in v if accepts(d, t, args.signal))
        tpr, fpr = tp / n_pos, fp / n_neg
        j = tpr - fpr                     # Youden's J:同时惩罚误拒和误答
        lo, hi = wilson(fp, n_neg)
        mark = "  <- 现行" if abs(t - current) < 1e-9 else ""
        print(f"  {t:7.3f} {tpr:6.3f} {n_pos - tp:4d} {fpr:6.3f} {fp:4d} {j:7.3f}"
              f"   [{lo:.3f}, {hi:.3f}]{mark}")
        if best is None or j > best[1]:
            best = (t, j, tpr, fpr, tp, fp)

    print()
    print("─" * 74)
    print("结论")
    print("─" * 74)
    t, j, tpr, fpr, tp, fp = best
    lo, hi = wilson(fp, n_neg)
    print(f"  J 最大处:阈值 {t:.3f}   TPR={tpr:.3f}(误拒 {n_pos - tp})   "
          f"FPR={fpr:.3f}(误答 {fp})   95% CI [{lo:.3f}, {hi:.3f}]")

    cur_tp = sum(1 for d in pos_top1 if accepts(d, current, args.signal))
    cur_fp = sum(1 for v in neg.values() for d in v if accepts(d, current, args.signal))
    clo, chi = wilson(cur_fp, n_neg)
    print(f"  现行 {current}({args.signal}): TPR={cur_tp / n_pos:.3f}(误拒 {n_pos - cur_tp})   "
          f"FPR={cur_fp / n_neg:.3f}(误答 {cur_fp})   95% CI [{clo:.3f}, {chi:.3f}]")

    print(f"\n  J 只给了个折中点,不等于最优。FPR 的样本量只有 {n_neg},区间上界 {hi:.3f}")
    print(f"  意味着真实误答率可能比点估计高好几倍 —— 想收紧就往下调,代价是误拒变多。")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "retriever": args.retriever, "signal": args.signal,
            "k": args.k, "current": current,
            "n_pos": n_pos, "n_neg": n_neg,
            "pos_top1": pos_top1, "pos_best_gold": pos_best_gold,
            "neg": neg, "missed": missed,
            # 注意两层 for 的变量名:外层 v 遍历 neg 的各个子类列表,内层 d 才是信号值。
            # 这里只能重算一遍扫描表,不能直接复用上面的 best —— best 只存了 J 最大的
            # 那一行,而落盘要的是整张表(报告里要把曲线画出来)。
            "sweep": [{"t": t2,
                       "tpr": sum(1 for d in pos_top1 if accepts(d, t2, args.signal)) / n_pos,
                       "fpr": sum(1 for v in neg.values() for d in v if accepts(d, t2, args.signal)) / n_neg}
                      for t2 in cands],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  原始距离与扫描表 -> {out}")


if __name__ == "__main__":
    main()
