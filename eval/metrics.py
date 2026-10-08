"""检索指标,从零实现。

约定:所有函数接收
- ranked: 检索结果按相关度降序的 key 列表(ranked[0] 是第 1 名)
- gold:   该 query 的正确答案 key 集合(可以多个)
- k:      只看前 k 名

gold 为空的样本(拒答题)检索指标无定义,统一返回 nan,由 mean() 跳过。
"""

import math


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Recall@K:前 k 名覆盖了多少比例的正确块。

    注意是"覆盖比例"不是"命中与否"。一个 query 有 3 个正确块、top-5 命中 2 个,
    得分是 2/3 而不是 1。文件级评测(line 里的 relevant_docs)把这件事粗化成了
    "命中了没有" —— 3 个正确块里出现 1 个就算满分,会掩盖真实差距。
    """
    if not gold:
        return float("nan")
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str], k: int | None = None) -> float:
    """RR:第一个正确块的倒数名次。第 1 名命中 = 1.0,第 2 名 = 0.5,第 4 名 = 0.25。

    k=None 扫全表(纯 RR);给了 k 就是 MRR@k —— 前 k 名没命中直接记 0。
    报告里写 "MRR@10" 就必须传 k=10,否则第 15 名的命中会被算成 1/15 ≈ 0.067。
    用 `k is not None` 而不是 `if k`:k=0 时后者会被当成"没传 k"。

    gold 为空(拒答题)返回 nan,和 recall/ndcg 保持一致 —— 否则拒答题会以
    0 分混进平均,把 MRR 系统性拉低。
    """
    if not gold:
        return float("nan")
    for i, key in enumerate(ranked[:k] if k is not None else ranked):
        if key in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """nDCG@K:二值相关度(命中记 1,否则 0)的归一化折损累计增益。

    DCG  = sum(1 / log2(位置 + 1)),只累加前 k 名里命中的位置(位置从 1 起)
    折扣的含义:第 1 名比第 8 名值钱,且"越靠后越不值钱"是凸的
    IDCG = 理想排序下的 DCG,即正确块全部排在最前面

    除以 IDCG 是为了归一化:不同 query 的正确块数量不同(1 个 vs 3 个),
    不归一化就没法跨 query 求平均。

    和 RR 的分工:RR 只看第一个命中,nDCG 看所有正确块排得好不好。
    多个正确块都重要时看 nDCG。
    """
    if not gold:
        return float("nan")
    dcg = 0.0
    for i, key in enumerate(ranked[:k]):
        if key in gold:
            dcg += 1.0 / math.log2(i + 2)  # i 从 0 起,位置 = i+1,折扣 = log2(位置+1)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else float("nan")


def mean(values: list[float]) -> float:
    """跳过 nan 求平均(nan 来自 gold 为空的拒答样本)。"""
    real = [v for v in values if not math.isnan(v)]
    return sum(real) / len(real) if real else float("nan")


if __name__ == "__main__":
    # 手算得出的期望值,改了这个文件就跑一遍:eval/metrics.py
    assert recall_at_k(["a", "b", "c"], {"b"}, 1) == 0.0
    assert recall_at_k(["a", "b", "c"], {"b"}, 2) == 1.0
    # 2 个正确块只命中 1 个 -> 0.5(文件级评测会把这种差距抹平)
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5
    assert math.isnan(recall_at_k(["a"], set(), 1))

    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    assert ndcg_at_k(["a", "x"], {"a"}, 2) == 1.0           # 理想排序
    assert abs(ndcg_at_k(["x", "a"], {"a"}, 2) - 1 / math.log2(3)) < 1e-12
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    # 两个正确块都在前两名 -> 满分,即使顺序反了
    assert ndcg_at_k(["b", "a", "x"], {"a", "b"}, 3) == 1.0

    assert mean([1.0, float("nan"), 0.0]) == 0.5
    assert math.isnan(mean([float("nan")]))

    assert math.isnan(reciprocal_rank(["a"], set()))  # 无 gold -> 无定义
    assert reciprocal_rank(["a", "b", "c"], {"c"}, k=2) == 0.0  # MRR@2 没命中记 0
    assert abs(reciprocal_rank(["a", "b", "c"], {"c"}, k=3) - 1 / 3) < 1e-12
    assert abs(reciprocal_rank(["a", "b", "c"], {"c"}) - 1 / 3) < 1e-12  # k=None 扫全表

    print("metrics.py 自检通过")
