"""RRF(Reciprocal Rank Fusion)—— 只吃名次,不吃分数。

为什么必须是名次
--------------
三路分数的量纲根本不可加:
    向量距离 l2 ∈ [0,4],越小越相关
    BM25 分数无界,越大越相关
    reranker sigmoid ∈ [0,1],越大越相关
归一化也救不了 —— BM25 的分布随查询剧变(实测:查询"的"有 873/1210 块得分非零,
"注意力掩码" 119/1210,"糖醋排骨" 0/1210),min-max 归一化会被查询自身的最值绑死。
RRF 直接扔了分数,只用名次,天然免疫这个问题。代价是丢掉了"第 1 名比第 2 名强多少"
这个信息。

k=60 在做什么
-------------
    1/(60+1) = 0.01639        第 1 名
    1/(60+20) = 0.01250       第 20 名
第 1 名和第 20 名只差 24%。这个"压平"是有意的:它让**是否出现在某一路**比
**在该路排第几**更重要。所以 RRF 本质是个"共识投票"机制,不是保序机制 ——
这直接解释了实测结果:RRF 的 top-10 召回赢过任何单路(hybrid 0.859 / BM25 0.836 /
vector 0.688),但 top-1 输给 BM25 单跑(0.375 vs 0.484)。它提高召回,不保证精度。
把 top-1 精度做上去是 Phase 3 reranker 的活。
"""

from collections import defaultdict
from dataclasses import replace

from retrieval.base import ScoredChunk

# 各路只填自己的分数字段(见 base.ScoredChunk)。融合后同一个 key 可能来自两路,
# 要合并而不是丢弃 —— run_ablation.py 会把这三个字段都写进日志,
# 留 None 就等于事后无法分析"这个块在各路各自排第几"。
_SCORE_FIELDS = ("vector_distance","bm25_score","rerank_score")

def _merge(into:ScoredChunk,other:ScoredChunk) ->ScoredChunk:
    """把 other 独有的分数字段补充进 into。返回新对象,不改入参。"""
    fill = {f:getattr(other,f) for f in _SCORE_FIELDS if getattr(into,f) is None and getattr(other,f) is not None}
    return replace(into,**fill) if fill else into

def rrf_fuse(ranked:list[list[ScoredChunk]],k:int=60,weights:list[float]|None = None) -> list[ScoredChunk]:
    """融合多路已排序结果。

        ranked   每路一个列表,各自内部已按"越靠前越相关"排好(调用方负责保证)
        k        平滑常数。调小 = 更看重头部名次,调大 = 更看重"是否出现"
        weights  每路权重,与 ranked 同序。默认等权。

        返回**并集**(不是交集):只被一路召回的块也有分,只是比两路都有的低。
        """
    if weights is None:
        weights = [1.0] * len(ranked)
    elif len(weights) != len(ranked):
        raise ValueError(f"weights 长度 {len(weights)} 与路数{len(ranked)} 不符")

    score: dict[str,float] = defaultdict(float)
    first: dict[str,ScoredChunk] = {}

    for w,lst in zip(weights,ranked):
        for rank,c in enumerate(lst,1):
            # 名字次必须**相加**:两路都排前面的块,证据是叠加的。
            # 取 max 会让"两路都认可"和"只有一路认可"打平,融合就白做了。
            score[c.key] += w / (rank + k)
            first[c.key] = _merge(first[c.key],c) if c.key in first else c

    # sorted 是稳定的,同分保持首次出现的顺序 —— 给定确定性输入,结果确定
    order = sorted(score,key=score.__getitem__,reverse=True)
    return [replace(first[key],fusion_score=score[key],rank=i) for i,key in enumerate(order,1)]
