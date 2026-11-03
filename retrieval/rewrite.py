"""查询改写实验(P5):原 query + 改写 query 各检索一次,RRF 合并。

为什么缓存
----------
LLM 改写不是确定性的。消融的前提是"同一配置重跑得到同一结果",所以第一次
改写就落盘;之后 eval 只读缓存。线上遇到新 query 仍会调 LLM 并写回缓存。

缓存位置放在 eval/results/ 下,因为这是实验产物,不进仓库;如果以后要上线,
把它挪到 data/ 并加进 .gitignore。
"""
import json
from pathlib import Path

from config import BASE_DIR, COARSE_K, LLM_MODEL, RRF_K
from llm import chat
from retrieval.base import Retriever, ScoredChunk
from retrieval.fusion import rrf_fuse

CACHE_PATH = BASE_DIR / "eval" / "results" / "rewrite_cache.json"

_PROMPT = """你是检索查询改写器。把用户问题改写成一句更适合在中文课程资料里检索的查询。

规则:
1. 保留原问题里的专有名词、缩写、数字和限定条件,不要替换成同义词。
2. 可以补上问题隐含的主题词,但不要引入原问题没有的新概念。
3. 只输出改写后的查询本身,不要回答、不要解释、不要引号或编号。

示例:
原问题:什么是过拟合
改写:过拟合 定义 原因 解决方法

原问题:{query}
改写:"""


def _load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    # 换模型后旧改写不能复用
    if data.get("model") != LLM_MODEL:
        return {}
    return data.get("rewrites", {})


def _save_cache(rewrites: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"model": LLM_MODEL, "rewrites": rewrites},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(CACHE_PATH)


def rewrite_query(query: str) -> str:
    """返回改写后的检索 query。失败/空输出时退回原 query。"""
    cache = _load_cache()
    if query in cache:
        return cache[query]

    raw = chat(
        [{"role": "user", "content": _PROMPT.format(query=query)}],
        temperature=0.0,
    )
    out = (raw or "").strip()

    if out:
        out = out.splitlines()[0].strip().strip('"“”').strip()
        for prefix in ("改写:", "改写：", "查询:", "查询："):
            if out.startswith(prefix):
                out = out[len(prefix):].strip()
                break

    if not out:
        # 模型偶尔返回空串。退回原 query,并把这个决定也缓存,
        # 否则同一 query 每次重跑都会再打一次 API。
        out = query

    cache[query] = out
    _save_cache(cache)
    return out



class QueryRewriteRetriever:
    """在任意 Retriever 外面包一层"原 query + 改写 query"的 RRF。

    不在这里 rerank:先融合两路的候选,再由外层的 RerankPipeline 统一打分。
    否则同一个块会被两个 query 各 rerank 一次,分数口径就乱了。
    """

    def __init__(self, base: Retriever, k_rrf: int = RRF_K,
                 weights: tuple[float, float] = (1.0, 1.0)):
        self.base = base
        self.k_rrf = k_rrf
        self.weights = list(weights)

    def retrieve(self, query: str, k: int = COARSE_K) -> list[ScoredChunk]:
        original = self.base.retrieve(query, k=k)
        rewritten = rewrite_query(query)
        if rewritten == query:
            return original
        alternate = self.base.retrieve(rewritten, k=k)
        return rrf_fuse([original, alternate], k=self.k_rrf,
                        weights=self.weights)[:k]
