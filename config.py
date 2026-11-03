import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def _require_env(name: str) -> str:
    """读取必需的环境变量,缺失直接抛错。

    在 import 阶段失败 = 应用根本起不来,而不是跑到某个请求中间才失败。
    用 raise 而不是 assert:assert 会在 python -O 下被整段优化掉,不能做生产校验。
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}:请复制 .env.example 为 .env 并填写后再启动"
        )
    return value

LLM_API_KEY = _require_env("LLM_API_KEY")
LLM_BASE_URL = _require_env("LLM_BASE_URL")
LLM_MODEL = _require_env("LLM_MODEL")
LLM_TIMEOUT = 30
LLM_MAX_RETRIES = 2

# 定长分块参数(CHUNK_MODE = "fixed" 时生效)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
# 结构感知分块参数(CHUNK_MODE = "struct" 时生效)
STRUCT_CHUNK_MAX = 500
STRUCT_CHUNK_OVERLAP = 80

# --- 检索管线:ablation 的全部旋钮集中于此,是实验可复现的前提 ---
# 线上用 hybrid_rerank:粗排 30 块再经 cross-encoder 重排,送进 LLM 的是 FINAL_K=10 块。
# 四路消融(R@10 / R@1):vector 0.719/0.344, bm25 0.836/0.492,
# hybrid 0.883/0.391, hybrid_rerank 0.898/0.578。
# vector / bm25 / hybrid 保留为消融配置;低配机器退回 hybrid,代价是 R@1 掉 0.187。
RETRIEVER = "hybrid_rerank"
COARSE_K = 30 # 粗排候选数(向量/BM25 各取多少)。20→30 换来 R@1 +0.016、R@20 +0.039,
              # 代价是 rerank 前向 1.5 倍。COARSE_K>20 时 hybrid 与 rerank 的 R@20 不再要求相同。
# 真正送进 LLM 的块数。重排后 R@5=0.867、R@10=0.898,收到 5 会丢 3.1 个点的覆盖率,
# 所以仍是 10。P2 时"没有 reranker 不能收 5"是将就,现在这个值有数字支撑。
FINAL_K = 10

RRF_K = 60 # RRF 平滑常数
# (向量权重, BM25 权重)。等权是 RRF 的原版形式,也是这里的默认值 —— 没有在
# 评测集上调过。实测压到 (0.3, 1.0) 能把 hybrid 的 R@1 从 0.375 提到 0.453,
# 但那是**在同一份 81 条评测集上选的**,等于拿测试集调参,不采信为默认。
# reranker 已经补上了 top-1 精度(R@1 0.531),权重不再是瓶颈,维持等权。
RRF_WEIGHTS = (1.0, 1.0)
# 上下文安全网,不是相关性过滤器:10 块 x 300 字 ≈ 3200 字,留 25% 余量。
# 真正的上限该由模型上下文窗口决定,等换成 tokenizer 精确计数时再调。
CONTEXT_MAX_CHARS = 4000
CHUNK_MODE = "structure" # fixed | structure
RERANKER_MODEL = "bge-reranker-base"

# --- 拒答闸门 ---
# 闸门读"候选里最能回答这个问题的证据"是否过线。两种信号:
#   "vector"  候选里最小的 vector_distance > DISTANCE_THRESHOLD  -> 拒答
#   "rerank"  候选里最大的 rerank_score < RERANK_THRESHOLD       -> 拒答
# 信号必须和检索器配套:只有 hybrid_rerank 会产生 rerank_score。配错时 rag.py
# 抛错而不是静默退回距离阈值 —— 换了一套闸门却没人知道,是最坏的情况。
GATE_SIGNAL = "rerank"

# 向量已归一化,距离是**平方** l2 = 2 - 2*cos,值域 [0, 4],越小越相关。
# 注意是平方不是 l2:开方后值域是 [0, 2],阈值口径会整个变掉。
# 闸门语义:候选的 top-1 距离 > 阈值 即拒答(见 rag.py)。
#
# 0.90 是在 81 条评测集 / 1210 块语料上重标的,证据在 eval/results/threshold.json。
# 旧的 1.15 是"3 篇文档 / 10 块"时代拍的,在新语料上误答 13/17 = 76.5%,等于没有闸门。
#
# 为什么停在 0.90 而不是 J 最大的 0.922:两者误答同为 5/17,0.90 多拒 2 题换一点余量。
# 结构性上限:5 条 OOD 埋在正样本的 p25~p75 区间 [0.593, 0.798] 里,要让误答归零
# 得压到 0.62,那时误拒 32/53 = 60%。根因是 embedding 距离量的是"话题相关",
# 不是"能不能回答"。所以 hybrid_rerank 下改用 rerank 分数当闸门。
DISTANCE_THRESHOLD = 0.90

# rerank 分数是 sigmoid 输出,不是概率,分布明显双峰(正样本中位 0.98,多数 OOD < 0.05)。
# 0.16 在 884 块结构分块语料上重标,59 道正样本 / 17 条 OOD:
#   0.45 -> 误拒 4、误答 4(J 0.697)
#   0.16 -> 误拒 2、误答 4(J 0.731)
# 0.16 落在正样本间隙 (0.0896, 0.2166) 内,这段区间里任何阈值结果相同。
# 再降到 0.05 会多放行 c79(Rust 借用检查)一条 OOD。
# 剩余 4 条误答(c33/c34/c77/c80)分数都在 0.75 以上,不是阈值能解决的。
# 换分块方案或 reranker 模型必须重标,这个数只对当前语料 + bge-reranker-base 成立。
RERANK_THRESHOLD = 0.16

# 逐块过滤的绝对下限,和拒答阈值分开:RERANK_THRESHOLD 管"候选里最好的证据",
# 这里管"这一块自己值不值得进上下文"。0.02 在 62 道闸门放行的题上保留
# 61/61 个 gold 块,同时把 top-10 上下文从 620 块降到 498 块(-20%)。
# 不要直接复用 0.16:那会筛掉 c46(0.041)、c71(0.119) 这类真正有 gold 的块。
CONTEXT_MIN_RERANK = 0.02
# 无引用回答是否直接拒答。严格模式下每个展示的回答至少带一个有效引用;
# 它不解决"引用合法但答不了"(holdout h17 仍被回答),也不保证语义忠实。
STRICT_CITATION = True




# TOP_K 已被 COARSE_K/FINAL_K 取代, 保留仅为兼容旧脚本
TOP_K = 10

# 以文件所在位置定位项目根,避免相对路径随工作目录漂移
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "data" / "chroma")