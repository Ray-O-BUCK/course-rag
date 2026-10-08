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
RETRIEVER = "vector"
COARSE_K = 20 # 粗排候选数(向量/BM25 各取多少)
# 真正送进 LLM 的块数。Phase 3 上 reranker 之前必须 >= 10:
# 没有 reranker 时排序就是向量距离,收到 5 会把 rank 7~9 的正确块切掉
# (README 踩坑 #3:ResNet/LSTM 短查询)。reranker 上线后才收到 5 —— 那是它挣来的收益。
FINAL_K = 10
RRF_K = 60 # RRF 平滑常数
# 上下文安全网,不是相关性过滤器:10 块 x 300 字 ≈ 3200 字,留 25% 余量。
# 真正的上限该由模型上下文窗口决定,等换成 tokenizer 精确计数时再调。
CONTEXT_MAX_CHARS = 4000
CHUNK_MODE = "fixed" # fixed | structure
RERANKER_MODEL = "bge-reranker-base"

# --- 拒答阈值 ---
# Chroma 集合是 l2 空间, 向量已归一化, l2 = 2 - 2*cos, 值域 [0, 4],越小越相关。
# 闸门语义:候选的 top-1 距离 > 阈值 即拒答(见 rag.py)。
#
# 0.90 是在 81 条评测集 / 1210 块语料上重标的,证据在 eval/results/threshold.json。
# 旧的 1.15 是"3 篇文档 / 10 块"时代拍的,在新语料上误答 13/17 = 76.5%,等于没有闸门。
#
# 为什么停在 0.90 而不是 J 最大的 0.922:两者误答同为 5/17,但 0.90 多拒 2 题换
# 一点余量。真正的限制在下面这条,不是这几个小数:
#
#   **top-1 距离挡不住"伪相关"题 —— 这是结构性上限,调参解决不了。**
#   5 条 OOD 埋在正样本的 p25~p75 区间 [0.593, 0.798] 里:
#     "训练 BERT 需要多少张 GPU" -> 0.716,命中真正在讲多 GPU 的块
#     "Transformer 一共有多少参数" -> 0.770,命中真正在讲 Transformer 的块
#   要误答归零,阈值得压到 0.62,那时误拒 32/53 = 60%,一半以上正常问题答不出来。
#   根因:embedding 距离量的是"话题相关",不是"能不能回答"。
#
# 后续两条路(见 README"实验"章):
#   Phase 3 用 reranker 分数当闸门重标一次 —— 交叉编码器对"这块回答了这个问题吗"
#   比双塔敏锐得多。eval/calibrate_threshold.py 到时换信号源重跑。
#   阈值只在 dev 集上标,holdout 集不参与调参。
DISTANCE_THRESHOLD = 0.90

# TOP_K 已被 COARSE_K/FINAL_K 取代, 保留仅为兼容旧脚本
TOP_K = 10

# 以文件所在位置定位项目根,避免相对路径随工作目录漂移
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "data" / "chroma")