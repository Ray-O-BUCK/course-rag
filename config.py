import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")

# 分片大小和重叠部分的设置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
# 实测 ResNet/LSTM 等短查询的正确块排在第 7~9 名,top_k=5 时进不了上下文
TOP_K = 10
# 实测 bge + 归一化后余弦距离:相关问题 top-1 <= 1.14,无关问题 >= 1.14。
# 取 1.15 优先保证相关问题不被误拒,无关问题交给 system prompt 的拒答规则兜底
DISTANCE_THRESHOLD = 1.15

# 以文件所在位置定位项目根,避免相对路径随工作目录漂移
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "data" / "chroma")