from sentence_transformers import SentenceTransformer
from pathlib import Path

_model = None
MODEL_NAME = str(Path(__file__).resolve().parent / "models" / "bge-small-zh-v1.5")

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    model = get_model() # type: SentenceTransformer
    vecs = model.encode(texts, normalize_embeddings=True) # type: np.ndarray
    return vecs.tolist() # type: list[list[float]]

def embed_query(query: str) -> list[float]:
    # bge 查询侧添加指令前缀
    return embed([f"为这个句子生成表示以用于检索相关文章:{query}"])[0]

