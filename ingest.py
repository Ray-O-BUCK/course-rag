from pathlib import Path
from chunker import chunk_struct,chunk_text
from config import (BASE_DIR, CHUNK_MODE,CHUNK_OVERLAP, CHUNK_SIZE,STRUCT_CHUNK_MAX,STRUCT_CHUNK_OVERLAP)
from db import collection
from embedder import embed

def split_text(text: str) -> list[str]:
    """按 config.CHUNK_MODE 选择分块器。配错直接报错,不静默退回 fixed。"""
    if CHUNK_MODE == "fixed":
        return chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    elif CHUNK_MODE == "structure":
        return chunk_struct(text, STRUCT_CHUNK_MAX, STRUCT_CHUNK_OVERLAP)
    raise ValueError(f"未知的 CHUNK_MODE: {CHUNK_MODE!r}(可选 fixed / structure)")

def ingest_file(path: str,doc_name: str) -> int:
    text = Path(path).read_text(encoding = "utf-8")
    chunks = split_text(text)
    if not chunks:
        print(f"{doc_name}: 无内容")
        return 0

    # 先算 embedding 再动库:embed 失败时旧数据仍然完整
    embeddings = embed(chunks)
    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc": doc_name, "chunk_id": i} for i in range(len(chunks))]

    old_ids = set(collection.get(where = {"doc": doc_name}, include=[])["ids"])

    # 先 upsert 再删旧尾巴:新块数变少时,多出来的旧 id 不会被覆盖,
    # 必须显式删除。反过来"先删后写"一旦中途崩溃,库里会直接缺一篇文档。
    collection.upsert(ids = ids, documents = chunks, embeddings = embeddings, metadatas = metadatas)
    stale = old_ids - set(ids)
    if stale:
        collection.delete(ids = sorted(stale))

    print(f"{doc_name}: 新增/更新 {len(chunks)} 块,清理旧块 {len(stale)}")
    return len(chunks)

def prune_missing(expected: set[str]) -> None:
    """删除 raw 目录里已经不存在的文档,否则旧文档会永远留在检索结果里。"""
    got = collection.get(include=["metadatas"])
    stale: dict[str, list[str]] = {}
    for _id, meta in zip(got["ids"], got.get("metadatas") or []):
        doc = (meta or {}).get("doc")
        if doc and doc not in expected:
            stale.setdefault(doc, []).append(_id)
    for doc, ids in sorted(stale.items(), key = lambda kv: str(kv[0])):
        collection.delete(ids = ids)
        print(f"{doc}: 源文件已删除,清理 {len(ids)} 块")

def ingest_all():
    raw_dir = BASE_DIR / "data" / "raw"
    files = [f for f in sorted(raw_dir.rglob("*")) if f.suffix.lower() in (".md", ".txt")]
    if not files:
        raise FileNotFoundError(f"{raw_dir} 下没有 .md / .txt 文件")
    names = [f.name for f in files]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"文件名重复:{dups}。doc_name 用 f.name,会互相覆盖")

    total = 0
    for f in files:
        total += ingest_file(str(f), f.name)
    prune_missing({f.name for f in files})
    print(f"完成:{len(files)} 篇,入库 {total} 块,库内共{collection.count()} 块")

if __name__ == "__main__":
    ingest_all()