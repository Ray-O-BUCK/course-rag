from pathlib import Path
from chunker import chunk_text
from config import BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from db import collection
from embedder import embed

def ingest_file(path: str,doc_name: str):
    text = Path(path).read_text(encoding = "utf-8")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print(f"{doc_name}: 无内容")
        return
    embeddings = embed(chunks)
    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc": doc_name, "chunk_id": i} for i in range(len(chunks))]
    collection.upsert(ids = ids, documents = chunks, embeddings = embeddings, metadatas = metadatas)
    print(f"{doc_name}: 入库 {len(chunks)} 块")

def ingest_all():
    raw_dir = BASE_DIR / "data" / "raw"
    # rglob 不是 glob:开源语料在 data/raw/fetched/ 子目录里,glob("*") 看不到。
    # doc_name 用扁平文件名 —— fetch_corpus.py 已经把目录层级拼进文件名了
    # (d2l-zh__chapter_xxx__yyy.md),所以跨目录不会撞名。
    # sorted 保证入库顺序确定,复现时块 id 不会因文件系统枚举顺序漂移。
    for f in sorted(raw_dir.rglob("*")):
        if f.suffix.lower() in (".md", ".txt"):
            ingest_file(str(f), f.name)

if __name__ == "__main__":
    ingest_all()