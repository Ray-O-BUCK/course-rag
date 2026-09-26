import chromadb
from pathlib import Path
from chunker import chunk_text
from embedder import embed
from config import CHROMA_PATH,CHUNK_SIZE,CHUNK_OVERLAP,BASE_DIR

client = chromadb.PersistentClient(path = CHROMA_PATH)
collection = client.get_or_create_collection("course_docs")

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
    for f in raw_dir.glob("*"):
        if f.suffix.lower() in (".md", ".txt"):
            ingest_file(str(f), f.name)

if __name__ == "__main__":
    ingest_all()