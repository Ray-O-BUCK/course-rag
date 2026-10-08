"""Chroma 客户端与集合的唯一定义处。"""
import chromadb

from config import CHROMA_PATH

client = chromadb.PersistentClient(path = CHROMA_PATH)
# 集合使用 Chroma 默认的 l2 空间, 与 config.DISTANCE_THRESHOLD 的标定一致
collection = client.get_or_create_collection("course_docs")