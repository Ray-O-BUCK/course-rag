import json
import re
import sys
from pathlib import Path

from config import TOP_K

# 无论从哪个目录运行,都能导入项目根目录的模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import collection
from rag import pipeline,answer

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.jsonl"

# 答案中的引用格式: [doc:文件名#chunk_id]
CITE_PATTERN = re.compile(r"\[doc:([^#\]]+)#(\d+)\]")


def load_eval_set() -> list[dict]:
    items = []
    for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def eval_recall(items: list[dict]) -> None:
    """Recall@K: 检索 top-K 是否命中期望文档"""
    total = hits = 0
    for item in items:
        if not item["relevant_docs"]:
            continue
        total += 1
        docs = pipeline.retrieve(item["query"], k=TOP_K)
        retrieved_docs = [d.doc for d in docs]
        hit = any(rd in retrieved_docs for rd in item["relevant_docs"])
        print(f"{'✓' if hit else '✗'} {item['query']}")
        print(f"   期望: {item['relevant_docs']}")
        print(f"   实际: {retrieved_docs}")
        if hit:
            hits += 1
    print()
    print(f"Recall@K: {hits}/{total} = {hits/total:.2%}" if total else "无 Recall 样本")
    print()


def eval_reject(items: list[dict]) -> None:
    """拒答准确率: 无关问题是否被正确拒答"""
    total = ok = 0
    for item in items:
        if item["relevant_docs"]:
            continue
        total += 1
        res = answer(item["query"])
        passed = "无法回答" in res["answer"]
        print(f"{'✓' if passed else '✗'} [拒答] {item['query']}")
        if passed:
            ok += 1
    print()
    print(f"拒答准确率: {ok}/{total} = {ok/total:.2%}" if total else "无拒答样本")
    print()


def eval_citation(items: list[dict]) -> None:
    """引用准确率,分两个指标:
    - 引用有效率: 答案至少有一个 [doc:file#chunk] 引用,且每个引用都能在
      本次实际检索结果中找到(不编造来源,引用的块确实进了上下文)
    - 引用命中率: 答案引用的文件覆盖了期望的 relevant_docs(引对了文件)
    """
    total = valid_ok = hit_ok = 0
    for item in items:
        if not item["relevant_docs"]:
            continue
        total += 1
        res = answer(item["query"])
        cites = CITE_PATTERN.findall(res["answer"])  # [(file, chunk_id), ...]
        # 本次检索结果中真实存在的 (文件, 块) 集合
        retrieved_keys = {(d.doc, str(d.chunk_id)) for d in res["sources"]}

        valid = bool(cites) and all(key in retrieved_keys for key in cites)
        cited_docs = {doc for doc, _ in cites}
        hit = bool(cited_docs & set(item["relevant_docs"]))

        if valid:
            valid_ok += 1
        if hit:
            hit_ok += 1

        mark = "✓" if (valid and hit) else "✗"
        print(f"{mark} {item['query']}")
        print(f"   期望文件: {item['relevant_docs']}")
        print(f"   实际引用: {[f'{d}#{c}' for d, c in cites] or '无引用'}")
        print(f"   有效={valid} 命中={hit}")

    print()
    print(f"引用有效率: {valid_ok}/{total} = {valid_ok/total:.2%}" if total else "无引用样本")
    print(f"引用命中率: {hit_ok}/{total} = {hit_ok/total:.2%}" if total else "无引用样本")


def main() -> None:
    items = load_eval_set()

    # 空库检查: 防止再次出现"查了另一个目录下的空 Chroma"这类静默失败
    count = collection.count()
    if count == 0:
        print(f"错误: Chroma 集合为空 ({collection.name}),请先运行 python ingest.py 入库")
        sys.exit(1)
    print(f"Chroma 集合 '{collection.name}' 共 {count} 块\n")

    print("== 召回测试 ==")
    eval_recall(items)

    print("== 拒答测试 ==")
    eval_reject(items)

    print("== 引用准确率测试 ==")
    eval_citation(items)


if __name__ == "__main__":
    main()
