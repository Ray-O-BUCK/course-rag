"""抓取开源语料到 data/raw/fetched/。

为什么不把语料提交进仓库
------------------------
- d2l-zh 正文是 CC BY-SA 4.0(share-alike),原文进公开仓库会传染许可
- 几 MB 文本进 git 只增不减
- 用 commit SHA 固定版本,评测数字才可复现(分支名会漂,今天抓的和
  下周抓的可能不是同一份语料,那数字就没法比)

为什么是显式文件列表,而不是按目录 glob
--------------------------------------
可用语料 5.7 MB,本项目只要 500~900 块,即 5% 左右。任何"按目录+预算截断"
的启发式都在做任意选择:实测第一版把 chapter_multilayer-perceptrons 排在最前,
结果 28 KB 的 environment.md(环境伦理)和 25 KB 的 kaggle-house-price.md
(Kaggle 教程)吃掉 38% 预算,而 mlp.md 这个核心文件一个字节都没进来。
显式列出路径可控、可审、可复现,而且这份列表本身就是语料清单文档。

为什么不调 GitHub API 列目录
---------------------------
api.github.com 未认证时限 60 次/小时,按**出口 IP** 算。这台机器走代理,
出口 IP 共享,实测探了 20 次就 403 了。raw.githubusercontent.com 是另一套,
不限流 —— 所以这里只用 raw,一个 API 调用都不发。

用法
----
    python scripts/fetch_corpus.py --dry-run    # 只列出要抓什么,不写盘
    python scripts/fetch_corpus.py              # 抓到 data/raw/fetched/
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "fetched"

# 没有合规 UA 的请求会被 GitHub 拒(403)。这不是可选项。
UA = {"User-Agent": "course-rag-corpus/0.1 (educational project; stdlib urllib)"}

RETRIES = 4
BACKOFF = 1.5  # 秒,线性退避 1.5 / 3.0 / 4.5


@dataclass
class Source:
    name: str  # 输出文件名前缀
    repo: str  # "owner/name"
    ref: str  # commit SHA —— 不用分支名,理由见文件头
    files: tuple[str, ...]  # 相对 dir 的路径,顺序即输出顺序
    budget_kb: int  # 体积守卫:超出直接报错,防止语料无声膨胀
    license: str
    note: str
    dir: str = ""  # 仓库内目录前缀。d2l-zh 的文件跨多个 chapter,留空


SOURCES: list[Source] = [
    Source(
        name="d2l-zh",
        repo="d2l-ai/d2l-zh",
        ref="e6b18ccea71451a55fcd861d7b96fddf2587b09a",
        files=(
            # 和 nn.md 重合度最高的一块:反向传播 / 梯度消失爆炸 / 初始化
            "chapter_multilayer-perceptrons/backprop.md",
            "chapter_multilayer-perceptrons/numerical-stability-and-init.md",
            "chapter_multilayer-perceptrons/mlp.md",
            "chapter_multilayer-perceptrons/dropout.md",
            # 和 transformer.md 重合:注意力评分 / 多头 / 自注意力 / 位置编码
            "chapter_attention-mechanisms/attention-scoring-functions.md",
            "chapter_attention-mechanisms/multihead-attention.md",
            "chapter_attention-mechanisms/self-attention-and-positional-encoding.md",
            # 序列模型,给 LSTM 类问题当对照
            "chapter_recurrent-neural-networks/rnn.md",
            # 优化器
            "chapter_optimization/adam.md",
            "chapter_optimization/momentum.md",
            # 分类与交叉熵
            "chapter_linear-networks/softmax-regression.md",
        ),
        budget_kb=190,
        license="CC BY-SA 4.0(正文)/ Apache-2.0(仓库)",
        note="《动手学深度学习》中文版;正文译自 CC BY-SA 4.0 的英文原版,按 CC BY-SA 对待",
    ),
    Source(
        name="hf-course",
        repo="huggingface/course",
        ref="92d4fb2c7094858430ce3e45851c12d165e36481",
        files=tuple(
            f"chapters/zh-CN/chapter{c}/{i}.mdx"
            for c, n in (("1", 10), ("6", 10))  # 1=Transformer 模型, 6=Tokenizer
            for i in range(1, n + 1)
        ),
        budget_kb=195,
        license="Apache-2.0",
        note="Hugging Face NLP 课程官方中文版;Notion 导出,含 [[双链]] 和 JSX 组件",
    ),
    Source(
        name="transformers-zh",
        repo="huggingface/transformers",
        ref="469230357aab0f2b303b0d638c1f8d06edb14184",
        dir="docs/source/zh/",
        files=(
            # 根目录:教程与概念
            "attention.md",
            "fast_tokenizers.md",
            "quicktour.md",
            "pipeline_tutorial.md",
            # 具体 NLP 任务 —— 比根目录那批 add_new_*/contributing 相关得多
            "tasks/sequence_classification.md",
            "tasks/token_classification.md",
            "tasks/question_answering.md",
            "tasks/summarization.md",
            "tasks/translation.md",
            "model_doc/bert.md",
        ),
        budget_kb=110,
        license="Apache-2.0",
        note="transformers 中文文档(部分翻译)",
    ),
]
# 注意:上面三个 file 列表的路径都**预先验证过存在**(raw HEAD 探测),
# 仓库里没有 glossary.md / perplexity.md / tasks/language_modeling.md,
# 不要凭印象往里加。


# --- 文本清洗 ---------------------------------------------------------------
# 只删"独占一行的标记"。<Tip> 这类标签本身没内容,但它包着的正文要保留,
# 所以不能整段删 —— 正则一律锚定行首行尾,避免误伤正文里的尖括号。
_LINE_NOISE = (
    re.compile(r"^\s*import\s.+from\s.+$"),  # mdx 的 import 语句
    re.compile(r"^\s*:label:`?[^`]*`?\s*$"),  # d2l 的章节锚点
    # hf-course 的 Svelte 模板指令,如 {#if fw === 'pt'} / {:else} / {/if}。
    # 只删指令行,两侧的分支内容(pytorch 版和 tensorflow 版)都留着。
    re.compile(r"^\s*\{[/:#].*$"),
    re.compile(r"^\s*</?[A-Za-z][\w.-]*(?:\s[^>]*)?/?>\s*$"),  # 独占一行的 JSX/HTML 标签
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# 跨行的空元素标签。img/br/hr 这类没有闭合标签,吞到第一个 > 就一定安全。
# 必须在 _JSX_BLOCK 之前跑:否则 bert.md 开头那个跨行 <img> 会被后面的规则
# 一路吞到下一个 /> 为止,把中间整段正文带走。
_VOID_TAG = re.compile(r"^[ \t]*<(?:img|br|hr|source|input)\b.*?/?>[ \t]*$", re.DOTALL | re.M)
# 跨行的自闭合 JSX 组件,如 <Question / <CourseFloatingBanner,形如:
#     <Question
#         choices={[{ text: "...", explain: "点击前往<a href=\"...\">..." }]}
#     />
# 内容全在属性里,正文只有外层那个 ### 小标题(那个保留),所以整块删。
#
# 三处细节,少一个就会吃掉正文:
#   [^>\n]*\n  起始行**必须没有 >**,即这个标签确实跨行。少了它,正文里独占
#              一行的 <a href="..."> 会被当起点,一路懒扫到远处的下一个 />,
#              把中间的代码块整段带走 —— 实测就是这么把 chapter6 的 ``` 围栏
#              删成奇数的。
#   [A-Za-z]   标签名首字符是字母,避免把 "< 0.05" 这种数学不等号当标签。
#   .*?        不能用 [^>]*,因为属性串里含 >(嵌套的 <a href="...">)。
_JSX_BLOCK = re.compile(r"^[ \t]*<[A-Za-z][\w.-]*[^>\n]*\n.*?/>[ \t]*$", re.DOTALL | re.M)
# Notion 导出的标题锚点,如 "# 本章简介 [[本章简介]]"
_NOTION_ANCHOR = re.compile(r"\s*\[\[[^\]]*\]\]")
# d2l 的行内交叉引用,如 "如 :numref:`sec_mlp` 所示" -> "如  所示"
_INLINE_DIRECTIVE = re.compile(r":(?:numref|eqref|ref|cite|label):`[^`]*`")
_BLANK_RUN = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    """去掉标记噪音。

    分两层:
    1. 整块删 —— 注释、跨行 <img>、跨行自闭合 JSX 组件(整个块没有正文)
    2. 逐行删 —— 独占一行的标签/import/章节锚点(<Tip> 这类要留内容,只删标签)

    不处理:配对 JSX 组件内部的多行属性、表格排版。实测全语料只剩
    Question / CourseFloatingBanner / img 三种跨行标签,都是自闭合,
    为它们写一个真正的 markdown 解析器不划算。
    """
    text = _HTML_COMMENT.sub("", text)
    text = _VOID_TAG.sub("", text)  # 顺序固定:见 _VOID_TAG 处的说明
    text = _JSX_BLOCK.sub("", text)
    text = _NOTION_ANCHOR.sub("", text)
    text = _INLINE_DIRECTIVE.sub("", text)
    lines = [ln for ln in text.splitlines() if not any(p.match(ln) for p in _LINE_NOISE)]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


# --- 网络 -------------------------------------------------------------------


def get(url: str) -> bytes:
    """带重试的 GET。走代理时 DNS 会间歇性失败,不重试必然中断。"""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc  # 路径错,重试没意义
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < RETRIES - 1:
            time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"重试 {RETRIES} 次仍失败: {url}\n  最后一次: {last}")


# --- 输出命名 ---------------------------------------------------------------


def common_dir_prefix(paths: tuple[str, ...]) -> str:
    """所有路径共有的目录前缀。

    transformers 的路径是 docs/source/zh/xxx.md,那一长串在输出名里毫无信息量;
    d2l-zh 的文件跨多个 chapter,没有公共前缀。算出来再截掉,两边都得到短名字。
    """
    if not paths:
        return ""
    parts = [Path(p).parts[:-1] for p in paths]
    shared: list[str] = []
    for i in range(min(len(p) for p in parts)):
        if len({p[i] for p in parts}) == 1:
            shared.append(parts[0][i])
        else:
            break
    return "/".join(shared) + "/" if shared else ""


def out_name(src: Source, path: str, prefix: str) -> str:
    """扁平化文件名,把剩余目录层级拼进名字避免撞名。

    hf-course 每个 chapter 下都有 1.mdx,只用 basename 会互相覆盖。
    后缀统一改成 .md —— ingest.py 只认 .md/.txt,.mdx 会被静默跳过。
    """
    rel = path[len(prefix):] if path.startswith(prefix) else path
    return f"{src.name}__" + "__".join(Path(rel).with_suffix("").parts) + ".md"


# --- 主流程 -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="抓取开源语料")
    ap.add_argument("--dry-run", action="store_true", help="只列出要抓什么,不写盘")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"sources": []}
    failures: list[str] = []
    total_files = total_bytes = total_chars = 0

    for src in SOURCES:
        full_paths = tuple(src.dir + p for p in src.files)
        prefix = common_dir_prefix(full_paths)
        print(f"[{src.name}] {src.repo}@{src.ref[:12]}  ({src.license})")

        names = [out_name(src, p, prefix) for p in full_paths]
        entry = {
            "name": src.name, "repo": src.repo, "ref": src.ref,
            "license": src.license, "note": src.note, "files": [],
        }
        src_bytes = src_chars = 0

        for path, name in zip(full_paths, names):
            if args.dry_run:
                print(f"    {path}  ->  {name}")
                continue
            dest = OUT_DIR / name
            try:
                raw = get(f"https://raw.githubusercontent.com/{src.repo}/{src.ref}/{path}")
            except FileNotFoundError:
                failures.append(f"{src.name}: 仓库里没有 {path}(SHA 变了?路径写错了?)")
                continue
            except RuntimeError as exc:
                failures.append(f"{src.name}: {exc}")
                continue
            raw_text = raw.decode("utf-8")
            text = clean(raw_text)
            # 守卫:清洗只该删标记,不该删正文。围栏变奇 = 某条规则吞了代码块。
            if text.count("```") % 2:
                failures.append(
                    f"{src.name}: {path} 清洗后代码围栏不成对"
                    f"(raw {raw_text.count('```')} -> clean {text.count('```')} 处)。"
                    f"清洗规则吃掉了正文,修脚本再抓,别直接入库。"
                )
            dest.write_text(text, encoding="utf-8")
            entry["files"].append(
                {"path": path, "out": name, "chars": len(text)}
            )
            src_bytes += len(text.encode("utf-8"))
            src_chars += len(text)

        if args.dry_run:
            print()
            continue

        kb = src_bytes / 1024
        flag = "  ← 超预算" if kb > src.budget_kb else ""
        print(f"  {len(entry['files'])}/{len(src.files)} 个文件,{kb:.0f} KB"
              f"(预算 {src.budget_kb} KB){flag}\n")
        if kb > src.budget_kb:
            failures.append(
                f"{src.name}: {kb:.0f} KB 超出预算 {src.budget_kb} KB。"
                f"语料体积会改变全部评测数字,所以卡一道。确认是有意扩的就把 budget_kb 改大。"
            )

        total_files += len(entry["files"])
        total_bytes += src_bytes
        total_chars += src_chars
        manifest["sources"].append(entry)

    if args.dry_run:
        print("dry-run 结束,未写盘。核对清单后正式跑。")
        return

    manifest["fetched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"共 {total_files} 个文件 / {total_bytes / 1024:.0f} KB / {total_chars} 字符 -> {OUT_DIR}")
    # 按**字符**估块数,不是字节 —— 中文 3 字节/字符,拿字节算会高估 3 倍
    print(f"预估 ~{total_chars // 240} 个新块(stride = CHUNK_SIZE - CHUNK_OVERLAP = 240)")

    if failures:
        print(f"\n{len(failures)} 个问题:", file=sys.stderr)
        for f in failures:
            print(f"  ! {f}", file=sys.stderr)
        sys.exit(1)

    print()
    print("下一步:")
    print("  python ingest.py         # 重新入库,块数会从 10 涨到数百")
    print("  python eval/anchors.py   # 新语料可能撞坏锚点,重点看有没有'锚点不唯一'")


if __name__ == "__main__":
    main()
