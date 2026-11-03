import streamlit as st
from rag import answer

st.set_page_config(page_title= "课程资料 RAG 问答", layout = "wide")
st.title("课程资料 RAG 问答助手")

with st.sidebar:
    st.header("文档入库")
    uploaded = st.file_uploader("上传 MarkDown / txt", type = ["md", "txt"])
    if uploaded is not None:
        from config import BASE_DIR
        raw_dir = BASE_DIR / "data" / "raw"
        raw_dir.mkdir(parents = True, exist_ok= True)
        path = raw_dir / uploaded.name
        path.write_bytes(uploaded.read())
        from ingest import ingest_file
        ingest_file(str(path), uploaded.name)
        st.success(f"{uploaded.name} 已入库")

st.subheader("提问")
q = st.text_input("输入你的问题")
if st.button("查询") and q:
    with st.spinner("检索并生成中..."):
        result = answer(q)
    st.markdown("### 回答")
    st.write(result["answer"])
    check = result.get("citation_check")
    if check and result["sources"]:
        if check["invalid"]:
            st.warning("以下引用不在本次上下文中:" + "、".join(check["invalid"]))
        elif check["no_citation"]:
            st.warning("回答没有标注任何来源")
    with st.expander("引用来源"):
        for s in result["sources"]:
            # 混合检索下每个块只带自己那一路的分数:BM25 独有的块没有 vector_distance。
            # 无条件格式化 {: .3f} 会在 None 上抛 TypeError。
            parts = []
            if s.vector_distance is not None:
                parts.append(f"dist {s.vector_distance:.3f}")
            if s.bm25_score is not None:
                parts.append(f"bm25 {s.bm25_score:.2f}")
            if s.fusion_score is not None:
                parts.append(f"rrf {s.fusion_score:.4f}")
            st.markdown(f"**{s.doc}#{s.chunk_id}**  " + " · ".join(parts))
            st.text(s.text[:300])