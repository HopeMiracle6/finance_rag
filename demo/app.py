from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bm25_retriever import BM25Retriever
from src.chunker import chunk_and_save
from src.config import load_config, resolve_path
from src.dense_retriever import DenseRetriever
from src.document_loader import load_and_save
from src.rag_pipeline import RAGPipeline
from src.schema import TextChunk
from src.utils import bool_arg, format_page_range, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rebuild_indexes(config: dict) -> tuple[int, int]:
    paths = config["paths"]
    chunk_cfg = config["chunking"]
    raw_docs_dir = resolve_path(paths["raw_docs_dir"], PROJECT_ROOT)
    documents = load_and_save(raw_docs_dir, resolve_path(paths["documents_path"], PROJECT_ROOT))
    chunks = chunk_and_save(
        documents,
        str(resolve_path(paths["chunks_path"], PROJECT_ROOT)),
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        min_chunk_size=chunk_cfg["min_chunk_size"],
    )
    bm25 = BM25Retriever(chunks)
    bm25.save(resolve_path(paths["bm25_index_path"], PROJECT_ROOT))
    emb_cfg = config["embedding"]
    dense = DenseRetriever(
        persist_dir=resolve_path(paths["chroma_persist_dir"], PROJECT_ROOT),
        embedding_model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
        device=emb_cfg.get("device", "auto"),
        batch_size=emb_cfg.get("batch_size", 16),
        chroma_batch_size=emb_cfg.get("chroma_batch_size", 512),
        allow_embedding_fallback=emb_cfg.get("allow_fallback", False),
    )
    dense.build_index(chunks)
    st.cache_resource.clear()
    return len(documents), len(chunks)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(config_path=PROJECT_ROOT / "configs/rag_config.yaml")


def main() -> None:
    st.set_page_config(page_title="中文金融公告/研报 RAG 问答", layout="wide")
    config = load_config(PROJECT_ROOT / "configs/rag_config.yaml")

    with st.sidebar:
        st.header("配置")
        retrieval_mode = st.selectbox("retrieval_mode", ["hybrid", "bm25", "dense"], index=0)
        use_reranker = st.checkbox("use_reranker", value=True)
        top_k = st.number_input("top_k", min_value=1, max_value=100, value=30, step=1)
        final_top_n = st.number_input("final_top_n", min_value=1, max_value=20, value=5, step=1)
        embedding_model = st.text_input("embedding_model", value=config["embedding"].get("model_name", "BAAI/bge-m3"))
        reranker_model = st.text_input("reranker_model", value=config["reranker"].get("model_name", "BAAI/bge-reranker-v2-m3"))
        st.caption(f"当前配置模型：{embedding_model} / {reranker_model}")

    st.title("中文金融公告/研报 RAG 问答与引用溯源系统")

    uploaded_files = st.file_uploader("上传 PDF / TXT / Markdown 文档", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True)
    if uploaded_files:
        raw_dir = resolve_path(config["paths"]["raw_docs_dir"], PROJECT_ROOT)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for uploaded in uploaded_files:
            (raw_dir / uploaded.name).write_bytes(uploaded.getbuffer())
        st.success(f"已保存 {len(uploaded_files)} 个文件到 data/raw_docs")

    if st.button("解析文档并重建索引"):
        with st.spinner("正在解析文档并构建索引"):
            doc_count, chunk_count = rebuild_indexes(config)
        st.success(f"完成：documents={doc_count}, chunks={chunk_count}")

    examples = [
        "这份公告的核心结论是什么？",
        "公司净利润变化的原因是什么？",
        "这份公告有哪些风险？",
        "这家公司值得买入吗？",
    ]
    selected = st.radio("示例问题", examples, horizontal=True)
    question = st.text_area("问题", value=selected, height=90)

    if st.button("开始问答", type="primary"):
        pipeline = get_pipeline()
        with st.spinner("检索和生成中"):
            result = pipeline.ask(
                question=question,
                retrieval_mode=retrieval_mode,
                use_reranker=use_reranker,
                top_k=int(top_k),
                final_top_n=int(final_top_n),
            )

        st.subheader("模型回答")
        st.write(result.answer)

        st.subheader("引用来源")
        if not result.citations:
            st.warning("当前文档库中未检索到足够依据，建议补充资料或换一种问法。")
        for idx, item in enumerate(result.citations, start=1):
            page = format_page_range(item.page, item.page_start, item.page_end)
            with st.expander(f"[S{idx}] {item.source_file} / 页码 {page} / {item.chunk_id} / score={item.score:.4f}"):
                st.caption(f"retrieval_type: {item.retrieval_type}")
                st.write(item.text)

        st.subheader("检索片段")
        retrieved_chunks = result.metadata.get("retrieved_chunks", [])
        rows = [
            {
                "rank": item.get("rank"),
                "source_file": item.get("source_file"),
                "page": format_page_range(item.get("page"), item.get("page_start"), item.get("page_end")),
                "score": round(float(item.get("score", 0.0)), 4),
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text", "")[:160],
            }
            for item in retrieved_chunks
        ]
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("暂无检索片段。")


if __name__ == "__main__":
    main()
