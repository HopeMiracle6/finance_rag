from src.bm25_retriever import BM25Retriever
from src.schema import TextChunk


def test_bm25_search_returns_result():
    chunks = [
        TextChunk(chunk_id="c1", doc_id="d1", source_file="a.txt", text="公司净利润增长主要由于订单增加和成本下降。"),
        TextChunk(chunk_id="c2", doc_id="d1", source_file="a.txt", text="投资者应注意市场竞争加剧风险。"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.search("净利润增长原因", top_k=1)
    assert results
    assert results[0].chunk_id == "c1"
