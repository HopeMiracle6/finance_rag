from src.chunker import chunk_document
from src.schema import RawDocument


def test_chunk_document_keeps_metadata():
    document = RawDocument(
        doc_id="doc1",
        source_file="sample.txt",
        file_type="txt",
        text="一、业绩预告\n公司预计净利润增长，主要由于订单增加、成本下降和效率提升。" * 4,
    )
    chunks = chunk_document(document, chunk_size=80, chunk_overlap=10, min_chunk_size=20)
    assert chunks
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].source_file == "sample.txt"
    assert chunks[0].chunk_id
