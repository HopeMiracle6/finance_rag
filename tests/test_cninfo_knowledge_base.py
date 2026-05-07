from src.cninfo_client import normalize_cninfo_item
from src.knowledge_base_builder import chunk_continuous_document, pages_to_raw_document, text_chunk_to_knowledge_chunk
from src.schema import KnowledgeDocumentPage, TextChunk


def test_normalize_cninfo_item_keeps_core_metadata():
    item = {
        "id": "000001_1_demo",
        "sec_code": "000001",
        "sec_name": "示例银行",
        "title": "示例银行2024年年度报告",
        "announcement_time": "2024-03-30",
        "pdf_url": "http://static.cninfo.com.cn/demo.pdf",
        "event_type": "年度报告",
    }
    metadata = normalize_cninfo_item(item, pdf_path="data/raw/cninfo_pdfs/demo.pdf")

    assert metadata.company_name == "示例银行"
    assert metadata.stock_code == "000001"
    assert metadata.report_type == "年度报告"
    assert metadata.publish_date == "2024-03-30"
    assert metadata.source_url == "http://static.cninfo.com.cn/demo.pdf"


def test_text_chunk_to_knowledge_chunk_keeps_citation_fields():
    chunk = TextChunk(
        chunk_id="chunk_1",
        doc_id="doc_1",
        source_file="demo.pdf",
        page=3,
        text="公司披露年度报告。",
        metadata={
            "file_name": "demo.pdf",
            "company_name": "示例银行",
            "stock_code": "000001",
            "source_url": "http://static.cninfo.com.cn/demo.pdf",
            "report_type": "年度报告",
            "publish_date": "2024-03-30",
        },
    )

    kb_chunk = text_chunk_to_knowledge_chunk(chunk)

    assert kb_chunk.file_name == "demo.pdf"
    assert kb_chunk.page == 3
    assert kb_chunk.company_name == "示例银行"
    assert kb_chunk.stock_code == "000001"
    assert kb_chunk.source_url == "http://static.cninfo.com.cn/demo.pdf"


def test_continuous_document_chunk_keeps_page_range():
    pages = [
        KnowledgeDocumentPage(doc_id="doc_1", file_name="demo.pdf", page=1, text="第一页说明公司订单增加。"),
        KnowledgeDocumentPage(doc_id="doc_1", file_name="demo.pdf", page=2, text="第二页继续说明成本下降。"),
    ]
    document = pages_to_raw_document(pages)
    chunks = chunk_continuous_document(document, chunk_size=200, chunk_overlap=20, min_chunk_size=10)

    assert chunks
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert "[[PAGE=" not in chunks[0].text
