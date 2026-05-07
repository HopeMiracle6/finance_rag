from src.prompt import build_rag_prompt, is_investment_advice_question
from src.schema import RetrievalResult


def test_investment_advice_detection():
    assert is_investment_advice_question("这家公司值得买入吗？")


def test_prompt_contains_citation_context():
    result = RetrievalResult(
        chunk_id="c1",
        doc_id="d1",
        source_file="a.txt",
        page=None,
        text="公司预计净利润增长。",
        score=1.0,
        retrieval_type="bm25",
        rank=1,
    )
    prompt = build_rag_prompt("原因是什么？", [result])
    assert "你只能基于【已检索资料】回答问题" in prompt
    assert "chunk_id：c1" in prompt
    assert "引用来源：" in prompt
