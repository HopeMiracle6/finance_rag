from src.text_cleaner import clean_text


def test_clean_text_keeps_financial_numbers():
    text = "第 1 页\n预计净利润为 1.4 亿元至 1.6 亿元，增长 40% 至 60%。\n2\n"
    cleaned = clean_text(text)
    assert "第 1 页" not in cleaned
    assert "1.4 亿元" in cleaned
    assert "40%" in cleaned


def test_clean_text_merges_chinese_linebreak():
    cleaned = clean_text("公司核心产品订单\n增加，生产效率提升。")
    assert "订单增加" in cleaned
