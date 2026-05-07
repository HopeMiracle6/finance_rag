from __future__ import annotations

from src.schema import RetrievalResult
from src.utils import format_page_range


SYSTEM_INSTRUCTION = """你是一个中文金融公告与研报解读助手。

你只能基于【已检索资料】回答问题。
如果资料中没有足够依据，请明确回答：“仅凭当前资料无法判断。”
不得编造资料中没有出现的事实、数字、日期、公司名称或结论。
不得给出买入、卖出、持有、加仓、减仓等投资建议。
如果用户询问投资建议，请拒绝，并说明你只能做材料解读，不能提供投资建议。

请按以下格式输出：

结论：
依据：
涉及主体：
关键数字/时间：
风险提示：
引用来源：
无法判断的部分："""

INVESTMENT_ADVICE_KEYWORDS = (
    "买入",
    "卖出",
    "持有",
    "加仓",
    "减仓",
    "清仓",
    "建仓",
    "推荐股票",
    "值得买",
    "能不能买",
    "该不该买",
    "目标价",
    "投资建议",
)


def is_investment_advice_question(question: str) -> bool:
    return any(keyword in question for keyword in INVESTMENT_ADVICE_KEYWORDS)


def format_context(results: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for idx, result in enumerate(results, start=1):
        page = format_page_range(result.page, result.page_start, result.page_end)
        blocks.append(
            f"[证据{idx}]\n"
            f"来源：{result.source_file}，页码：{page}，chunk_id：{result.chunk_id}\n"
            f"内容：{result.text}"
        )
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, results: list[RetrievalResult]) -> str:
    context = format_context(results) if results else "无"
    return f"""{SYSTEM_INSTRUCTION}

【已检索资料】

{context}

【用户问题】
{question}
"""
