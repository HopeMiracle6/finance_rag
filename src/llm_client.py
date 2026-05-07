from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from src.logging_utils import logger
from src.prompt import is_investment_advice_question
from src.schema import RetrievalResult
from src.utils import format_page_range


class LLMClient:
    def __init__(
        self,
        provider: str = "openai_compatible",
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_model: str | None = None,
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.provider = os.getenv("LLM_PROVIDER", provider)
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.base_model = base_model or os.getenv("QLORA_BASE_MODEL", "Qwen/Qwen3-4B")
        self.adapter_path = os.getenv("QLORA_ADAPTER_PATH") or adapter_path
        self.load_in_4bit = load_in_4bit
        self.temperature = float(os.getenv("LLM_TEMPERATURE", temperature))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", max_tokens))
        self.client: Any = None
        self.local_client: Any = None
        if self.provider == "local_qlora":
            self._init_local_qlora_client()
        elif self.provider == "openai_compatible" and self.api_key and self.api_key != "your_api_key_here":
            self._init_openai_client()

    def _init_openai_client(self) -> None:
        try:
            from openai import OpenAI

            kwargs: dict[str, str] = {"api_key": self.api_key or ""}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)
        except Exception as exc:
            logger.warning(f"OpenAI-compatible client 初始化失败，使用 mock LLM: {exc!r}")
            self.client = None

    def _init_local_qlora_client(self) -> None:
        try:
            from src.local_qlora_client import LocalQLoRAClient

            self.local_client = LocalQLoRAClient(
                base_model=self.base_model,
                adapter_path=self.adapter_path,
                load_in_4bit=self.load_in_4bit,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning(f"Local QLoRA client 初始化失败，使用 mock LLM: {exc!r}")
            self.local_client = None

    def generate(self, prompt: str, question: str, citations: list[RetrievalResult]) -> str:
        if self.provider == "local_qlora" and self.local_client is not None:
            try:
                raw_answer = self.local_client.generate([{"role": "user", "content": prompt}])
                return self._normalize_local_answer(raw_answer, question, citations)
            except Exception as exc:
                logger.warning(f"Local QLoRA 调用失败，使用 mock LLM: {exc!r}")
                return self._mock_generate(question, citations)

        if self.client is None:
            return self._mock_generate(question, citations)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"LLM 调用失败，使用 mock LLM: {exc!r}")
            return self._mock_generate(question, citations)

    def _mock_generate(self, question: str, citations: list[RetrievalResult]) -> str:
        if is_investment_advice_question(question):
            return (
                "结论：无法提供买入、卖出、持有、加仓、减仓等投资建议。\n"
                "依据：我只能基于已检索资料做公告和研报内容解读，不能替代投资决策。\n"
                "涉及主体：无法判断。\n"
                "关键数字/时间：无法判断。\n"
                "风险提示：投资决策需结合个人风险承受能力、完整公开信息和专业意见。\n"
                "引用来源：无。\n"
                "无法判断的部分：公司是否值得投资、具体交易方向或目标价格。"
            )

        if not citations or max((item.score for item in citations), default=0.0) <= 0 or self._evidence_insufficient(question, citations):
            return (
                "结论：仅凭当前资料无法判断。\n"
                "依据：当前检索结果不足以支持回答。\n"
                "涉及主体：无法判断。\n"
                "关键数字/时间：无法判断。\n"
                "风险提示：需要补充更相关的公告、财报或研报材料。\n"
                "引用来源：无。\n"
                "无法判断的部分：问题所需事实依据未在当前资料中出现。"
            )

        evidence_text = " ".join(item.text for item in citations)[:500]
        source_lines = [
            f"{item.source_file}，页码：{format_page_range(item.page, item.page_start, item.page_end)}，chunk_id：{item.chunk_id}"
            for item in citations
        ]
        return (
            f"结论：根据当前资料，问题可参考已检索证据进行判断，但应以原文披露为准。\n"
            f"依据：{evidence_text}\n"
            f"涉及主体：{self._extract_subject(citations)}\n"
            f"关键数字/时间：{self._extract_numbers(citations)}\n"
            f"风险提示：材料可能为阶段性披露，需关注正式报告、经营环境变化、价格波动和竞争加剧等风险。\n"
            f"引用来源：{'; '.join(source_lines)}\n"
            f"无法判断的部分：未被检索资料直接支持的细节不能判断。"
        )

    def _normalize_local_answer(self, raw_answer: str, question: str, citations: list[RetrievalResult]) -> str:
        data = self._extract_first_json(raw_answer)
        if not data:
            return raw_answer.strip()

        conclusion = data.get("结论") or data.get("事件类型") or "根据当前资料，可基于已检索证据进行解读。"
        evidence = data.get("依据") or " ".join(item.text for item in citations)[:500]
        subject = data.get("涉及主体") or self._extract_subject(citations)
        numbers = data.get("关键数字/时间") or data.get("关键金额/时间") or self._extract_numbers(citations)
        risk = data.get("风险提示") or "材料可能存在阶段性和不确定性，请以正式披露文件为准。"
        unknown = (
            data.get("无法判断的部分")
            or data.get("不能判断的部分")
            or "未被检索资料直接支持的细节不能判断。"
        )
        source_lines = [
            f"{item.source_file}，页码：{format_page_range(item.page, item.page_start, item.page_end)}，chunk_id：{item.chunk_id}"
            for item in citations
        ]
        citations_text = "; ".join(source_lines) if source_lines else "无。"
        return (
            f"结论：{conclusion}\n"
            f"依据：{evidence}\n"
            f"涉及主体：{subject}\n"
            f"关键数字/时间：{numbers}\n"
            f"风险提示：{risk}\n"
            f"引用来源：{citations_text}\n"
            f"无法判断的部分：{unknown}"
        )

    @staticmethod
    def _extract_first_json(text: str) -> dict[str, Any] | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        return None
                    return data if isinstance(data, dict) else None
        return None

    @staticmethod
    def _extract_subject(citations: list[RetrievalResult]) -> str:
        for item in citations:
            if "示例公司股份有限公司" in item.text:
                return "示例公司股份有限公司"
        return "以引用资料披露主体为准"

    @staticmethod
    def _extract_numbers(citations: list[RetrievalResult]) -> str:
        import regex as re

        text = "\n".join(item.text for item in citations)
        matches = re.findall(r"\d+(?:\.\d+)?\s*(?:年|亿元|%|％|至|月|日)", text)
        return "、".join(matches[:8]) if matches else "当前资料未提取到明确数字或时间"

    @staticmethod
    def _evidence_insufficient(question: str, citations: list[RetrievalResult]) -> bool:
        import regex as re

        evidence = "\n".join(item.text for item in citations)
        years = re.findall(r"\d{4}", question)
        if any(year not in evidence for year in years):
            return True

        key_terms = (
            "分红",
            "股息",
            "派息",
            "回购",
            "目标价",
            "毛利率",
            "营业收入",
            "研发费用",
            "董事长",
            "实际控制人",
        )
        return any(term in question and term not in evidence for term in key_terms)
