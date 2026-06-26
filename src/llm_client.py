from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.logging_utils import logger
from src.prompt import is_investment_advice_question
from src.schema import RetrievalResult
from src.utils import format_page_range, project_root


@dataclass
class GenerationResult:
    text: str
    requested_provider: str
    requested_model: str
    actual_provider: str
    actual_model: str
    backend: str
    latency_seconds: float
    fallback_used: bool = False
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_model: str | None = None,
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        strict: bool | None = None,
        reasoning_effort: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> None:
        self._load_dotenv()
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai_compatible")
        if self.provider == "deepseek":
            self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            self.model = model or os.getenv("DEEPSEEK_MODEL") or os.getenv("LLM_MODEL") or "deepseek-v4-pro"
        else:
            self.base_url = base_url or os.getenv("LLM_BASE_URL")
            self.api_key = api_key or os.getenv("LLM_API_KEY")
            self.model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.base_model = base_model or os.getenv("QLORA_BASE_MODEL", "Qwen/Qwen3-4B")
        self.adapter_path = os.getenv("QLORA_ADAPTER_PATH") or adapter_path
        self.load_in_4bit = load_in_4bit
        self.temperature = float(os.getenv("LLM_TEMPERATURE", temperature))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", max_tokens))
        self.strict = self._env_bool("LLM_STRICT", strict if strict is not None else False)
        self.reasoning_effort = reasoning_effort or os.getenv("DEEPSEEK_REASONING_EFFORT")
        self.thinking_enabled = self._env_bool(
            "DEEPSEEK_THINKING",
            thinking_enabled if thinking_enabled is not None else False,
        )
        self.client: Any = None
        self.local_client: Any = None
        self.init_error: Exception | None = None
        self.last_generation: GenerationResult | None = None
        if self.provider == "local_qlora":
            self._init_local_qlora_client()
        elif self.provider in {"deepseek", "openai_compatible"}:
            self._init_openai_client()
        elif self.provider != "mock":
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    @staticmethod
    def _load_dotenv() -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv(project_root() / ".env", override=False)
        except ImportError:
            return

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _init_openai_client(self) -> None:
        if not self.api_key or self.api_key == "your_api_key_here":
            self._handle_init_failure(ValueError(f"{self.provider} 缺少 API Key"))
            return
        try:
            from openai import OpenAI

            kwargs: dict[str, str] = {"api_key": self.api_key or ""}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)
        except Exception as exc:
            self._handle_init_failure(exc)

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
            self._handle_init_failure(exc)

    def _handle_init_failure(self, exc: Exception) -> None:
        self.init_error = exc
        self.client = None
        self.local_client = None
        if self.strict:
            raise RuntimeError(f"LLM 初始化失败，strict 模式禁止 fallback: {exc}") from exc
        logger.warning(f"LLM 初始化失败，后续将使用 mock fallback: {exc!r}")

    def generate(self, prompt: str, question: str, citations: list[RetrievalResult]) -> GenerationResult:
        started_at = time.perf_counter()
        if self.provider == "mock":
            return self._record_generation(
                text=self._mock_generate(question, citations),
                actual_provider="mock",
                actual_model="mock",
                backend="mock_template",
                started_at=started_at,
            )

        if self.provider == "local_qlora" and self.local_client is not None:
            try:
                raw_answer = self.local_client.generate([{"role": "user", "content": prompt}])
                answer = self._normalize_local_answer(raw_answer, question, citations)
                if not answer.strip():
                    raise RuntimeError("本地模型返回空答案")
                adapter = getattr(self.local_client, "adapter_path", None)
                actual_model = f"{self.local_client.base_model}+{adapter}" if adapter else self.local_client.base_model
                return self._record_generation(
                    text=answer,
                    actual_provider="local_qlora",
                    actual_model=str(actual_model),
                    backend="transformers+peft" if adapter else "transformers",
                    started_at=started_at,
                )
            except Exception as exc:
                return self._fallback_or_raise(exc, question, citations, started_at, stage="本地 QLoRA 调用")

        if self.client is None:
            error = self.init_error or RuntimeError(f"{self.provider} client 未初始化")
            return self._fallback_or_raise(error, question, citations, started_at, stage="LLM 初始化")

        try:
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.provider == "deepseek":
                request_kwargs["extra_body"] = {
                    "thinking": {"type": "enabled" if self.thinking_enabled else "disabled"}
                }
                if self.thinking_enabled and self.reasoning_effort:
                    request_kwargs["reasoning_effort"] = self.reasoning_effort
            response = self.client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            answer = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            if not answer.strip():
                reasoning_content = getattr(choice.message, "reasoning_content", None) or ""
                raise RuntimeError(
                    f"API 返回空答案: finish_reason={finish_reason}, reasoning_chars={len(reasoning_content)}"
                )
            usage = getattr(response, "usage", None)
            token_details = getattr(usage, "completion_tokens_details", None)
            return self._record_generation(
                text=answer,
                actual_provider=self.provider,
                actual_model=str(getattr(response, "model", None) or self.model),
                backend="openai_sdk",
                started_at=started_at,
                request_id=getattr(response, "id", None),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                reasoning_tokens=getattr(token_details, "reasoning_tokens", None),
                finish_reason=finish_reason,
            )
        except Exception as exc:
            return self._fallback_or_raise(exc, question, citations, started_at, stage=f"{self.provider} API 调用")

    def _fallback_or_raise(
        self,
        exc: Exception,
        question: str,
        citations: list[RetrievalResult],
        started_at: float,
        stage: str,
    ) -> GenerationResult:
        if self.strict:
            raise RuntimeError(f"{stage}失败，strict 模式禁止 fallback: {exc}") from exc
        logger.warning(f"{stage}失败，使用 mock LLM: {exc!r}")
        return self._record_generation(
            text=self._mock_generate(question, citations),
            actual_provider="mock",
            actual_model="mock",
            backend="mock_template",
            started_at=started_at,
            fallback_used=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    def _record_generation(
        self,
        *,
        text: str,
        actual_provider: str,
        actual_model: str,
        backend: str,
        started_at: float,
        fallback_used: bool = False,
        request_id: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        finish_reason: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> GenerationResult:
        result = GenerationResult(
            text=text,
            requested_provider=self.provider,
            requested_model=self._requested_model(),
            actual_provider=actual_provider,
            actual_model=actual_model,
            backend=backend,
            latency_seconds=round(time.perf_counter() - started_at, 4),
            fallback_used=fallback_used,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=finish_reason,
            error_type=error_type,
            error_message=error_message,
        )
        self.last_generation = result
        return result

    def _requested_model(self) -> str:
        if self.provider == "local_qlora":
            return f"{self.base_model}+{self.adapter_path}" if self.adapter_path else self.base_model
        if self.provider == "mock":
            return "mock"
        return self.model

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
