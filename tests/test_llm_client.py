from types import SimpleNamespace

import pytest

from src.llm_client import LLMClient


class FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def test_deepseek_strict_requires_api_key(monkeypatch):
    monkeypatch.setattr(LLMClient, "_load_dotenv", staticmethod(lambda: None))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="strict 模式禁止 fallback"):
        LLMClient(provider="deepseek", api_key=None, strict=True)


def test_deepseek_records_actual_response_metadata():
    response = SimpleNamespace(
        id="req_test_001",
        model="deepseek-v4-pro",
        choices=[SimpleNamespace(message=SimpleNamespace(content="真实 API 回答"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    client = LLMClient(provider="deepseek", api_key="test-key", strict=True)
    completions = FakeCompletions(response=response)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = client.generate("prompt", question="question", citations=[])

    assert result.text == "真实 API 回答"
    assert result.actual_provider == "deepseek"
    assert result.actual_model == "deepseek-v4-pro"
    assert result.backend == "openai_sdk"
    assert result.fallback_used is False
    assert result.request_id == "req_test_001"
    assert result.total_tokens == 15
    assert completions.last_kwargs["model"] == "deepseek-v4-pro"
    assert completions.last_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_non_strict_failure_is_explicitly_marked_as_fallback():
    client = LLMClient(provider="deepseek", api_key="test-key", strict=False)
    completions = FakeCompletions(error=TimeoutError("timeout"))
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = client.generate("prompt", question="无法回答的问题", citations=[])

    assert result.actual_provider == "mock"
    assert result.actual_model == "mock"
    assert result.fallback_used is True
    assert result.error_type == "TimeoutError"


def test_strict_failure_does_not_return_mock():
    client = LLMClient(provider="deepseek", api_key="test-key", strict=True)
    completions = FakeCompletions(error=TimeoutError("timeout"))
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(RuntimeError, match="strict 模式禁止 fallback"):
        client.generate("prompt", question="question", citations=[])
