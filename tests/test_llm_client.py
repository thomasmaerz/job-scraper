from types import SimpleNamespace

import pytest

import llm_client


def test_generate_content_uses_custom_model_chain_in_order_on_rate_limits(monkeypatch):
    calls = []

    class RateLimitError(Exception):
        pass

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RateLimitError("429 rate limit")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_args, **_kwargs: None)

    client = llm_client.LLMClient(
        model="gemini",
        api_key="test-key",
        max_rpm=100,
        max_retries=1,
        retry_base_delay=0,
        daily_budget=0,
        request_delay=0,
        model_chain=[
            "gemini/gemini-3-flash-preview",
            "gemini/gemma-4-31b-it",
            "gemini/gemini-3.1-flash-lite",
        ],
    )

    result = client.generate_content(prompt="hello")

    assert result == "ok"
    assert [call["model"] for call in calls] == [
        "gemini/gemini-3-flash-preview",
        "gemini/gemma-4-31b-it",
        "gemini/gemini-3.1-flash-lite",
    ]


def test_generate_content_forwards_reasoning_effort(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="42"))]
        )

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)

    client = llm_client.LLMClient(
        model="gemini",
        api_key="test-key",
        max_rpm=100,
        max_retries=0,
        retry_base_delay=0,
        daily_budget=0,
        request_delay=0,
    )

    result = client.generate_content(prompt="hello", reasoning_effort="medium")

    assert result == "42"
    assert calls[0]["reasoning_effort"] == "medium"


def test_generate_content_omits_reasoning_effort_for_gemma_fallback(monkeypatch):
    calls = []

    class RateLimitError(Exception):
        pass

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RateLimitError("429 rate limit")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_args, **_kwargs: None)

    client = llm_client.LLMClient(
        model="gemini",
        api_key="test-key",
        max_rpm=100,
        max_retries=1,
        retry_base_delay=0,
        daily_budget=0,
        request_delay=0,
        model_chain=["gemini/gemini-3.1-flash-lite", "gemini/gemma-4-26b-a4b-it"],
    )

    assert client.generate_content(prompt="hello", reasoning_effort="low") == "ok"
    assert calls[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in calls[1]


def test_generate_content_omits_temperature_when_not_provided(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)

    client = llm_client.LLMClient(
        model="gemini",
        api_key="test-key",
        max_rpm=100,
        max_retries=0,
        retry_base_delay=0,
        daily_budget=0,
        request_delay=0,
    )

    result = client.generate_content(prompt="hello")

    assert result == "ok"
    assert "temperature" not in calls[0]


def test_job_scoring_model_chain_prefers_flash_lite_then_gemma_31b():
    import config

    assert config.JOB_SCORING_MODEL_CHAIN == [
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemma-4-31b-it",
        "gemini/gemini-3-flash-preview",
        "gemini/gemma-4-26b-a4b-it",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
    ]
