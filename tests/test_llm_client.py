import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import llm_client


def test_is_configured(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert llm_client.is_configured() is False
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    assert llm_client.is_configured() is True


def test_extract_json_parses_model_output(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    class _Msg:    content = '{"board": "科创板", "industry_pe_at_ipo": 38.5}'
    class _Choice: message = _Msg()
    class _Resp:   choices = [_Choice()]

    class _FakeClient:
        def __init__(self, *a, **k): self.chat = self
        @property
        def completions(self): return self
        def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert kwargs["response_format"] == {"type": "json_object"}
            return _Resp()

    monkeypatch.setattr(llm_client, "OpenAI", _FakeClient)
    out = llm_client.extract_json("sys", "user text")
    assert out == {"board": "科创板", "industry_pe_at_ipo": 38.5}


def test_extract_json_requires_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(llm_client.LLMConfigError):
        llm_client.extract_json("sys", "user")
