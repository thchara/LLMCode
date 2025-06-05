import json
import datetime
import types
from pathlib import Path

import pytest

from llmcode import logging_utils
from tests.test_llm_backend import load_llms


def _patch_datetime(monkeypatch):
    class DDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2001, 1, 2)

    class DDateTime(datetime.datetime):
        @classmethod
        def utcnow(cls):
            return cls(2001, 1, 2, 3, 4, 5)

    monkeypatch.setattr(
        logging_utils,
        "datetime",
        types.SimpleNamespace(date=DDate, datetime=DDateTime),
    )


@pytest.mark.usefixtures("monkeypatch")
def test_log_prompt_creates_file(tmp_path, monkeypatch):
    _patch_datetime(monkeypatch)
    monkeypatch.setenv("LLMCODE_LOG_DIR", str(tmp_path))
    logging_utils.log_prompt("p", "r", model="m", temperature=0.1, system_prompt="s")
    path = tmp_path / "llmcode_2001-01-02.jsonl"
    data = [json.loads(path.read_text(encoding="utf-8"))]
    assert data == [
        {
            "ts": "2001-01-02T03:04:05Z",
            "model": "m",
            "temperature": 0.1,
            "system_prompt": "s",
            "prompt": "p",
            "response": "r",
        }
    ]


def test_log_prompt_no_log_env(tmp_path, monkeypatch):
    _patch_datetime(monkeypatch)
    monkeypatch.setenv("LLMCODE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LLMCODE_NO_LOG", "1")
    logging_utils.log_prompt("p", "r", model="m", temperature=0.1, system_prompt="s")
    assert not (tmp_path / "llmcode_2001-01-02.jsonl").exists()


def test_query_llm_logs(monkeypatch):
    llms = load_llms(monkeypatch)

    class DummyBackend:
        def __init__(self, api_key):
            self.api_key = api_key

        def query(self, prompt, **kwargs):
            return f"{prompt}|{self.api_key}"

    monkeypatch.setattr(llms, "OpenAIBackend", DummyBackend)
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")

    records = []

    def fake_log(prompt, response, *, model, temperature, system_prompt):
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
            }
        )

    monkeypatch.setattr(llms, "log_prompt", fake_log)

    llms.query_LLM("hi", model="m", temperature=0.5, system_message="sys")

    assert records == [
        {
            "prompt": "hi",
            "response": "hi|KEY",
            "model": "m",
            "temperature": 0.5,
            "system_prompt": "sys",
        }
    ]
