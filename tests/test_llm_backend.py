import importlib.util
import sys
import types
import os
from pathlib import Path

import pytest


def load_llms(monkeypatch):
    # Stub out third-party modules missing from the test environment
    fake_openai = types.ModuleType("openai")

    class DummyChatCompletions:
        def create(self, **kwargs):
            text = kwargs.get("messages", [{"content": ""}])[-1]["content"]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=f"{text}-reply")
                    )
                ]
            )

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(completions=DummyChatCompletions())

    fake_openai.OpenAI = DummyClient
    fake_openai.AzureOpenAI = DummyClient
    fake_openai.AsyncOpenAI = DummyClient
    fake_openai.AsyncAzureOpenAI = DummyClient
    fake_openai.RateLimitError = Exception
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    fake_tiktoken = types.ModuleType("tiktoken")

    class DummyEncoding:
        def encode(self, text):
            return text.split()

    fake_tiktoken.get_encoding = lambda name: DummyEncoding()
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    for mod in ["pandas", "numpy", "scipy", "umap", "hdbscan"]:
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))

    fake_sklearn = types.ModuleType("sklearn")
    fake_neighbors = types.ModuleType("neighbors")

    class DummyNearestNeighbors:
        def __init__(self, *a, **kw):
            pass

    fake_neighbors.NearestNeighbors = DummyNearestNeighbors
    fake_sklearn.neighbors = fake_neighbors
    fake_sklearn.__path__ = []
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.neighbors", fake_neighbors)

    fake_httpx = types.ModuleType("httpx")

    class DummyURL:
        def __init__(self, path=""):
            self.path = path

        def copy_with(self, path=""):
            self.path = path
            return self

    class DummyRequest:
        def __init__(self, url=None):
            self.url = DummyURL(url)

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

    fake_httpx.Request = DummyRequest
    fake_httpx.Client = DummyClient
    fake_httpx.AsyncClient = DummyClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    package = types.ModuleType("llmcode")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llmcode")]
    sys.modules.setdefault("llmcode", package)

    spec_b = importlib.util.spec_from_file_location(
        "llmcode.backends",
        Path(__file__).resolve().parents[1] / "llmcode" / "backends.py",
    )
    backends = importlib.util.module_from_spec(spec_b)
    sys.modules["llmcode.backends"] = backends
    spec_b.loader.exec_module(backends)

    spec = importlib.util.spec_from_file_location(
        "llmcode.llms", Path(__file__).resolve().parents[1] / "llmcode" / "llms.py"
    )
    llms = importlib.util.module_from_spec(spec)
    sys.modules["llmcode.llms"] = llms
    spec.loader.exec_module(llms)
    return llms


def test_query_llm_single_and_multi(monkeypatch):
    llms = load_llms(monkeypatch)

    class DummyBackend:
        def __init__(self):
            pass

        def query(
            self,
            prompt,
            *,
            system_prompt=None,
            temperature=0.2,
            model="gpt-4",
            **kwargs,
        ):
            key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
                "AALTO_OPENAI_API_KEY"
            )
            return f"{prompt}|{key}"

    monkeypatch.setattr(llms, "OpenAIBackend", DummyBackend)
    monkeypatch.setitem(llms.BACKENDS, "openai", DummyBackend)
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")

    assert llms.query_LLM("hi") == "hi|KEY"
    # list inputs are treated as a single prompt
    assert llms.query_LLM(["a", "b"]) == "['a', 'b']|KEY"


def test_query_llm_env_fallback(monkeypatch):
    llms = load_llms(monkeypatch)

    class DummyBackend:
        def __init__(self):
            pass

        def query(self, prompt, **kwargs):
            return os.environ.get("OPENAI_API_KEY") or os.environ.get(
                "AALTO_OPENAI_API_KEY"
            )

    monkeypatch.setattr(llms, "OpenAIBackend", DummyBackend)
    monkeypatch.setitem(llms.BACKENDS, "openai", DummyBackend)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AALTO_OPENAI_API_KEY", "AALTOKEY")

    assert llms.query_LLM("prompt") == "AALTOKEY"
