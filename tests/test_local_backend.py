import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_llms(monkeypatch):
    """Reuse the loader from test_llm_backend without importing the module."""
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


def test_local_backend_routing(monkeypatch):
    llms = load_llms(monkeypatch)
    monkeypatch.setenv("LLMCODE_BACKEND", "local")

    def fake_query(self, prompt, **kw):
        return "pong"

    monkeypatch.setattr(llms.LocalLlamaBackend, "query", fake_query)

    assert llms.query_LLM("ping") == "pong"
