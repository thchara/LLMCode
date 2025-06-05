from __future__ import annotations

from abc import ABC, abstractmethod
import os

try:
    import openai  # for OpenAIBackend only
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openai = None


class LLMBackend(ABC):
    """Strategy base class for all LLM engines."""

    @abstractmethod
    def query(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        model: str = "gpt-4",
        **kwargs,
    ) -> str:
        pass


class OpenAIBackend(LLMBackend):
    """Uses openai.ChatCompletion under the hood."""

    def __init__(self) -> None:
        pass

    def query(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        model: str = "gpt-4",
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        **kwargs,
    ) -> str:
        if openai is None:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for OpenAI backend")

        import time

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AALTO_OPENAI_API_KEY")
        assert (
            api_key
        ), "you must set the `OPENAI_API_KEY` or `AALTO_OPENAI_API_KEY` environment variable."

        client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        messages = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        for _ in range(3):
            try:
                response = client.chat.completions.create(**payload)
                return response.choices[0].message.content.strip()
            except Exception as exc:  # pragma: no cover - network errors
                last = exc
                time.sleep(2)
        raise RuntimeError(f"OpenAI backend failed: {last}")


class LocalLlamaBackend(LLMBackend):
    """Call a local LM Studio/llama.cpp server running the OpenAI
    compatible REST API (default: http://127.0.0.1:1234)."""

    BASE_URL = os.getenv(
        "LLMCODE_LLAMA_URL", "http://127.0.0.1:1234/v1/chat/completions"
    )

    def query(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        model: str = "local-llama",
        **kwargs,
    ) -> str:
        import httpx, json, time

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": prompt},
            ],
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        payload["messages"] = [m for m in payload["messages"] if m]
        for _ in range(3):
            try:
                resp = httpx.post(self.BASE_URL, json=payload, timeout=90)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # pragma: no cover - network failures
                last = exc
                time.sleep(2)
        raise RuntimeError(f"Local Llama backend failed: {last}")  # noqa: BLE001
