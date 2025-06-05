from __future__ import annotations

from abc import ABC, abstractmethod

import openai


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

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

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
        messages = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content.strip()


class LocalLlamaBackend(LLMBackend):
    """Placeholder for a future on-prem model."""

    def query(self, *args, **kwargs) -> str:  # pragma: no cover - stub
        raise NotImplementedError("Local llama backend not implemented yet.")
