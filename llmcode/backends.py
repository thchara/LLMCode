from __future__ import annotations

from abc import ABC, abstractmethod
import os
import time

import openai
import httpx

openai2aalto = {
    "gpt-3.5-turbo": "/v1/chat",
    "gpt-4-turbo": "/v1/openai/gpt4-turbo/chat/completions",
    "gpt-4o": "/v1/openai/gpt4o/chat/completions",
    "text-embedding-3-large": "/v1/openai/text-embedding-3-large/embeddings",
    "text-embedding-ada-002": "/v1/openai/ada-002/embeddings",
}

current_openai_model: str | None = None


def update_base_url_for_aalto(request: httpx.Request) -> None:
    """Adjust OpenAI request path for Aalto's Azure gateway."""
    if request.url.path == "/chat/completions":
        if current_openai_model not in openai2aalto:
            raise Exception(
                f"Model {current_openai_model} not available via the Aalto API"
            )
        request.url = request.url.copy_with(path=openai2aalto[current_openai_model])


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

    def __init__(self):
        self.client = None

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
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "AALTO_OPENAI_API_KEY"
        )
        assert api_key, (
            "you must set the `OPENAI_API_KEY` or `AALTO_OPENAI_API_KEY`"
            " environment variable."
        )

        use_aalto = "AALTO_OPENAI_API_KEY" in os.environ and not os.environ.get(
            "OPENAI_API_KEY"
        )
        if self.client is None:
            if use_aalto:
                self.client = openai.OpenAI(
                    base_url="https://aalto-openai-apigw.azure-api.net",
                    api_key=False,
                    default_headers={"Ocp-Apim-Subscription-Key": api_key},
                    http_client=httpx.Client(
                        event_hooks={"request": [update_base_url_for_aalto]}
                    ),
                )
            else:
                self.client = openai.OpenAI(api_key=api_key)

        global current_openai_model
        current_openai_model = model

        messages = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        success = False
        while not success:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    stream=kwargs.get("stream", False),
                )
                success = True
            except openai.RateLimitError:
                print("Rate limit error! Will retry in 5 seconds")
                time.sleep(5)

        if kwargs.get("stream", False):
            text = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    text += delta
            return text.strip()

        return response.choices[0].message.content.strip()


class LocalLlamaBackend(LLMBackend):
    """Placeholder for a future on-prem model."""

    def query(self, *args, **kwargs) -> str:  # pragma: no cover - stub
        raise NotImplementedError("Local llama backend not implemented yet.")
