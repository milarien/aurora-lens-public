"""OpenAI adapter — OpenAI-compatible API integration."""

from __future__ import annotations

import json
import httpx

from .base import LLMAdapter, AdapterResponse


def _parse_sse_line(line: bytes) -> dict | None:
    """Parse SSE data line. Returns dict if line starts with 'data: ', else None."""
    if not line.startswith(b"data: "):
        return None
    payload = line[6:].decode("utf-8", errors="replace").strip()
    if payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


class OpenAIUpstreamAdapter(LLMAdapter):
    """Adapter that forwards requests to an OpenAI-compatible API.

    Uses httpx to call the upstream. Transport-only.

    Notes:
    - Network exceptions (timeout/connect/status) are allowed to propagate as httpx exceptions.
      The proxy hot-path catches them and returns HTTP 200 with aurora.error codes.
    - Response-shape errors are converted to KeyError/IndexError so the proxy can map them to
      aurora.error="upstream_invalid_response".
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4",
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        # Strip so env-sourced keys with trailing newlines remain valid HTTP header values.
        self._api_key = str(api_key or "").strip()
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_s = float(timeout_s)

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AdapterResponse:
        """Send messages upstream unchanged."""
        url = f"{self._base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError as e:
                # Treat upstream invalid JSON as an "invalid response shape"
                raise KeyError("upstream_returned_invalid_json") from e

        # Response parsing: normalize response-shape errors into KeyError/IndexError.
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise KeyError("upstream_missing_choices_message_content") from e

        model = data.get("model", self._model)
        usage = data.get("usage")

        return AdapterResponse(text=text, model=model, usage=usage)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ):
        """Stream completion from OpenAI-compatible API. Yields (chunk_dict, content_delta)."""
        url = f"{self._base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        buffer = b""
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        for line in event.split(b"\n"):
                            data = _parse_sse_line(line)
                            if data is None:
                                continue
                            if data.get("done"):
                                return
                            try:
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content") or ""
                                if isinstance(content, str):
                                    yield (data, content)
                            except (KeyError, IndexError, TypeError):
                                yield (data, "")
