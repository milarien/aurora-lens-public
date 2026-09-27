"""Deterministic upstream adapter for provider=mock (no HTTP)."""

from __future__ import annotations

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter


class MockUpstreamAdapter(LLMAdapter):
    """Governance-only / local-dev upstream: no external HTTP, deterministic echo."""

    def __init__(self, model: str = "mock", **_kwargs: object) -> None:
        self._model = (model or "mock").strip() or "mock"

    @staticmethod
    def _last_user_text(messages: list[dict[str, str]]) -> str:
        for msg in reversed(messages):
            if (msg.get("role") or "").lower() == "user":
                return str(msg.get("content") or "").strip()
        return ""

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> AdapterResponse:
        user = self._last_user_text(messages)
        text = f"[mock upstream] {user}" if user else "[mock upstream]"
        return AdapterResponse(text=text, model=self._model)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ):
        user = self._last_user_text(messages)
        text = f"[mock upstream] {user}" if user else "[mock upstream]"
        chunk = {"choices": [{"delta": {"content": text}, "index": 0}]}
        yield (chunk, text)
