"""Small shared helpers for the curated public test suite."""

from __future__ import annotations

import json
from pathlib import Path

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter


class MockAdapter(LLMAdapter):
    """Return canned responses while recording calls made by ``Lens``."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["I don't know."]
        self._call_count = 0
        self.last_pef_context = ""
        self.last_messages: list[dict[str, str]] | None = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> AdapterResponse:
        del kwargs
        for message in messages:
            if message.get("role") == "system":
                self.last_pef_context = message.get("content", "")
        user_assistant = [
            message for message in messages if message.get("role") in {"user", "assistant"}
        ]
        self.last_messages = (
            user_assistant[:-1] if len(user_assistant) > 1 else user_assistant
        )
        index = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return AdapterResponse(text=self._responses[index], model="mock-1")


def _read_last_jsonl_object(path: Path) -> dict[str, object]:
    """Read the last non-empty JSON object from a JSON Lines file."""

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"JSON Lines file is empty: {path}")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise TypeError(f"Last JSON Lines value is not an object: {path}")
    return payload
