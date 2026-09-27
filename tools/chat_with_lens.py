#!/usr/bin/env python3
"""Minimal interactive CLI for the Aurora-Lens proxy.

Connects to ``http://localhost:8081``, sends each line as a chat turn, prints
the assistant reply. Type ``quit`` to exit.

    python tools/chat_with_lens.py
"""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_BASE_URL = "http://localhost:8081"
_CHAT_URL = f"{_BASE_URL}/v1/chat/completions"
_HEALTH_URL = f"{_BASE_URL}/health"
_MODEL = "openclaw"
_TIMEOUT = 180.0


def _check_proxy() -> None:
    try:
        with urlopen(_HEALTH_URL, timeout=5) as r:
            health = json.loads(r.read().decode("utf-8"))
    except (OSError, URLError) as e:
        raise SystemExit(
            f"Proxy not reachable at {_BASE_URL}. Start it with: python run_aurora_lens.py\n{e}"
        ) from e
    if health.get("status") not in ("ok", "degraded"):
        print(f"WARNING: /health status={health.get('status')!r}", file=sys.stderr)


def _post_chat(messages: list[dict[str, str]], pef_context_id: str) -> dict[str, Any]:
    body = {
        "model": _MODEL,
        "messages": messages,
        "stream": False,
        # aurora_session_id is the legacy field name; pef_context_id and
        # session_id are the same store key on the server (see
        # docs/FRAME_LIFECYCLE_INVARIANT.md), so this still round-trips
        # correctly even though the header below is the canonical name.
        "aurora_session_id": pef_context_id,
    }
    payload = json.dumps(body).encode("utf-8")
    req = Request(
        _CHAT_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer chat",
            "x-aurora-pef-context-id": pef_context_id,
        },
        method="POST",
    )
    with urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pef_context_id_from_response(resp: dict[str, Any]) -> str | None:
    """Prefer the canonical continuity handle the server assigned/echoed."""
    aurora = resp.get("aurora") or {}
    cid = aurora.get("pef_context_id") or aurora.get("session_id")
    return str(cid) if cid else None


def _assistant_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    return content.strip() if isinstance(content, str) else ""


def main() -> None:
    _check_proxy()
    # Locally-minted only until the server assigns/echoes its own pef_context_id
    # (first response below); subsequent turns use the server's value.
    pef_context_id = f"chat-{uuid.uuid4().hex[:12]}"
    history: list[dict[str, str]] = []

    print("Aurora-Lens chat (proxy on localhost:8081). Type 'quit' to exit.\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            break

        history.append({"role": "user", "content": line})
        try:
            resp = _post_chat(history, pef_context_id)
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            print(f"Lens: [HTTP {e.code}] {detail}\n", file=sys.stderr)
            history.pop()
            continue
        except URLError as e:
            print(f"Lens: [error] {e}\n", file=sys.stderr)
            history.pop()
            continue

        server_cid = _pef_context_id_from_response(resp)
        if server_cid:
            pef_context_id = server_cid

        answer = _assistant_text(resp)
        if not answer:
            aurora = resp.get("aurora") or {}
            answer = f"[no content] governance={aurora.get('governance', '?')!r}"

        print(f"Lens: {answer}\n")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
