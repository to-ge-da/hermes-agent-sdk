"""OpenAI-compatible shim over the official Cursor Agent SDK.

Cursor has no public ``POST /v1/chat/completions``. Dashboard ``crsr_`` keys
authenticate ``cursor-sdk`` (``Agent.prompt``). This client keeps Hermes as
the harness: it flattens the Hermes transcript + tool schema into one prompt
and maps ``<tool_call>`` blocks back to OpenAI tool_calls, same contract as
``CopilotACPClient``.

``cursor-sdk`` is an optional extra — import is lazy so CI and default
installs do not need the proprietary package.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import (
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
)

CURSOR_MARKER_BASE_URL = "cursor-sdk://local"


def normalize_cursor_model_id(model_id: str) -> str:
    """Return the Hermes-facing slug for a Cursor catalog id."""
    raw = (model_id or "").strip()
    if not raw:
        return ""
    bare = raw[7:] if raw.lower().startswith("cursor-") else raw
    for suffix in ("-high-fast", "-low-fast", "-high", "-fast"):
        if bare.lower().endswith(suffix):
            bare = bare[: -len(suffix)]
            break
    return bare or raw


def expand_cursor_model_ids(raw_ids: list[str]) -> list[str]:
    """Keep official ids and add the short Hermes aliases beside them."""
    out: list[str] = []
    seen: set[str] = set()
    for mid in raw_ids:
        text = str(mid or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
        alias = normalize_cursor_model_id(text)
        if alias and alias not in seen:
            out.append(alias)
            seen.add(alias)
    return out


class _CursorChatCompletions:
    def __init__(self, client: "CursorSDKClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _CursorChatNamespace:
    def __init__(self, client: "CursorSDKClient") -> None:
        self.completions = _CursorChatCompletions(client)


class CursorSDKClient:
    """Minimal OpenAI-client facade for cursor-sdk Agent.prompt()."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> None:
        self.api_key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
        self.base_url = base_url or CURSOR_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self.chat = _CursorChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        response_text = self._run_prompt(prompt_text, model=model or "grok-4.6")
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "grok-4.6",
        )
        if stream:
            from agent.copilot_acp_client import _completion_to_stream_chunks

            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(self, prompt_text: str, *, model: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "CURSOR_API_KEY is not set. Create a key at "
                "https://cursor.com/dashboard/api and put it in ~/.hermes/.env."
            )
        try:
            from cursor_sdk import Agent
        except ImportError as exc:
            raise RuntimeError(
                "The Cursor provider requires the official cursor-sdk package. "
                "Install it into the Hermes environment: pip install cursor-sdk"
            ) from exc

        cwd = str(Path.cwd().resolve())
        result = Agent.prompt(
            prompt_text,
            options={
                "model": model,
                "api_key": self.api_key,
                "local": {"cwd": cwd},
                "mcp_servers": {},
            },
        )
        text = getattr(result, "result", None)
        if isinstance(text, str):
            return text
        if text is None:
            return ""
        return str(text)
