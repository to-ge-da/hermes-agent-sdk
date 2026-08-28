"""OpenAI-compatible shim over the official Cursor Agent SDK.

Cursor has no public POST /v1/chat/completions. Dashboard crsr_ keys talk to
cursor-sdk. Hermes stays the harness: every Hermes tool is described with its
real JSON schema, the model emits <tool_call> JSON, and Hermes executes it —
same shape as Copilot ACP, without ACP wording (that makes Cursor walk the
repo with its own tools).

Cursor-native tools stay off (tools=[], mcp_servers={}). That is not a Hermes
allowlist; it is "do not give Cursor a second filesystem."

A live Agent is reused across turns in this process. Cold start sends a
windowed transcript; later turns send only the latest user line plus any
trailing tool results. Do not flatten a 300k resume into Agent.prompt.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import _extract_tool_calls_from_text

log = logging.getLogger(__name__)

CURSOR_MARKER_BASE_URL = "cursor-sdk://local"

# Size cap only — not a task-specific stub. Cursor Agent.prompt/create hangs
# if the first send is a 300k Hermes resume.
_COLD_TRANSCRIPT_CHARS = 48_000
_TOOL_DESC_CHARS = 800
_SYSTEM_CHARS = 12_000

# One Cursor Agent per Hermes session. A slot also holds the in-flight Run
# so a Hermes retry/timeout cannot send() while that run is still open.
_slots: dict[str, dict[str, Any]] = {}
_slots_guard = threading.Lock()


def _slot_record(slot: str) -> dict[str, Any]:
    with _slots_guard:
        rec = _slots.get(slot)
        if rec is None:
            rec = {"agent": None, "run": None, "lock": threading.Lock()}
            _slots[slot] = rec
        return rec


def _cancel_run(rec: dict[str, Any]) -> None:
    run = rec.get("run")
    rec["run"] = None
    if run is None:
        return
    cancel = getattr(run, "cancel", None)
    if not callable(cancel):
        return
    try:
        cancel()
    except Exception:
        pass


def _close_agent(agent: Any) -> None:
    """Dispose the SDK agent — it owns a cursor-sdk-bridge child process."""
    if agent is None:
        return
    close = getattr(agent, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        pass


def _drop_agent(rec: dict[str, Any]) -> None:
    _cancel_run(rec)
    agent = rec.get("agent")
    rec["agent"] = None
    _close_agent(agent)


_RUN_FAILURE_STATUSES = frozenset({"error", "failed", "cancelled", "canceled"})


def _run_status(*sources: Any) -> str:
    for source in sources:
        status = getattr(source, "status", None)
        if isinstance(status, str) and status.strip():
            return status.strip().lower()
    return ""


def _run_failure_detail(*sources: Any) -> str:
    for source in sources:
        if source is None:
            continue
        for attr in ("error", "message", "detail"):
            value = getattr(source, attr, None)
            if value is None:
                continue
            text = value if isinstance(value, str) else str(value)
            text = text.strip()
            if text and text != "None":
                return text
    return ""


def _raise_for_failed_run(*sources: Any) -> None:
    """A thrown exception means the run never started; ``status == "error"``
    means it ran and failed. Returning that as empty assistant text would
    trigger the Hermes empty-response retry storm on a dead run."""
    status = _run_status(*sources)
    if status not in _RUN_FAILURE_STATUSES:
        return
    detail = _run_failure_detail(*sources)
    suffix = f": {detail}" if detail else " (cursor-sdk returned no error detail)"
    raise RuntimeError(f"Cursor run ended with status '{status}'{suffix}")


_CURSOR_AUTH_HINTS = (
    "401",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "forbidden",
    "token expired",
    "access denied",
)


def _cursor_startup_error(exc: Exception, *, phase: str) -> RuntimeError:
    """Map cursor-sdk startup failures onto actionable Hermes errors.

    The original SDK text stays in the message so the Hermes error
    classifier (message-matched auth patterns) still routes 401s to the
    auth/failover path.
    """
    text = str(exc).strip() or exc.__class__.__name__
    if any(hint in text.lower() for hint in _CURSOR_AUTH_HINTS):
        from hermes_constants import display_hermes_home

        return RuntimeError(
            "Cursor rejected the API key (authentication failed). Check "
            f"CURSOR_API_KEY in {display_hermes_home()}/.env or regenerate it at "
            "https://cursor.com/dashboard/api. "
            f"cursor-sdk said: {text}"
        )
    return RuntimeError(f"Cursor Agent SDK {phase} failed: {text}")


def _cursor_token_usage(source: Any) -> Any:
    """Best-effort TokenUsage from a Run, RunResult, or None."""
    if source is None:
        return None
    usage = getattr(source, "usage", None)
    return usage if usage is not None else None


def _openai_usage_from_cursor(token_usage: Any) -> SimpleNamespace:
    """Map cursor-sdk TokenUsage onto the OpenAI chat-completions usage shape.

    Cursor's ``input_tokens`` excludes cache; OpenAI's ``prompt_tokens``
    includes it. Hermes ``normalize_usage`` subtracts the details, so we
    add cache back into ``prompt_tokens``. Missing usage (the #88212 stub)
    stays zeros.
    """
    if token_usage is None:
        return SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
    input_tokens = int(getattr(token_usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(token_usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(token_usage, "cache_read_tokens", 0) or 0)
    cache_write = int(getattr(token_usage, "cache_write_tokens", 0) or 0)
    total = int(getattr(token_usage, "total_tokens", 0) or 0)
    reasoning = getattr(token_usage, "reasoning_tokens", None)
    prompt_tokens = input_tokens + cache_read + cache_write
    if not total:
        total = prompt_tokens + output_tokens
    details = SimpleNamespace(cached_tokens=cache_read, cache_write_tokens=cache_write)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=total,
        prompt_tokens_details=details,
        cache_creation_input_tokens=cache_write,
    )
    if reasoning:
        usage.completion_tokens_details = SimpleNamespace(
            reasoning_tokens=int(reasoning)
        )
    return usage


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url" or part.get("image_url"):
                    parts.append("[image]")
                else:
                    parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    if isinstance(content, dict) and content.get("_multimodal"):
        return str(content.get("text_summary") or content.get("text") or "[image]")
    return str(content or "")


def _unwrap_stored_content(content: Any) -> Any:
    """Hermes DB sometimes stores multimodal as ``\\0json:[...]``."""
    if isinstance(content, bytes):
        content = content.decode("utf-8", "replace")
    if isinstance(content, str):
        text = content
        if text.startswith("\0"):
            text = text[1:]
        if text.startswith("json:"):
            try:
                return json.loads(text[5:])
            except json.JSONDecodeError:
                return content
        return content
    return content


def _iter_image_urls(content: Any) -> list[str]:
    urls: list[str] = []
    content = _unwrap_stored_content(content)
    if isinstance(content, str):
        text = content.strip()
        if text.startswith("{") and "_multimodal" in text:
            try:
                content = json.loads(text)
            except json.JSONDecodeError:
                return urls
        elif text.startswith("data:image") or (
            text.startswith("/") and _looks_like_image_path(text)
        ):
            urls.append(text)
            return urls
    if isinstance(content, dict) and content.get("_multimodal"):
        content = content.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            image = part.get("image_url")
            if isinstance(image, dict):
                url = image.get("url")
                if isinstance(url, str) and url.strip():
                    urls.append(url.strip())
            elif isinstance(image, str) and image.strip():
                urls.append(image.strip())
            elif part.get("type") == "image_url" and isinstance(part.get("url"), str):
                urls.append(part["url"].strip())
    return urls


def _looks_like_image_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))


def extract_cursor_images(
    messages: list[dict[str, Any]] | None,
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Pull Hermes image_url / native-vision parts for Cursor SDKImage."""
    found: list[str] = []
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        for url in _iter_image_urls(message.get("content")):
            if url not in found:
                found.append(url)
            if len(found) >= limit:
                break
        if len(found) < limit:
            for url in _iter_tool_call_image_urls(message):
                if url not in found:
                    found.append(url)
                if len(found) >= limit:
                    break
        if len(found) >= limit:
            break
    found.reverse()
    images: list[dict[str, Any]] = []
    for url in found:
        converted = _cursor_image_payload(url)
        if converted:
            images.append(converted)
    return images


def _iter_tool_call_image_urls(message: dict[str, Any]) -> list[str]:
    """Native vision results are often text-only; the path is in the tool call."""
    urls: list[str] = []
    calls = message.get("tool_calls")
    if calls is None:
        return urls
    if not isinstance(calls, list):
        calls = [calls]
    for call in calls:
        fn = None
        if isinstance(call, dict):
            fn = call.get("function")
        else:
            fn = getattr(call, "function", None)
        name = ""
        raw: Any = None
        if isinstance(fn, dict):
            name = str(fn.get("name") or "")
            raw = fn.get("arguments")
        elif fn is not None:
            name = str(getattr(fn, "name", "") or "")
            raw = getattr(fn, "arguments", None)
        if name not in {"vision_analyze", "vision_analyze_tool"}:
            continue
        args: Any = raw
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        url = args.get("image_url")
        if isinstance(url, str) and url.strip():
            urls.append(url.strip())
    return urls


def _cursor_image_payload(url: str) -> dict[str, Any] | None:
    text = (url or "").strip()
    if not text:
        return None
    if text.startswith("data:"):
        header, _, b64 = text.partition(",")
        if not b64:
            return None
        mime = "image/png"
        if header.startswith("data:") and ";" in header:
            mime = header[5:].split(";", 1)[0] or mime
        return {"data": b64, "mime_type": mime}
    if text.startswith("http://") or text.startswith("https://"):
        return {"url": text}
    path = Path(text).expanduser()
    if path.is_file():
        return {"path": str(path)}
    return None


def window_cursor_messages(
    messages: list[dict[str, Any]],
    *,
    budget: int = _COLD_TRANSCRIPT_CHARS,
) -> list[dict[str, Any]]:
    """Keep system (truncated) plus a recent tail that includes the last user."""
    typed = [m for m in messages if isinstance(m, dict)]
    if not typed:
        return []
    systems = [m for m in typed if str(m.get("role") or "").lower() == "system"]
    rest = [m for m in typed if str(m.get("role") or "").lower() != "system"]
    kept_sys: list[dict[str, Any]] = []
    for sys_msg in systems[:1]:
        text = _message_text(sys_msg)
        if len(text) > _SYSTEM_CHARS:
            text = text[:_SYSTEM_CHARS] + "\n…[system truncated]"
        kept_sys.append({"role": "system", "content": text})
    if not rest:
        return kept_sys
    last_user_i = len(rest) - 1
    for i in range(len(rest) - 1, -1, -1):
        if str(rest[i].get("role") or "").lower() == "user":
            last_user_i = i
            break
    tail: list[dict[str, Any]] = [rest[last_user_i]]
    used = len(_message_text(rest[last_user_i]))
    i = last_user_i - 1
    while i >= 0 and used < budget:
        msg = rest[i]
        n = len(_message_text(msg))
        if tail and used + n > budget:
            break
        tail.append(msg)
        used += n
        i -= 1
    tail.reverse()
    return kept_sys + tail


def hermes_tools_spec(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Pass through every Hermes tool with its real parameter schema."""
    if not isinstance(tools, list):
        return []
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip() or name in seen:
            continue
        desc = str(fn.get("description") or "")
        if len(desc) > _TOOL_DESC_CHARS:
            desc = desc[:_TOOL_DESC_CHARS] + "…"
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        specs.append(
            {
                "name": name.strip(),
                "description": desc,
                "parameters": params,
            }
        )
        seen.add(name.strip())
    return specs


def format_hermes_cursor_prompt(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    resume: bool = False,
) -> str:
    """Hermes-as-harness prompt. No ACP wording. Full tool schemas."""
    sections: list[str] = [
        "You are the Hermes agent. Hermes executes tools locally.",
        "If you need a tool, emit one or more "
        "<tool_call>{\"id\":\"call_1\",\"type\":\"function\","
        "\"function\":{\"name\":\"NAME\",\"arguments\":\"{...}\"}}</tool_call> "
        "blocks. arguments must be a JSON string. Do not use Cursor, ACP, or "
        "workspace tools — only the Hermes tools listed below.",
        "If no tool is needed, answer in plain text.",
    ]
    if model:
        sections.append(f"Model hint: {model}")
    specs = hermes_tools_spec(tools)
    if specs:
        sections.append(
            "Hermes tools (OpenAI function schema):\n"
            + json.dumps(specs, ensure_ascii=False)
        )
    if resume:
        delta = _resume_delta(messages)
        blocked = _blocked_reads(messages)
        if blocked:
            sections.append(
                "Hermes already returned these paths. Do not read_file them again:\n"
                + blocked
            )
        if delta:
            sections.append("New turn:\n\n" + delta)
        sections.append("Continue from the new turn. Do not re-open files you already have.")
        return "\n\n".join(sections)

    windowed = window_cursor_messages(messages)
    transcript: list[str] = []
    for message in windowed:
        role = str(message.get("role") or "context").strip().lower()
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }.get(role, "Context")
        text = _message_text(message).strip()
        if text:
            transcript.append(f"{label}:\n{text}")
    if transcript:
        sections.append("Conversation:\n\n" + "\n\n".join(transcript))
    sections.append("Continue from the latest user request.")
    return "\n\n".join(sections)


def _resume_delta(messages: list[dict[str, Any]]) -> str:
    """Latest user text plus trailing tool results (the Hermes turn delta)."""
    typed = [m for m in messages if isinstance(m, dict)]
    if not typed:
        return ""
    last_user = None
    last_user_i = 0
    for i in range(len(typed) - 1, -1, -1):
        if str(typed[i].get("role") or "").lower() == "user":
            last_user = typed[i]
            last_user_i = i
            break
    parts: list[str] = []
    if last_user is not None:
        parts.append("User:\n" + _message_text(last_user).strip())
        for message in typed[last_user_i + 1 :]:
            role = str(message.get("role") or "").lower()
            text = _message_text(message).strip()
            if not text:
                continue
            if role == "tool":
                parts.append("Tool:\n" + text)
            elif role == "assistant":
                parts.append("Assistant:\n" + text)
    else:
        parts.append(_message_text(typed[-1]).strip())
    return "\n\n".join(p for p in parts if p)


# Catalog SKU suffixes, longest first so "-low-fast" wins over "-fast".
_CURSOR_VARIANT_SUFFIXES = ("-high-fast", "-low-fast", "-high", "-low", "-fast")

# Nous #88212 pin: suffix-less grok-* is high / not-fast (reasoning tokens).
_GROK_DEFAULT_REASONING = "high"
_GROK_DEFAULT_FAST = "false"


def _cursor_catalog_bare(model_id: str) -> str:
    raw = (model_id or "").strip()
    if raw.lower().startswith("cursor-"):
        return raw[7:]
    return raw


def _cursor_variant(model_id: str) -> str:
    """SKU suffix of a catalog id ("" when bare/unknown)."""
    lowered = (model_id or "").strip().lower()
    for suffix in _CURSOR_VARIANT_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix[1:]
    return ""


def _cursor_sdk_slot_flavor(model_id: str) -> str:
    """Agent-slot tag so high vs high-fast do not reuse each other's Agent."""
    sel = cursor_sdk_model(model_id)
    if not isinstance(sel, dict):
        return "plain"
    params = {p["id"]: str(p["value"]).lower() for p in sel.get("params") or []}
    reasoning = params.get("reasoning") or _GROK_DEFAULT_REASONING
    speed = "fast" if params.get("fast") == "true" else "nofast"
    return f"{reasoning}-{speed}"


def normalize_cursor_model_id(model_id: str) -> str:
    """Return the Hermes-facing slug for a Cursor catalog id."""
    raw = (model_id or "").strip()
    if not raw:
        return ""
    bare = _cursor_catalog_bare(raw)
    for suffix in _CURSOR_VARIANT_SUFFIXES:
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


# SKU params per catalog variant. Bare grok-4.6 keeps the tuned default
# (high, not fast) — see cursor_sdk_model. Explicit variants map to their
# own SKU so picking a cheap "-low-fast" in /model does not silently run
# the most expensive configuration.
_GROK_VARIANT_PARAMS: dict[str, list[dict[str, str]]] = {
    "high-fast": [
        {"id": "fast", "value": "true"},
        {"id": "reasoning", "value": "high"},
    ],
    "low-fast": [
        {"id": "fast", "value": "true"},
        {"id": "reasoning", "value": "low"},
    ],
    "fast": [
        {"id": "fast", "value": "true"},
        {"id": "reasoning", "value": "high"},
    ],
    "low": [
        {"id": "fast", "value": "false"},
        {"id": "reasoning", "value": "low"},
    ],
    "high": [
        {"id": "fast", "value": "false"},
        {"id": "reasoning", "value": "high"},
    ],
}


def cursor_sdk_model(model_id: str) -> str | dict[str, Any]:
    """SDK inference id plus ``fast`` / ``reasoning`` from the catalog id.

    ``Agent.prompt`` accepts ``grok-4.6``, not catalog
    ``cursor-grok-4.6-high-fast``. Suffixes on the picker id become params
    *before* the strip. Bare ``grok-*`` keeps the Nous #88212 pin (high /
    not-fast). Explicit variants (``-low-fast``, ``-high-fast``, ``-fast``,
    ``-low``, ``-high``) map to matching params; ``-fast`` with no tier
    still emits both ``fast=true`` and ``reasoning=high``.
    ``grok-4.6-high`` is still rejected, so it never goes on the wire.
    """
    raw = (model_id or "").strip()
    alias = normalize_cursor_model_id(raw) or raw or "grok-4.6"
    if alias.startswith("grok-"):
        params = _GROK_VARIANT_PARAMS.get(_cursor_variant(model_id))
        if params is None:
            params = _GROK_VARIANT_PARAMS["high"]
        return {"id": alias, "params": [dict(p) for p in params]}
    return alias


class _CursorChatCompletions:
    def __init__(self, client: "CursorSDKClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _CursorChatNamespace:
    def __init__(self, client: "CursorSDKClient") -> None:
        self.completions = _CursorChatCompletions(client)


class CursorSDKClient:
    """OpenAI-client facade over a stateful cursor-sdk Agent."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        session_id: str | None = None,
        session_id_fn: Any = None,
        **_: Any,
    ) -> None:
        self.api_key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
        self.base_url = base_url or CURSOR_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        # Snapshot captured at construction — often None because agent_init
        # builds the client before assigning session_id. Prefer session_id_fn
        # (live agent.session_id) so /new rotation is visible at slot-key time.
        self._session_id = (session_id or "").strip() or None
        self._session_id_fn = session_id_fn if callable(session_id_fn) else None
        self.chat = _CursorChatNamespace(self)
        self.is_closed = False
        self._last_usage: Any = None

    def close(self) -> None:
        self.is_closed = True

    def _bound_session_id(self) -> str | None:
        fn = self._session_id_fn
        if fn is not None:
            try:
                live = fn()
            except Exception:
                live = None
            if isinstance(live, str):
                live = live.strip() or None
            else:
                live = None
            if live:
                return live
        return self._session_id

    def _slot_key(self, model: str) -> str:
        cwd = (
            os.environ.get("HERMES_CURSOR_SDK_CWD")
            or os.environ.get("CURSOR_SDK_CWD")
            or str(Path.cwd().resolve())
        )
        # Prefer the owning agent's live session id over a construction-time
        # snapshot: create_openai_client runs at agent_init *before* session_id
        # is assigned, and /new rotates the id without rebuilding the client.
        # HERMES_SESSION_ID is process-global and bleeds multiplexed gateway
        # sessions into one Cursor Agent conversation. HERMES_CURSOR_SESSION
        # stays the manual override and wins over everything.
        session = (
            os.environ.get("HERMES_CURSOR_SESSION")
            or self._bound_session_id()
            or os.environ.get("HERMES_SESSION_ID")
            or "default"
        )
        return f"{session}::{model}::{_cursor_sdk_slot_flavor(model)}::{cwd}"

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
        del timeout, tool_choice
        model_id = model or "grok-4.6"
        slot = self._slot_key(model_id)
        resume = slot in _slots and _slots[slot].get("agent") is not None
        prompt_text = format_hermes_cursor_prompt(
            messages or [],
            model=model_id,
            tools=tools,
            resume=resume,
        )
        images = extract_cursor_images(messages or [])
        log.debug(
            "cursor turn: chars=%d tools=%d resume=%s images=%d",
            len(prompt_text),
            len(hermes_tools_spec(tools)),
            resume,
            len(images),
        )
        response_text = self._run_turn(
            prompt_text, model=model_id, resume=resume, images=images
        )
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        usage = _openai_usage_from_cursor(self._last_usage)
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
            ],
            usage=usage,
            model=model_id,
        )
        if stream:
            from agent.copilot_acp_client import _completion_to_stream_chunks

            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(self, prompt_text: str, *, model: str) -> str:
        """Test hook / oneshot path: one Agent.prompt, no session cache."""
        return self._run_turn(prompt_text, model=model, resume=False, oneshot=True)

    def _run_turn(
        self,
        prompt_text: str,
        *,
        model: str,
        resume: bool,
        oneshot: bool = False,
        images: list[dict[str, Any]] | None = None,
    ) -> str:
        if not self.api_key:
            from hermes_constants import display_hermes_home

            raise RuntimeError(
                "CURSOR_API_KEY is not set. Create a key at "
                f"https://cursor.com/dashboard/api and put it in "
                f"{display_hermes_home()}/.env."
            )
        try:
            from cursor_sdk import Agent
        except ImportError:
            try:
                from tools.lazy_deps import ensure

                ensure("provider.cursor", prompt=False)
                from cursor_sdk import Agent
            except Exception as exc:
                raise RuntimeError(
                    "The Cursor provider requires the official cursor-sdk package. "
                    "Install it into the Hermes environment: uv pip install cursor-sdk"
                ) from exc

        cwd = (
            os.environ.get("HERMES_CURSOR_SDK_CWD")
            or os.environ.get("CURSOR_SDK_CWD")
            or ""
        ).strip()
        if not cwd:
            cwd = str(Path.cwd().resolve())

        log.debug("cursor model select: %s", json.dumps(cursor_sdk_model(model)))
        options = {
            "model": cursor_sdk_model(model),
            "api_key": self.api_key,
            "local": {"cwd": cwd},
            "mcp_servers": {},
            "tools": [],
        }

        self._last_usage = None
        if oneshot:
            try:
                result = Agent.prompt(_cursor_user_message(prompt_text, images), options=options)
            except Exception as exc:
                raise _cursor_startup_error(exc, phase="prompt") from exc
            self._last_usage = _cursor_token_usage(result)
            _raise_for_failed_run(result)
            text = getattr(result, "result", None)
            if isinstance(text, str):
                return text
            return "" if text is None else str(text)

        slot = self._slot_key(model)
        rec = _slot_record(slot)
        run = None
        with rec["lock"]:
            _cancel_run(rec)
            if rec.get("agent") is None:
                try:
                    rec["agent"] = Agent.create(options=options)
                except Exception as exc:
                    raise _cursor_startup_error(exc, phase="agent create") from exc
            try:
                run = rec["agent"].send(_cursor_user_message(prompt_text, images))
            except Exception as exc:
                if "already has active run" not in str(exc).lower():
                    _drop_agent(rec)
                    raise _cursor_startup_error(exc, phase="send") from exc
                _drop_agent(rec)
                try:
                    rec["agent"] = Agent.create(options=options)
                    run = rec["agent"].send(_cursor_user_message(prompt_text, images))
                except Exception as retry_exc:
                    _drop_agent(rec)
                    raise _cursor_startup_error(retry_exc, phase="send") from retry_exc
            rec["run"] = run
        try:
            waited = None
            wait = getattr(run, "wait", None)
            if callable(wait):
                waited = wait()
            self._last_usage = _cursor_token_usage(waited) or _cursor_token_usage(run)
            _raise_for_failed_run(waited, run)
            text = getattr(run, "text", None)
            if callable(text):
                text = text()
            if isinstance(text, str):
                return text
            result = getattr(run, "result", None)
            if isinstance(result, str):
                return result
            waited_text = getattr(waited, "result", None) if waited is not None else None
            if isinstance(waited_text, str):
                return waited_text
            return "" if text is None else str(text or "")
        except BaseException:
            # KeyboardInterrupt (Ctrl-C) is not an Exception: without this the
            # run pointer was cleared in `finally` without cancel(), leaving a
            # live Cursor run burning subscription tokens after the interrupt.
            with rec["lock"]:
                if rec.get("run") is run:
                    _drop_agent(rec)
            raise
        finally:
            with rec["lock"]:
                if rec.get("run") is run:
                    rec["run"] = None


def _cursor_user_message(
    prompt_text: str, images: list[dict[str, Any]] | None
) -> str | dict[str, Any]:
    if not images:
        return prompt_text
    resolved: list[dict[str, Any]] = []
    for image in images:
        path = image.get("path") if isinstance(image, dict) else None
        if isinstance(path, str) and Path(path).is_file():
            import mimetypes

            mime = mimetypes.guess_type(path)[0] or "image/png"
            raw = Path(path).read_bytes()
            import base64

            resolved.append(
                {"data": base64.b64encode(raw).decode("ascii"), "mime_type": mime}
            )
        elif isinstance(image, dict):
            resolved.append(image)
    if not resolved:
        return prompt_text
    return {"text": prompt_text, "images": resolved}


def _blocked_reads(messages: list[dict[str, Any]]) -> str:
    """Call out Hermes BLOCKED read_file results so the model stops retrying."""
    hits: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() != "tool":
            continue
        text = _message_text(message)
        if "BLOCKED: You have called read_file" not in text:
            continue
        snippet = text.replace("\n", " ").strip()[:240]
        if snippet not in hits:
            hits.append(snippet)
    return "\n".join(hits[-6:])
