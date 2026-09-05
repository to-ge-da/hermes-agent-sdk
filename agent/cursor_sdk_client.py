"""OpenAI-compatible shim over the official Cursor Agent SDK.

Cursor has no public POST /v1/chat/completions. Dashboard crsr_ keys talk to
cursor-sdk. Hermes stays the harness: every Hermes tool is described with its
real JSON schema, the model emits <tool_call> JSON, and Hermes executes it —
same shape as Copilot ACP, without ACP wording (that makes Cursor walk the
repo with its own tools).

Cursor-native tools stay off (tools=["mcp"], mcp_servers={}). That is not a
Hermes allowlist; it is "do not give Cursor a second filesystem."
`tools=[]` is documented as "no built-in tools; the model can only respond
with text", and deny-wins requires a tool to be in `tools` when that field
is set. Custom tools ride the built-in `custom-user-tools` MCP server, so
omitting the `mcp` group hides Hermes `local.custom_tools` and Part 1
captures never fire. `tools=["mcp"]` + empty `mcp_servers` exposes ONLY
those Hermes names. Assumption per cursor.com/docs/sdk/python — a live
Cursor-hosted session test is still required before merge.

A live Agent is reused across turns in this process. Cold start sends a
windowed transcript; later turns send only the latest user line plus any
trailing tool results. Do not flatten a 300k resume into Agent.prompt.

The live Agent keeps its own server-side copy of the conversation, so the
slot also remembers a transcript anchor: when Hermes rewrites history
(context compression, /clear, a checkpoint rewind) the Agent is dropped and
the next turn cold-starts from the rewritten transcript. Without that, a
compressed Hermes session keeps paying Cursor for the pre-compression
history it can no longer see.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import (
    _CUSTOM_TOOL_DEFERRAL,
    _bridge_fail_once_error,
    _captures_to_tool_calls,
    _extract_tool_calls_from_text,
    _hermes_tool_names,
    _looks_like_untranslated_bridge_markup,
    _select_hermes_tool_calls,
    _unknown_bridge_tool_names,
)
from hermes_constants import display_hermes_home

log = logging.getLogger(__name__)

CURSOR_MARKER_BASE_URL = "cursor-sdk://local"

# Size cap only — not a task-specific stub. Cursor Agent.prompt/create hangs
# if the first send is a 300k Hermes resume.
_COLD_TRANSCRIPT_CHARS = 48_000
_TOOL_DESC_CHARS = 800
_SYSTEM_CHARS = 12_000

# Each live Agent owns a cursor-sdk-bridge subprocess, so abandoned slots
# (/new, /resume, a finished gateway chat) cannot accumulate. Only agents
# that are BOTH over the cap and idle are retired — a genuinely busy
# multi-chat gateway keeps its agents rather than paying repeated cold starts.
_MAX_LIVE_SLOTS = 3
_SLOT_IDLE_SECONDS = 600.0

# One Cursor Agent per Hermes session. A slot also holds the in-flight Run
# so a Hermes retry/timeout cannot send() while that run is still open.
_slots: dict[str, dict[str, Any]] = {}
_slots_guard = threading.Lock()


class CursorSDKError(RuntimeError):
    """Cursor-side failure carrying an HTTP status for Hermes classification.

    ``agent/error_classifier.py`` reads ``status_code`` off the raised
    exception, so a rejected key becomes an auth failure and an exhausted
    plan becomes a rate limit instead of an unknown error the loop retries
    blindly.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ColdStartRequired(Exception):
    """Internal: the live Agent is gone, so the delta prompt is orphaned."""


_AUTH_MARKERS = (
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "invalid token",
    "authentication",
    "forbidden",
    "401",
    "403",
)
_QUOTA_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota",
    "usage limit",
    "insufficient credit",
    "429",
)
_MODEL_MARKERS = (
    "model not found",
    "unknown model",
    "invalid model",
    "unsupported model",
    "not available",
)


def _short(text: str, limit: int = 240) -> str:
    flat = " ".join(str(text or "").split())
    return flat[:limit]


def cursor_sdk_error(exc: Exception, *, phase: str) -> CursorSDKError:
    """Translate a cursor-sdk exception into an actionable Hermes error.

    The SDK surfaces failures as bare exceptions whose text is often a raw
    JSON body; printing that in the CLI tells the user nothing about what to
    do next.
    """
    detail = _short(str(exc))
    low = detail.lower()
    if any(marker in low for marker in _AUTH_MARKERS):
        return CursorSDKError(
            "Cursor rejected CURSOR_API_KEY. Create a fresh dashboard key at "
            "https://cursor.com/dashboard/api and update it in "
            f"{display_hermes_home()}/.env, then restart Hermes. "
            f"(cursor-sdk {phase}: {detail})",
            status_code=401,
        )
    if any(marker in low for marker in _QUOTA_MARKERS):
        return CursorSDKError(
            "Cursor usage limit reached on this plan. Wait for the quota to "
            "reset or switch to another model/provider with /model. "
            f"(cursor-sdk {phase}: {detail})",
            status_code=429,
        )
    if any(marker in low for marker in _MODEL_MARKERS):
        return CursorSDKError(
            "Cursor does not serve this model id. Pick one from `hermes model` "
            f"(the catalog is fetched live). (cursor-sdk {phase}: {detail})",
            status_code=404,
        )
    return CursorSDKError(
        f"Cursor SDK {phase} failed: {detail}", status_code=502
    )


def _slot_record(slot: str) -> dict[str, Any]:
    with _slots_guard:
        rec = _slots.get(slot)
        if rec is None:
            rec = {
                "agent": None,
                "run": None,
                "lock": threading.Lock(),
                "anchor": None,
                "images": set(),
                "used": time.monotonic(),
            }
            _slots[slot] = rec
        rec["used"] = time.monotonic()
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
    """Best-effort teardown of the Agent's bridge subprocess."""
    if agent is None:
        return
    for name in ("close", "dispose", "shutdown", "stop"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return


def _drop_agent(rec: dict[str, Any]) -> None:
    _cancel_run(rec)
    agent = rec.get("agent")
    rec["agent"] = None
    rec["images"] = set()
    if agent is not None:
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
        return RuntimeError(
            "Cursor rejected the API key (authentication failed). Check "
            f"CURSOR_API_KEY in {display_hermes_home()}/.env or regenerate it at "
            "https://cursor.com/dashboard/api. "
            f"cursor-sdk said: {text}"
        )
    return RuntimeError(f"Cursor Agent SDK {phase} failed: {text}")


def _evict_idle_slots(keep: str) -> None:
    """Retire live agents that are over the cap and idle.

    Slots busy in a run on another thread are skipped (non-blocking lock).
    """
    now = time.monotonic()
    with _slots_guard:
        live = [
            (float(rec.get("used") or 0.0), key)
            for key, rec in _slots.items()
            if key != keep and rec.get("agent") is not None
        ]
        if len(live) < _MAX_LIVE_SLOTS:
            return
        live.sort()
        doomed = [
            key
            for used, key in live[: len(live) - _MAX_LIVE_SLOTS + 1]
            if now - used > _SLOT_IDLE_SECONDS
        ]
        victims = [(key, _slots[key]) for key in doomed if key in _slots]
    for key, rec in victims:
        if not rec["lock"].acquire(blocking=False):
            continue
        try:
            _drop_agent(rec)
        finally:
            rec["lock"].release()
        with _slots_guard:
            if _slots.get(key) is rec and rec.get("agent") is None:
                _slots.pop(key, None)


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


def _message_digest(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "")
    text = _message_text(message)
    seed = f"{role}|{len(text)}|{text[:512]}"
    return hashlib.blake2b(
        seed.encode("utf-8", "replace"), digest_size=8
    ).hexdigest()


def transcript_anchor(messages: list[dict[str, Any]] | None) -> tuple[int, str]:
    """``(message count, digest of the first non-system message)``.

    Both halves move when Hermes rewrites the transcript: compression drops
    middle turns and prepends a summary, ``/clear`` empties it, a checkpoint
    rewind shortens it. A growing conversation never shrinks and never
    changes its head, so this is a false-positive-cheap rewrite signal.
    """
    typed = [m for m in messages or [] if isinstance(m, dict)]
    head = ""
    for message in typed:
        if str(message.get("role") or "").lower() != "system":
            head = _message_digest(message)
            break
    return len(typed), head


def history_was_rewritten(
    previous: tuple[int, str] | None,
    current: tuple[int, str],
) -> bool:
    """True when *current* is not a continuation of the anchored transcript."""
    if not previous:
        return False
    prev_count, prev_head = previous
    count, head = current
    if not prev_count:
        return False
    if count < prev_count:
        return True
    return bool(prev_head) and head != prev_head


def turn_delta_messages(
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """The slice a resume prompt actually carries: last user turn onward."""
    typed = [m for m in messages or [] if isinstance(m, dict)]
    if not typed:
        return []
    for i in range(len(typed) - 1, -1, -1):
        if str(typed[i].get("role") or "").lower() == "user":
            return typed[i:]
    return typed[-1:]


def cursor_image_key(image: dict[str, Any] | None) -> str:
    """Identity of an attachment, so the same bytes are uploaded once.

    Paths include size + mtime: a screenshot re-saved to the same path is a
    different image and must be re-sent.
    """
    if not isinstance(image, dict):
        return ""
    path = image.get("path")
    if isinstance(path, str) and path:
        try:
            stat = Path(path).stat()
            return f"path:{path}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            return f"path:{path}"
    url = image.get("url")
    if isinstance(url, str) and url:
        return f"url:{url}"
    data = str(image.get("data") or "")
    if not data:
        return ""
    digest = hashlib.blake2b(data.encode("utf-8", "replace"), digest_size=8)
    return f"data:{digest.hexdigest()}"


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


def describe_cursor_images(images: list[dict[str, Any]] | None) -> str:
    """Compact descriptor of the attachments for ``agent.log``. No bytes read."""
    parts: list[str] = []
    for image in images or []:
        if not isinstance(image, dict):
            continue
        path = image.get("path")
        if isinstance(path, str) and path:
            try:
                parts.append(f"{path} ({Path(path).stat().st_size}B)")
            except OSError:
                parts.append(path)
        elif image.get("data"):
            parts.append(
                f"inline {image.get('mime_type') or 'image/png'} "
                f"({len(str(image.get('data') or ''))} b64 chars)"
            )
        elif image.get("url"):
            parts.append(str(image["url"])[:120])
    return ", ".join(parts) or "none"


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


def _hermes_custom_tools(
    tools: list[dict[str, Any]] | None,
    bucket: list[Any],
) -> dict[str, dict[str, Any]]:
    """Allowlist Hermes names as Cursor local.custom_tools; do not execute."""
    custom: dict[str, dict[str, Any]] = {}
    for spec in hermes_tools_spec(tools):
        name = spec["name"]

        def _execute(args: Any, context: Any = None, *, _name: str = name) -> str:
            call_id = None
            if context is not None:
                call_id = getattr(context, "tool_call_id", None)
            bucket.append((_name, args, call_id))
            return _CUSTOM_TOOL_DEFERRAL

        custom[name] = {
            "description": spec.get("description") or "",
            "input_schema": spec.get("parameters")
            or {"type": "object", "properties": {}},
            "execute": _execute,
        }
    return custom


_MAX_SDK_TOOL_WALK = 32


def _iter_sdk_tool_calls(
    source: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> list[tuple[str, Any, str | None]]:
    """Walk a materialized conversation snapshot for tool_call events.

    Does not drain ``run.messages()`` (a live generator). Only sequences
    already in memory are walked, with a depth/identity bound so cycles
    cannot rely on the caller's ``except`` to terminate.
    """
    out: list[tuple[str, Any, str | None]] = []
    if source is None or isinstance(source, (str, bytes, bytearray)):
        return out
    if _depth > _MAX_SDK_TOOL_WALK:
        return out
    seen = _seen if _seen is not None else set()
    ident = id(source)
    if ident in seen:
        return out
    seen.add(ident)
    if isinstance(source, dict):
        items: list[Any] = [source]
    elif isinstance(source, (list, tuple)):
        items = list(source)
    else:
        messages = getattr(source, "messages", None)
        if callable(messages):
            messages = None
        if isinstance(messages, (list, tuple)) and messages is not source:
            return _iter_sdk_tool_calls(
                messages, _depth=_depth + 1, _seen=seen
            )
        items = [source]
    for msg in items:
        if msg is None or msg is source:
            continue
        if isinstance(msg, dict):
            mtype = msg.get("type")
            name = msg.get("name")
            args = msg.get("args", msg.get("arguments"))
            call_id = msg.get("call_id") or msg.get("tool_call_id") or msg.get("id")
            nested = msg.get("message")
        else:
            mtype = getattr(msg, "type", None)
            name = getattr(msg, "name", None)
            args = getattr(msg, "args", None)
            if args is None:
                args = getattr(msg, "arguments", None)
            call_id = getattr(msg, "call_id", None) or getattr(
                msg, "tool_call_id", None
            )
            nested = getattr(msg, "message", None)
        if str(mtype or "").lower() in {"tool_call", "tooluse", "tool_use"}:
            if isinstance(name, str) and name.strip():
                out.append(
                    (
                        name.strip(),
                        args,
                        call_id if isinstance(call_id, str) else None,
                    )
                )
            continue
        if nested is not None and nested is not msg:
            out.extend(
                _iter_sdk_tool_calls(nested, _depth=_depth + 1, _seen=seen)
            )
    return out


def _tool_event_dedup_key(item: tuple[str, Any, str | None]) -> tuple[Any, ...]:
    return (item[0], json.dumps(item[1], sort_keys=True, default=str), item[2])


def _dedup_tool_events(
    items: list[tuple[str, Any, str | None]],
) -> list[tuple[str, Any, str | None]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[tuple[str, Any, str | None]] = []
    for item in items:
        key = _tool_event_dedup_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _collect_run_tool_events(*sources: Any) -> list[tuple[str, Any, str | None]]:
    """Best-effort tool_call events after wait().

    Walks ``conversation()`` snapshots and ``tool_calls`` / ``tool_events``.
    Does not call ``run.messages()`` — that generator is a live drain.
    """
    out: list[tuple[str, Any, str | None]] = []

    def _add(items: list[tuple[str, Any, str | None]]) -> None:
        out.extend(items)

    for source in sources:
        if source is None:
            continue
        conv = getattr(source, "conversation", None)
        if callable(conv):
            try:
                _add(_iter_sdk_tool_calls(conv()))
            except Exception:
                pass
        for attr in ("tool_calls", "tool_events"):
            val = getattr(source, attr, None)
            if val:
                _add(_iter_sdk_tool_calls(val))
    return _dedup_tool_events(out)


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


# Nous #88212 pin: suffix-less grok-* is high / not-fast (reasoning tokens).
_GROK_DEFAULT_REASONING = "high"

_CURSOR_REASONING_TIERS = ("low", "medium", "high")


def split_cursor_model_id(model_id: str) -> tuple[str, str, bool]:
    """``(inference id, reasoning tier, fast)`` parsed from a catalog slug.

    Catalog ids carry the SKU in the suffix (``cursor-grok-4.6-high-fast``,
    ``…-low-fast``, ``…-high``). Both suffix parts have to come off in one
    pass: stripping only the first match left ids like ``grok-4.6-medium``
    or ``grok-4.6-low``, which the SDK rejects outright.
    """
    raw = (model_id or "").strip()
    if not raw:
        return "", "", False
    bare = raw[7:] if raw.lower().startswith("cursor-") else raw
    parts = bare.split("-")
    fast = False
    tier = ""
    if len(parts) > 1 and parts[-1].lower() == "fast":
        fast = True
        parts = parts[:-1]
    if len(parts) > 1 and parts[-1].lower() in _CURSOR_REASONING_TIERS:
        tier = parts[-1].lower()
        parts = parts[:-1]
    return ("-".join(parts) or bare), tier, fast


def normalize_cursor_model_id(model_id: str) -> str:
    """Return the Hermes-facing slug for a Cursor catalog id."""
    bare, _tier, _fast = split_cursor_model_id(model_id)
    return bare or (model_id or "").strip()


def _cursor_variant(model_id: str) -> str:
    """SKU suffix of a catalog id ("" when bare/unknown)."""
    _bare, tier, fast_flag = split_cursor_model_id(model_id)
    return "-".join(p for p in (tier, "fast" if fast_flag else "") if p)


def _cursor_sdk_slot_flavor(model_id: str) -> str:
    """Agent-slot tag so high vs high-fast do not reuse each other's Agent."""
    sel = cursor_sdk_model(model_id)
    if not isinstance(sel, dict):
        return "plain"
    params = {p["id"]: str(p["value"]).lower() for p in sel.get("params") or []}
    reasoning = params.get("reasoning") or _GROK_DEFAULT_REASONING
    speed = "fast" if params.get("fast") == "true" else "nofast"
    return f"{reasoning}-{speed}"


CURSOR_PROVIDER_NAMES = frozenset({"cursor", "cursor-sdk", "cursor-composer"})


def is_grok_cursor_model(model_id: str) -> bool:
    """True when *model_id* is a Grok SKU after catalog-affix stripping."""
    bare = normalize_cursor_model_id(model_id).lower()
    return bare == "grok" or bare.startswith("grok-")


def cursor_aux_http_twin(model_id: str) -> tuple[str, str] | None:
    """HTTP ``(provider, model)`` for auxiliary tasks, or ``None``.

    Cursor has no chat-completions endpoint. Grok SKUs are served by xAI
    under the same id, so compression / titles / vision can reuse that
    HTTP twin. Claude, Composer, GPT and Gemini SKUs have no twin — the
    caller must fall through to normal auto-detection rather than asking
    xAI for ``claude-opus-5`` (which 404s every auxiliary call).
    """
    if not is_grok_cursor_model(model_id or "grok-4.6"):
        return None
    return "xai-oauth", normalize_cursor_model_id(model_id or "grok-4.6")


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
    not-fast). Explicit variants map to matching params; ``-fast`` with no
    tier still emits both ``fast=true`` and ``reasoning=high``. Unknown
    tiers such as ``-medium-fast`` honor both parsed suffix parts.
    """
    bare, tier, fast_flag = split_cursor_model_id(model_id)
    alias = bare or "grok-4.6"
    if alias.startswith("grok-"):
        key = "-".join(p for p in (tier, "fast" if fast_flag else "") if p)
        params = _GROK_VARIANT_PARAMS.get(key)
        if params is None:
            params = [
                {"id": "fast", "value": "true" if fast_flag else "false"},
                {"id": "reasoning", "value": tier or "high"},
            ]
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
        self._last_captures: list[Any] = []

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
        # HERMES_CURSOR_SESSION is the manual override. Then the live
        # session_id_fn (agent.session_id after init / after /new). Then the
        # gateway ContextVar via get_session_env (os.environ is the default
        # profile in a multiplex process). Resolved SDK selection is the
        # slot tag so identical params share an Agent and high-fast does not
        # reuse a high-nofast Agent.
        gateway_session = ""
        try:
            from gateway.session_context import get_session_env

            gateway_session = get_session_env("HERMES_SESSION_ID", "")
        except Exception:
            gateway_session = os.environ.get("HERMES_SESSION_ID", "")
        session = (
            os.environ.get("HERMES_CURSOR_SESSION")
            or self._bound_session_id()
            or gateway_session
            or os.environ.get("HERMES_SESSION_ID")
            or "default"
        )
        selection = json.dumps(cursor_sdk_model(model), sort_keys=True)
        return f"{session}::{selection}::{cwd}"

    def _prepare_slot(
        self, rec: dict[str, Any], anchor: tuple[int, str]
    ) -> bool:
        """Decide resume vs cold start, dropping a diverged Cursor Agent.

        Returns True when the live Agent may continue with a delta prompt.
        """
        with rec["lock"]:
            previous = rec.get("anchor")
            if rec.get("agent") is not None and history_was_rewritten(
                previous, anchor
            ):
                log.info(
                    "Cursor: Hermes rewrote the transcript (%s → %s messages) — "
                    "starting a fresh Cursor agent so it stops billing for "
                    "history Hermes no longer sends",
                    previous[0] if previous else 0,
                    anchor[0],
                )
                _drop_agent(rec)
            rec["anchor"] = anchor
            return rec.get("agent") is not None

    def _turn_images(
        self,
        rec: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        resume: bool,
    ) -> list[dict[str, Any]]:
        """Attachments this turn owes Cursor, each uploaded at most once.

        ``extract_cursor_images`` walks the whole transcript, so without a
        scope and a sent-set every turn after a vision call re-uploaded the
        same screenshots for the rest of the session.
        """
        scope = turn_delta_messages(messages) if resume else messages
        candidates = extract_cursor_images(scope)
        fresh: list[dict[str, Any]] = []
        with rec["lock"]:
            sent = rec.get("images")
            if not isinstance(sent, set):
                sent = set()
                rec["images"] = sent
            for image in candidates:
                key = cursor_image_key(image)
                if not key or key in sent:
                    continue
                sent.add(key)
                fresh.append(image)
        return fresh

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
        rec = _slot_record(slot)
        resume = self._prepare_slot(rec, transcript_anchor(messages))
        prompt_text = format_hermes_cursor_prompt(
            messages or [],
            model=model_id,
            tools=tools,
            resume=resume,
        )
        images = self._turn_images(rec, messages or [], resume=resume)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Cursor turn: chars=%d tools=%d resume=%s images=[%s]",
                len(prompt_text),
                len(hermes_tools_spec(tools)),
                resume,
                describe_cursor_images(images),
            )
        try:
            response_text = self._run_turn(
                prompt_text,
                model=model_id,
                resume=resume,
                images=images,
                slot=slot,
                tools=tools,
            )
        except _ColdStartRequired:
            # The live agent could not take the delta, so the replacement
            # agent knows nothing about this conversation — reseed it with the
            # windowed transcript instead of handing it an orphaned delta.
            log.info(
                "Cursor: live agent was unusable — reseeding a fresh agent "
                "with the windowed transcript"
            )
            prompt_text = format_hermes_cursor_prompt(
                messages or [], model=model_id, tools=tools, resume=False
            )
            images = self._turn_images(rec, messages or [], resume=False)
            response_text = self._run_turn(
                prompt_text,
                model=model_id,
                resume=False,
                images=images,
                slot=slot,
                tools=tools,
            )
        tool_calls, cleaned_text = self._tool_calls_after_turn(
            response_text, tools=tools, model_id=model_id
        )
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

    def _abandon_slot(self, model: str) -> None:
        """Drop the live Agent so the next turn does not resume poisoned markup."""
        slot = self._slot_key(model)
        with _slots_guard:
            rec = _slots.get(slot)
        if rec is None:
            return
        with rec["lock"]:
            _drop_agent(rec)

    def _tool_calls_after_turn(
        self,
        response_text: str,
        *,
        tools: list[dict[str, Any]] | None,
        model_id: str,
    ) -> tuple[list[Any], str]:
        hermes_names = _hermes_tool_names(tools)
        captured = _captures_to_tool_calls(self._last_captures)
        if captured:
            fail = _bridge_fail_once_error(response_text, captured, hermes_names)
            if fail and _unknown_bridge_tool_names(captured, hermes_names):
                self._abandon_slot(model_id)
                return [], fail
            matched = _select_hermes_tool_calls(captured, hermes_names)
            if matched:
                return matched, ""
        extracted, cleaned = _extract_tool_calls_from_text(response_text)
        fail = _bridge_fail_once_error(response_text, extracted, hermes_names)
        if fail:
            self._abandon_slot(model_id)
            return [], fail
        matched = _select_hermes_tool_calls(extracted, hermes_names)
        if matched:
            return matched, cleaned
        return [], cleaned

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
        slot: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        if not self.api_key:
            raise CursorSDKError(
                "CURSOR_API_KEY is not set. Create a key at "
                f"https://cursor.com/dashboard/api and put it in "
                f"{display_hermes_home()}/.env.",
                status_code=401,
            )
        try:
            from cursor_sdk import Agent
        except ImportError:
            try:
                from tools.lazy_deps import ensure

                ensure("provider.cursor", prompt=False)
                from cursor_sdk import Agent
            except Exception as exc:
                raise CursorSDKError(
                    "The Cursor provider requires the official cursor-sdk package. "
                    "Install it into the Hermes environment: uv pip install cursor-sdk",
                    status_code=501,
                ) from exc

        cwd = (
            os.environ.get("HERMES_CURSOR_SDK_CWD")
            or os.environ.get("CURSOR_SDK_CWD")
            or ""
        ).strip()
        if not cwd:
            cwd = str(Path.cwd().resolve())

        selection = cursor_sdk_model(model)
        log.debug("Cursor model selection: %s", json.dumps(selection, sort_keys=True))

        self._last_usage = None
        self._last_captures = []

        def _options_for(bucket: list[Any]) -> dict[str, Any]:
            # Documented cursor-sdk semantics (cursor.com/docs/sdk/python):
            # `tools=[]` offers no built-in tools and the model can only
            # respond with text. Deny wins: when `tools` is set, a tool must
            # be listed. Custom tools ride the built-in `custom-user-tools`
            # MCP server and are gated by the `mcp` capability group —
            # omitting `mcp` hides Hermes `local.custom_tools`, so Part 1
            # captures never fire. `tools=["mcp"]` + `mcp_servers={}`
            # exposes ONLY those Hermes custom tools; Cursor shell/read/edit
            # stay off.
            # Assumption not live-verified in this change — a Cursor-hosted
            # session test is still required before merge.
            return {
                "model": selection,
                "api_key": self.api_key,
                "local": {
                    "cwd": cwd,
                    "custom_tools": _hermes_custom_tools(tools, bucket),
                },
                "mcp_servers": {},
                "tools": ["mcp"],
            }

        def _finish_turn(text: str, *sources: Any, bucket: list[Any]) -> str:
            events = _collect_run_tool_events(*sources) if sources else []
            self._last_captures = _dedup_tool_events(list(bucket) + events)
            if sources:
                _raise_for_failed_run(*sources)
            if self._last_captures:
                return text if text.strip() else _CUSTOM_TOOL_DEFERRAL
            detail = _run_failure_detail(*sources) if sources else ""
            combined = "\n".join(
                part for part in (text, detail) if part and str(part).strip()
            )
            if _looks_like_untranslated_bridge_markup(combined):
                return combined
            return text

        if oneshot:
            capture_bucket: list[Any] = []
            options = _options_for(capture_bucket)
            try:
                result = Agent.prompt(
                    _cursor_user_message(prompt_text, images), options=options
                )
            except Exception as exc:
                raise cursor_sdk_error(exc, phase="prompt") from exc
            self._last_usage = _cursor_token_usage(result)
            text = getattr(result, "result", None)
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            return _finish_turn(text, result, bucket=capture_bucket)

        slot = slot or self._slot_key(model)
        rec = _slot_record(slot)
        _evict_idle_slots(slot)
        run = None
        with rec["lock"]:
            bucket = rec.get("captures")
            if not isinstance(bucket, list):
                bucket = []
                rec["captures"] = bucket
            bucket.clear()
            options = _options_for(bucket)
            _cancel_run(rec)
            try:
                if rec.get("agent") is None:
                    rec["agent"] = Agent.create(options=options)
                try:
                    run = rec["agent"].send(_cursor_user_message(prompt_text, images))
                except Exception as exc:
                    if "already has active run" not in str(exc).lower():
                        raise
                    _drop_agent(rec)
                    if resume:
                        # A replacement agent has none of this conversation;
                        # the caller must resend the windowed transcript.
                        raise _ColdStartRequired() from exc
                    rec["agent"] = Agent.create(options=options)
                    run = rec["agent"].send(_cursor_user_message(prompt_text, images))
            except _ColdStartRequired:
                raise
            except Exception as exc:
                _drop_agent(rec)
                raise cursor_sdk_error(exc, phase="send") from exc
            rec["run"] = run
        try:
            waited = None
            wait = getattr(run, "wait", None)
            if callable(wait):
                try:
                    waited = wait()
                except Exception as exc:
                    raise cursor_sdk_error(exc, phase="run") from exc
            self._last_usage = _cursor_token_usage(waited) or _cursor_token_usage(run)
            text = _run_text(run, waited)
            text = _finish_turn(text, waited, run, bucket=bucket)
            if self._last_captures:
                return text
            if _looks_like_untranslated_bridge_markup(text):
                return text
            if not text.strip():
                # A silent turn reads as a hung Hermes. Drop the agent so the
                # retry cold-starts with the full window instead of a delta
                # the live agent may never have received.
                with rec["lock"]:
                    _drop_agent(rec)
                raise CursorSDKError(
                    "Cursor returned an empty response (the run was cancelled "
                    "or produced no output). Retrying on a fresh Cursor agent.",
                    status_code=502,
                )
            return text
        except Exception:
            with rec["lock"]:
                if rec.get("run") is run:
                    _drop_agent(rec)
            raise
        except BaseException:
            # Ctrl-C / Esc during a run. Cancel the run but KEEP the agent:
            # dropping it here made the next message land on a brand-new
            # Cursor agent that only ever saw the delta prompt, so the model
            # answered with no memory of the session.
            with rec["lock"]:
                if rec.get("run") is run:
                    _cancel_run(rec)
            raise
        finally:
            with rec["lock"]:
                if rec.get("run") is run:
                    rec["run"] = None


def _run_text(run: Any, waited: Any) -> str:
    """Assistant text off a cursor-sdk Run / RunResult, whichever carries it."""
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
