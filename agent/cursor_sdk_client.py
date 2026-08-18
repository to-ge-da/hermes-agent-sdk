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
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import _extract_tool_calls_from_text

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


def _drop_agent(rec: dict[str, Any]) -> None:
    _cancel_run(rec)
    rec["agent"] = None


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


def _log_cursor_images(images: list[dict[str, Any]] | None) -> None:
    """Write path/sha of attached images. No file bytes."""
    import hashlib

    rec: list[dict[str, Any]] = []
    for image in images or []:
        if not isinstance(image, dict):
            continue
        path = image.get("path")
        if isinstance(path, str) and Path(path).is_file():
            raw = Path(path).read_bytes()
            rec.append(
                {
                    "path": path,
                    "sha256_16": hashlib.sha256(raw).hexdigest()[:16],
                    "bytes": len(raw),
                }
            )
            continue
        if image.get("data"):
            rec.append(
                {
                    "kind": "data",
                    "mime": image.get("mime_type"),
                    "b64_chars": len(str(image.get("data") or "")),
                }
            )
        elif image.get("url"):
            rec.append({"kind": "url", "url": str(image["url"])[:120]})
    Path("/tmp/cursor_images_sent.json").write_text(
        json.dumps({"n": len(rec), "images": rec}, indent=2)
    )


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


def cursor_sdk_model(model_id: str) -> str | dict[str, Any]:
    """SDK inference id plus high / not-fast params.

    Live 2026-08-17: Agent.prompt accepts ``grok-4.6``, not catalog
    ``cursor-grok-4.6-high-fast``. Bare ``grok-4.6`` is the high-fast SKU
    (no reasoning_tokens). ``params: [{id: fast, value: false}]`` is the
    only setting that produced reasoning tokens. ``grok-4.6-high`` is
    rejected. This session stays on grok-4.6 high without fast.
    """
    alias = normalize_cursor_model_id(model_id) or (model_id or "").strip() or "grok-4.6"
    if alias.startswith("grok-"):
        return {
            "id": alias,
            "params": [
                {"id": "fast", "value": "false"},
                {"id": "reasoning", "value": "high"},
            ],
        }
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
        **_: Any,
    ) -> None:
        self.api_key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
        self.base_url = base_url or CURSOR_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self.chat = _CursorChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _slot_key(self, model: str) -> str:
        cwd = (
            os.environ.get("HERMES_CURSOR_SDK_CWD")
            or os.environ.get("CURSOR_SDK_CWD")
            or str(Path.cwd().resolve())
        )
        session = (
            os.environ.get("HERMES_SESSION_ID")
            or os.environ.get("HERMES_CURSOR_SESSION")
            or "default"
        )
        return f"{session}::{model}::high-nofast::{cwd}"

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
        try:
            Path("/tmp/cursor_prompt_len.txt").write_text(
                f"chars={len(prompt_text)} tools={len(hermes_tools_spec(tools))} "
                f"resume={resume} images={len(images)}\n"
            )
            _log_cursor_images(images)
        except Exception:
            pass
        response_text = self._run_turn(
            prompt_text, model=model_id, resume=resume, images=images
        )
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

        cwd = (
            os.environ.get("HERMES_CURSOR_SDK_CWD")
            or os.environ.get("CURSOR_SDK_CWD")
            or ""
        ).strip()
        if not cwd:
            cwd = str(Path.cwd().resolve())

        try:
            Path("/tmp/cursor_model_sel.txt").write_text(
                json.dumps(cursor_sdk_model(model), sort_keys=True)
            )
        except Exception:
            pass
        options = {
            "model": cursor_sdk_model(model),
            "api_key": self.api_key,
            "local": {"cwd": cwd},
            "mcp_servers": {},
            "tools": [],
        }

        if oneshot:
            result = Agent.prompt(_cursor_user_message(prompt_text, images), options=options)
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
                rec["agent"] = Agent.create(options=options)
            try:
                run = rec["agent"].send(_cursor_user_message(prompt_text, images))
            except Exception as exc:
                if "already has active run" not in str(exc).lower():
                    rec["agent"] = None
                    raise
                _drop_agent(rec)
                rec["agent"] = Agent.create(options=options)
                run = rec["agent"].send(_cursor_user_message(prompt_text, images))
            rec["run"] = run
        try:
            wait = getattr(run, "wait", None)
            if callable(wait):
                wait()
            text = getattr(run, "text", None)
            if callable(text):
                text = text()
            if isinstance(text, str):
                return text
            result = getattr(run, "result", None)
            if isinstance(result, str):
                return result
            return "" if text is None else str(text or "")
        except Exception:
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
