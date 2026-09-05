"""Cursor native provider — catalog parse, client shim, no live network."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.cursor_sdk_client import (
    CursorSDKClient,
    CursorSDKError,
    _cursor_sdk_slot_flavor,
    _openai_usage_from_cursor,
    cursor_aux_http_twin,
    cursor_image_key,
    cursor_sdk_error,
    cursor_sdk_model,
    extract_cursor_images,
    expand_cursor_model_ids,
    format_hermes_cursor_prompt,
    hermes_tools_spec,
    history_was_rewritten,
    is_grok_cursor_model,
    normalize_cursor_model_id,
    split_cursor_model_id,
    transcript_anchor,
    turn_delta_messages,
    window_cursor_messages,
)


def test_normalize_cursor_model_id_strips_catalog_affixes():
    assert normalize_cursor_model_id("cursor-grok-4.6-high-fast") == "grok-4.6"
    assert normalize_cursor_model_id("cursor-grok-4.5-high") == "grok-4.5"
    assert normalize_cursor_model_id("grok-4.6") == "grok-4.6"
    assert normalize_cursor_model_id("composer-2.5") == "composer-2.5"


def _params(sel):
    return {p["id"]: p["value"] for p in sel["params"]}


def test_cursor_sdk_model_uses_high_without_fast_for_bare_grok():
    """Suffix-less grok-* keeps the Nous #88212 pin (reasoning tokens)."""
    sel = cursor_sdk_model("grok-4.6")
    assert sel["id"] == "grok-4.6"
    params = _params(sel)
    assert params["fast"] == "false"
    assert params["reasoning"] == "high"
    assert cursor_sdk_model("composer-2.5") == "composer-2.5"


def test_cursor_sdk_model_honors_catalog_high_fast_suffix():
    sel = cursor_sdk_model("cursor-grok-4.6-high-fast")
    assert sel["id"] == "grok-4.6"
    params = _params(sel)
    assert params["fast"] == "true"
    assert params["reasoning"] == "high"


def test_cursor_sdk_model_honors_catalog_high_without_fast():
    sel = cursor_sdk_model("cursor-grok-4.5-high")
    assert sel["id"] == "grok-4.5"
    params = _params(sel)
    assert params["fast"] == "false"
    assert params["reasoning"] == "high"


def test_cursor_sdk_slot_flavor_separates_fast_from_nofast():
    assert _cursor_sdk_slot_flavor("grok-4.6") == "high-nofast"
    assert _cursor_sdk_slot_flavor("cursor-grok-4.6-high-fast") == "high-fast"
    assert _cursor_sdk_slot_flavor("cursor-grok-4.5-high") == "high-nofast"
    assert _cursor_sdk_slot_flavor("composer-2.5") == "plain"
    client = CursorSDKClient(api_key="crsr_test")
    assert client._slot_key("grok-4.6") != client._slot_key("cursor-grok-4.6-high-fast")


def test_expand_cursor_model_ids_adds_short_aliases():
    assert expand_cursor_model_ids(["cursor-grok-4.6-high-fast"]) == [
        "cursor-grok-4.6-high-fast",
        "grok-4.6",
    ]


def test_fetch_models_parses_items_shape():
    from providers import get_provider_profile

    payload = json.dumps(
        {"items": [{"id": "cursor-grok-4.6-high-fast"}, {"id": "composer-2.5"}]}
    ).encode()

    class _Resp:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    profile = get_provider_profile("cursor")
    assert profile is not None
    with patch(
        "hermes_cli.urllib_security.open_credentialed_url",
        return_value=_Resp(),
    ):
        models = profile.fetch_models(api_key="crsr_test")
    assert models is not None
    assert "grok-4.6" in models
    assert "composer-2.5" in models


def test_fetch_models_parses_live_models_list_shape():
    from providers import get_provider_profile

    payload = json.dumps(
        {"models": ["cursor-grok-4.6-high-fast", "composer-2.5"]}
    ).encode()

    class _Resp:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    profile = get_provider_profile("cursor")
    assert profile is not None
    with patch(
        "hermes_cli.urllib_security.open_credentialed_url",
        return_value=_Resp(),
    ):
        models = profile.fetch_models(api_key="crsr_test")
    assert models is not None
    assert "grok-4.6" in models
    assert "composer-2.5" in models


def test_cursor_client_maps_text_and_tool_calls():
    client = CursorSDKClient(api_key="crsr_test")
    tool_text = (
        "<tool_call>"
        '{"id":"call_1","type":"function",'
        '"function":{"name":"terminal","arguments":"{\\"command\\":\\"uname\\"}"}}'
        "</tool_call>"
    )
    with patch.object(client, "_run_turn", return_value=tool_text):
        completion = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "uname"}],
            tools=[{"type": "function", "function": {"name": "terminal"}}],
        )
    assert completion.choices[0].finish_reason == "tool_calls"
    assert completion.choices[0].message.tool_calls[0].function.name == "terminal"

    with patch.object(client, "_run_turn", return_value="NATIVE_CURSOR_OK"):
        done = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "ping"}],
        )
    assert done.choices[0].finish_reason == "stop"
    assert done.choices[0].message.content == "NATIVE_CURSOR_OK"


def test_cursor_client_requires_key():
    client = CursorSDKClient(api_key="")
    try:
        client._run_prompt("hi", model="grok-4.6")
        raise AssertionError("expected missing-key error")
    except RuntimeError as exc:
        assert "CURSOR_API_KEY" in str(exc)


def test_profile_build_openai_client_returns_shim():
    from providers import get_provider_profile

    profile = get_provider_profile("cursor")
    assert profile is not None
    built = profile.build_openai_client(api_key="crsr_test")
    assert isinstance(built, CursorSDKClient)
    assert built.api_key == "crsr_test"
    assert profile.env_vars == ("CURSOR_API_KEY",)
    assert str(profile.base_url).startswith("cursor-sdk://")


def test_create_openai_client_uses_profile_hook():
    from agent.agent_runtime_helpers import create_openai_client

    fake_client = SimpleNamespace(marker="cursor")
    fake_profile = SimpleNamespace(
        name="cursor",
        build_openai_client=lambda **_k: fake_client,
    )
    agent = SimpleNamespace(
        provider="cursor",
        _client_log_context=lambda: "",
    )
    with (
        patch("providers.get_provider_profile", return_value=fake_profile),
        patch("agent.agent_runtime_helpers._ra") as ra,
    ):
        ra.return_value.logger.info = lambda *a, **k: None
        ra.return_value.OpenAI = lambda **k: (_ for _ in ()).throw(
            AssertionError("OpenAI() must not be called")
        )
        client = create_openai_client(
            agent,
            {"api_key": "crsr_test", "base_url": "cursor-sdk://local"},
            reason="test",
            shared=False,
        )
    assert client is fake_client


def test_window_cursor_messages_keeps_latest_user_and_caps_tail():
    messages = [{"role": "system", "content": "sys " * 100}]
    messages.extend({"role": "assistant", "content": "x" * 4000} for _ in range(20))
    messages.append({"role": "user", "content": "continue now"})
    windowed = window_cursor_messages(messages, budget=12_000)
    assert windowed[0]["role"] == "system"
    assert windowed[-1]["content"] == "continue now"
    body_chars = sum(len(str(m.get("content") or "")) for m in windowed)
    assert body_chars < 30_000


def test_hermes_tools_spec_keeps_all_tools_and_real_parameters():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "vision_analyze",
                "description": "see",
                "parameters": {
                    "type": "object",
                    "properties": {"image_url": {"type": "string"}},
                    "required": ["image_url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cronjob",
                "description": "schedule",
                "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
            },
        },
    ]
    specs = hermes_tools_spec(tools)
    names = [s["name"] for s in specs]
    assert names == ["vision_analyze", "cronjob"]
    assert specs[0]["parameters"]["required"] == ["image_url"]


def test_format_prompt_is_not_acp_and_includes_tool_schema():
    prompt = format_hermes_cursor_prompt(
        [{"role": "user", "content": "see this"}],
        model="grok-4.6",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "vision_analyze",
                    "description": "see",
                    "parameters": {
                        "type": "object",
                        "properties": {"image_url": {"type": "string"}},
                        "required": ["image_url"],
                    },
                },
            }
        ],
    )
    assert "Use ACP capabilities" not in prompt
    assert "active ACP agent backend" not in prompt
    assert "vision_analyze" in prompt
    assert "image_url" in prompt
    assert "see this" in prompt


def test_resume_prompt_includes_tool_result_delta():
    prompt = format_hermes_cursor_prompt(
        [
            {"role": "user", "content": "open last.png"},
            {"role": "assistant", "content": "looking"},
            {"role": "tool", "content": "BLOCKED: already read"},
        ],
        tools=[],
        resume=True,
    )
    assert "BLOCKED: already read" in prompt
    assert "open last.png" in prompt


def test_extract_cursor_images_from_native_vision_envelope():
    messages = [
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Image loaded"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aaa"},
                },
            ],
        }
    ]
    images = extract_cursor_images(messages)
    assert images == [{"data": "aaa", "mime_type": "image/png"}]


def test_extract_cursor_images_from_hermes_nul_json_blob():
    blob = (
        "\0json:"
        + json.dumps(
            [
                {"type": "text", "text": "Image loaded"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,bbb"},
                },
            ]
        )
    )
    images = extract_cursor_images([{"role": "tool", "content": blob}])
    assert images == [{"data": "bbb", "mime_type": "image/png"}]


def test_extract_cursor_images_from_vision_tool_call_path(tmp_path):
    png = tmp_path / "f001.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "vision_analyze",
                        "arguments": json.dumps({"image_url": str(png)}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "Image loaded into your context"},
    ]
    images = extract_cursor_images(messages)
    assert images == [{"path": str(png)}]


def test_extract_cursor_images_from_namespace_tool_calls(tmp_path):
    png = tmp_path / "f003.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    fn = SimpleNamespace(name="vision_analyze", arguments=json.dumps({"image_url": str(png)}))
    call = SimpleNamespace(function=fn)
    messages = [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "content": "Image loaded into your context"},
    ]
    images = extract_cursor_images(messages)
    assert images == [{"path": str(png)}]


def test_run_turn_recreates_agent_when_send_already_active():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    run = SimpleNamespace(text="RECOVERED_OK", wait=lambda: None, cancel=lambda: None)
    created = {"n": 0}

    class _Agent:
        def send(self, _msg):
            if created["n"] == 1:
                raise RuntimeError("internal: Agent agent-x already has active run")
            return run

    def _create(**_k):
        created["n"] += 1
        return _Agent()

    fake = SimpleNamespace(create=_create, prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        text = client._run_turn("hi", model="grok-4.5", resume=False)
    assert text == "RECOVERED_OK"
    assert created["n"] == 2


def test_openai_usage_from_cursor_none_is_zeros():
    usage = _openai_usage_from_cursor(None)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.prompt_tokens_details.cached_tokens == 0


def test_openai_usage_from_cursor_adds_cache_into_prompt_tokens():
    # Cursor input excludes cache; OpenAI prompt_tokens includes it.
    usage = _openai_usage_from_cursor(
        SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=40,
            cache_write_tokens=10,
            total_tokens=170,
            reasoning_tokens=8,
        )
    )
    assert usage.prompt_tokens == 150
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 170
    assert usage.prompt_tokens_details.cached_tokens == 40
    assert usage.prompt_tokens_details.cache_write_tokens == 10
    assert usage.completion_tokens_details.reasoning_tokens == 8


def test_create_surfaces_cursor_run_usage():
    client = CursorSDKClient(api_key="crsr_test")

    def _run(*_a, **_k):
        client._last_usage = SimpleNamespace(
            input_tokens=80,
            output_tokens=12,
            cache_read_tokens=20,
            cache_write_tokens=0,
            total_tokens=112,
        )
        return "NATIVE_CURSOR_OK"

    with patch.object(client, "_run_turn", side_effect=_run):
        done = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "ping"}],
        )
    assert done.choices[0].message.content == "NATIVE_CURSOR_OK"
    assert done.usage.prompt_tokens == 100
    assert done.usage.completion_tokens == 12
    assert done.usage.total_tokens == 112
    assert done.usage.prompt_tokens_details.cached_tokens == 20


def _fake_cursor_sdk(sent: list, *, created: dict, text: str = "NATIVE_CURSOR_OK"):
    """cursor_sdk stand-in that records every send and Agent.create."""
    run = SimpleNamespace(text=text, wait=lambda: None, cancel=lambda: None)

    class _Agent:
        def send(self, msg):
            sent.append(msg)
            return run

    def _create(**_k):
        created["n"] = created.get("n", 0) + 1
        return _Agent()

    return SimpleNamespace(create=_create, prompt=lambda *a, **k: None)


def test_transcript_anchor_tracks_count_and_head():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    count, head = transcript_anchor(messages)
    assert count == 3
    assert head == transcript_anchor([{"role": "user", "content": "first"}])[1]


def test_history_was_rewritten_only_on_shrink_or_new_head():
    grown = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "next"},
    ]
    previous = transcript_anchor(grown[:3])
    assert history_was_rewritten(previous, transcript_anchor(grown)) is False
    # Compression: middle turns dropped, a summary takes the head slot.
    compressed = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "[SUMMARY] earlier turns"},
        {"role": "user", "content": "next"},
    ]
    assert history_was_rewritten(previous, transcript_anchor(compressed)) is True
    assert history_was_rewritten(None, transcript_anchor(grown)) is False


def test_compressed_transcript_starts_a_fresh_cursor_agent():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    sent: list = []
    created: dict = {}
    history = [{"role": "system", "content": "sys"}]
    history += [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    history += [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    history += [{"role": "user", "content": "q3"}]

    with patch.dict(
        "sys.modules", {"cursor_sdk": SimpleNamespace(Agent=_fake_cursor_sdk(sent, created=created))}
    ):
        client.chat.completions.create(model="grok-4.6", messages=history)
        client.chat.completions.create(
            model="grok-4.6",
            messages=history + [{"role": "assistant", "content": "a3"}, {"role": "user", "content": "q4"}],
        )
        # Compression rewrote the transcript: fewer messages, summary head.
        client.chat.completions.create(
            model="grok-4.6",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "[SUMMARY] q1..q3"},
                {"role": "user", "content": "q4"},
            ],
        )

    assert len(sent) == 3
    assert "Conversation:" in sent[0]  # cold start seeds the window
    assert "New turn:" in sent[1]  # live agent gets a delta
    assert "Conversation:" in sent[2]  # rewrite → cold start again
    assert created["n"] == 2


def test_turn_delta_scopes_images_to_the_current_turn():
    older = {
        "role": "tool",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,old"}}
        ],
    }
    messages = [
        {"role": "user", "content": "look at this"},
        older,
        {"role": "assistant", "content": "seen"},
        {"role": "user", "content": "unrelated follow-up"},
    ]
    assert extract_cursor_images(messages) == [{"data": "old", "mime_type": "image/png"}]
    assert extract_cursor_images(turn_delta_messages(messages)) == []


def test_images_are_uploaded_once_per_agent():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    sent: list = []
    created: dict = {}
    messages = [
        {"role": "user", "content": "read the screenshot"},
        {
            "role": "tool",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,shot"}}
            ],
        },
    ]
    with patch.dict(
        "sys.modules", {"cursor_sdk": SimpleNamespace(Agent=_fake_cursor_sdk(sent, created=created))}
    ):
        client.chat.completions.create(model="grok-4.6", messages=messages)
        client.chat.completions.create(
            model="grok-4.6",
            messages=messages + [{"role": "assistant", "content": "ok"}, {"role": "tool", "content": "done"}],
        )

    assert isinstance(sent[0], dict) and sent[0]["images"] == [
        {"data": "shot", "mime_type": "image/png"}
    ]
    assert isinstance(sent[1], str)  # same image, no second upload


def test_cursor_image_key_follows_file_mutations(tmp_path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    first = cursor_image_key({"path": str(shot)})
    shot.write_bytes(b"\x89PNG\r\n\x1a\nDIFFERENT")
    assert cursor_image_key({"path": str(shot)}) != first
    assert cursor_image_key({"url": "https://x/y.png"}) == "url:https://x/y.png"


def test_slot_key_isolates_sessions_and_shares_sku_aliases():
    import os

    client = CursorSDKClient(api_key="crsr_test")
    with patch.dict(os.environ, {"HERMES_SESSION_ID": "sess-a"}):
        a_catalog = client._slot_key("cursor-grok-4.6-high-fast")
        a_alias = client._slot_key("grok-4.6")
        a_low = client._slot_key("cursor-grok-4.6-low-fast")
    with patch.dict(os.environ, {"HERMES_SESSION_ID": "sess-b"}):
        b_alias = client._slot_key("grok-4.6")
    # Honor-both-params (#2): catalog high-fast is not the same SKU as bare
    # grok-4.6 (high / not-fast). Aliases of the *same* params still share.
    assert a_catalog != a_alias
    assert a_low != a_alias  # different reasoning tier → its own agent
    assert a_alias != b_alias  # sessions never share an agent
    with patch.dict(os.environ, {"HERMES_SESSION_ID": "sess-a"}):
        assert client._slot_key("grok-4.6") == client._slot_key("cursor-grok-4.6-high")


def test_slot_key_prefers_the_session_contextvar():
    import os

    from gateway.session_context import _SESSION_ID

    client = CursorSDKClient(api_key="crsr_test")
    token = _SESSION_ID.set("ctx-session")
    try:
        with patch.dict(os.environ, {"HERMES_SESSION_ID": "env-session"}):
            key = client._slot_key("grok-4.6")
    finally:
        _SESSION_ID.reset(token)
    assert "ctx-session" in key
    assert "env-session" not in key


def test_split_cursor_model_id_strips_tier_and_fast_together():
    assert split_cursor_model_id("cursor-grok-4.6-high-fast") == ("grok-4.6", "high", True)
    assert split_cursor_model_id("cursor-grok-4.6-low-fast") == ("grok-4.6", "low", True)
    assert split_cursor_model_id("cursor-grok-4.6-medium-fast") == ("grok-4.6", "medium", True)
    assert split_cursor_model_id("grok-4.6-low") == ("grok-4.6", "low", False)
    assert split_cursor_model_id("composer-2.5") == ("composer-2.5", "", False)
    # Every parse must yield an id the SDK accepts — never a tier-suffixed one.
    for raw in ("cursor-grok-4.6-medium-fast", "grok-4.6-low", "cursor-grok-4.6-high"):
        assert normalize_cursor_model_id(raw) == "grok-4.6"


def test_cursor_aux_http_twin_only_maps_grok_skus():
    assert cursor_aux_http_twin("cursor-grok-4.6-high-fast") == ("xai-oauth", "grok-4.6")
    assert cursor_aux_http_twin("grok-4.6") == ("xai-oauth", "grok-4.6")
    assert cursor_aux_http_twin("cursor-claude-opus-5-high") is None
    assert cursor_aux_http_twin("composer-2.5") is None
    assert cursor_aux_http_twin("gpt-5.5") is None
    assert is_grok_cursor_model("cursor-grok-4.6-low-fast") is True
    assert is_grok_cursor_model("claude-opus-5") is False


def test_cursor_sdk_model_honors_catalog_reasoning_tier():
    low = cursor_sdk_model("cursor-grok-4.6-low-fast")
    assert low["id"] == "grok-4.6"
    assert {p["id"]: p["value"] for p in low["params"]}["reasoning"] == "low"
    # Bare ids keep the deliberate high-not-fast default.
    assert {p["id"]: p["value"] for p in cursor_sdk_model("grok-4.6")["params"]} == {
        "fast": "false",
        "reasoning": "high",
    }


def test_cursor_sdk_error_classifies_auth_and_quota():
    auth = cursor_sdk_error(RuntimeError("401 Unauthorized: invalid api key"), phase="send")
    assert auth.status_code == 401
    assert "CURSOR_API_KEY" in str(auth)
    quota = cursor_sdk_error(RuntimeError('{"error":"rate limit exceeded"}'), phase="run")
    assert quota.status_code == 429
    assert "usage limit" in str(quota).lower()
    unknown = cursor_sdk_error(RuntimeError("bridge exited"), phase="send")
    assert unknown.status_code == 502


def test_cursor_sdk_error_uses_profile_aware_home():
    with patch(
        "agent.cursor_sdk_client.display_hermes_home",
        return_value="~/.hermes/profiles/coder",
    ):
        auth = cursor_sdk_error(
            RuntimeError("401 Unauthorized: invalid api key"), phase="send"
        )
        try:
            CursorSDKClient(api_key="")._run_prompt("hi", model="grok-4.6")
            raise AssertionError("expected missing-key error")
        except CursorSDKError as exc:
            assert "~/.hermes/profiles/coder/.env" in str(exc)
    assert "~/.hermes/profiles/coder/.env" in str(auth)


def test_missing_key_error_is_classified_as_auth():
    client = CursorSDKClient(api_key="")
    try:
        client._run_prompt("hi", model="grok-4.6")
        raise AssertionError("expected missing-key error")
    except CursorSDKError as exc:
        assert exc.status_code == 401


def test_empty_cursor_response_raises_and_drops_the_agent():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    sent: list = []
    created: dict = {}
    fake = _fake_cursor_sdk(sent, created=created, text="   ")
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        try:
            client.chat.completions.create(
                model="grok-4.6", messages=[{"role": "user", "content": "ping"}]
            )
            raise AssertionError("expected an empty-response error")
        except CursorSDKError as exc:
            assert exc.status_code == 502
    assert all(rec.get("agent") is None for rec in mod._slots.values())


def test_busy_agent_on_a_resume_turn_reseeds_the_full_window():
    """A replacement agent must never be handed an orphaned delta prompt."""
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    sent: list = []
    created: dict = {}
    run = SimpleNamespace(text="OK", wait=lambda: None, cancel=lambda: None)

    class _Agent:
        """Accepts one send, then reports its run as still active."""

        def __init__(self):
            self.sends = 0

        def send(self, msg):
            self.sends += 1
            if self.sends > 1:
                raise RuntimeError("internal: Agent agent-x already has active run")
            sent.append(msg)
            return run

    def _create(**_k):
        created["n"] = created.get("n", 0) + 1
        return _Agent()

    fake = SimpleNamespace(create=_create, prompt=lambda *a, **k: None)
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "remember apples"},
        {"role": "assistant", "content": "noted"},
    ]
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        client.chat.completions.create(model="grok-4.6", messages=history)
        client.chat.completions.create(
            model="grok-4.6",
            messages=history + [{"role": "user", "content": "what fruit?"}],
        )

    assert created["n"] == 2  # the busy agent is replaced, not reused
    assert len(sent) == 2
    assert "remember apples" in sent[1]  # the reseed carries the history
    assert "Conversation:" in sent[1]


def test_interrupt_keeps_the_live_agent_for_the_next_turn():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    cancelled: list = []

    def _wait():
        raise KeyboardInterrupt

    run = SimpleNamespace(
        text="", wait=_wait, cancel=lambda: cancelled.append(True)
    )

    class _Agent:
        def send(self, _msg):
            return run

    fake = SimpleNamespace(create=lambda **_k: _Agent(), prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        try:
            client.chat.completions.create(
                model="grok-4.6", messages=[{"role": "user", "content": "long task"}]
            )
            raise AssertionError("expected the interrupt to propagate")
        except KeyboardInterrupt:
            pass

    assert cancelled  # the orphaned run is cancelled, not left open
    assert any(rec.get("agent") is not None for rec in mod._slots.values())
    mod._slots.clear()


def test_idle_slots_over_the_cap_are_evicted_and_closed():
    import time as _time

    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    closed: list = []
    stale = _time.monotonic() - mod._SLOT_IDLE_SECONDS - 60
    for i in range(mod._MAX_LIVE_SLOTS + 1):
        rec = mod._slot_record(f"slot-{i}")
        rec["agent"] = SimpleNamespace(close=lambda i=i: closed.append(i))
        rec["used"] = stale + i
    mod._evict_idle_slots(f"slot-{mod._MAX_LIVE_SLOTS}")
    live = [k for k, rec in mod._slots.items() if rec.get("agent") is not None]
    assert len(live) <= mod._MAX_LIVE_SLOTS
    assert closed == [0]  # the bridge is torn down, not just dereferenced
    mod._slots.clear()


def test_busy_multi_chat_slots_are_not_evicted():
    """A gateway juggling several live chats must not pay repeated cold starts."""
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    closed: list = []
    for i in range(mod._MAX_LIVE_SLOTS + 2):
        rec = mod._slot_record(f"chat-{i}")
        rec["agent"] = SimpleNamespace(close=lambda i=i: closed.append(i))
    mod._evict_idle_slots("chat-0")
    assert closed == []
    mod._slots.clear()


def test_run_turn_reads_usage_from_wait_result():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    cursor_usage = SimpleNamespace(
        input_tokens=42,
        output_tokens=7,
        cache_read_tokens=0,
        cache_write_tokens=0,
        total_tokens=49,
    )
    waited = SimpleNamespace(result="USAGE_OK", usage=cursor_usage)
    run = SimpleNamespace(text="USAGE_OK", wait=lambda: waited, cancel=lambda: None)

    class _Agent:
        def send(self, _msg):
            return run

    fake = SimpleNamespace(create=lambda **_k: _Agent(), prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        text = client._run_turn("hi", model="grok-4.6", resume=False)
    assert text == "USAGE_OK"
    assert client._last_usage is cursor_usage


def test_run_turn_cancels_and_closes_on_keyboard_interrupt():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    state = {"cancelled": False, "closed": False}
    run = SimpleNamespace(
        text=None,
        wait=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
        cancel=lambda: state.update(cancelled=True),
    )

    class _Agent:
        def send(self, _msg):
            return run

        def close(self):
            state["closed"] = True

    fake = SimpleNamespace(create=lambda **_k: _Agent(), prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        with pytest.raises(KeyboardInterrupt):
            client._run_turn("hi", model="grok-4.6", resume=False)
    assert state["cancelled"] is True
    # Keep the agent so the next turn is not an amnesia cold-start (#5).
    assert state["closed"] is False


def test_run_turn_raises_on_error_status_instead_of_empty_text():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    state = {"closed": False}
    waited = SimpleNamespace(status="error", error="model exploded", result=None, usage=None)
    run = SimpleNamespace(text="", wait=lambda: waited, cancel=lambda: None)

    class _Agent:
        def send(self, _msg):
            return run

        def close(self):
            state["closed"] = True

    fake = SimpleNamespace(create=lambda **_k: _Agent(), prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        with pytest.raises(RuntimeError, match="model exploded"):
            client._run_turn("hi", model="grok-4.6", resume=False)
    assert state["closed"] is True


def test_run_turn_success_status_passes_through():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    waited = SimpleNamespace(status="finished", result="DONE_OK", usage=None)
    run = SimpleNamespace(text=None, wait=lambda: waited, cancel=lambda: None)

    class _Agent:
        def send(self, _msg):
            return run

    fake = SimpleNamespace(create=lambda **_k: _Agent(), prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        text = client._run_turn("hi", model="grok-4.6", resume=False)
    assert text == "DONE_OK"


def test_startup_error_wraps_auth_failures_with_guidance():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")

    class _AgentFactory:
        @staticmethod
        def create(**_k):
            raise Exception("CursorAgentError: 401 Unauthorized: invalid key")

    fake = SimpleNamespace(create=_AgentFactory.create, prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        with pytest.raises(RuntimeError) as excinfo:
            client._run_turn("hi", model="grok-4.6", resume=False)
    msg = str(excinfo.value)
    assert "CURSOR_API_KEY" in msg
    assert "cursor.com/dashboard/api" in msg
    # Original text retained so the Hermes error classifier still sees auth markers.
    assert "401" in msg and "unauthorized" in msg.lower()


def test_send_error_non_active_run_is_wrapped_and_agent_dropped():
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    state = {"closed": 0}

    class _Agent:
        def send(self, _msg):
            raise Exception("CursorAgentError: connection refused by bridge")

        def close(self):
            state["closed"] += 1

    fake = SimpleNamespace(create=lambda **_k: _Agent(), prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        with pytest.raises(RuntimeError, match="send failed"):
            client._run_turn("hi", model="grok-4.6", resume=False)
    assert state["closed"] == 1


def test_oneshot_prompt_error_is_wrapped():
    client = CursorSDKClient(api_key="crsr_test")

    def _prompt(*_a, **_k):
        raise Exception("CursorAgentError: 403 Forbidden")

    fake = SimpleNamespace(create=lambda **_k: None, prompt=_prompt)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        with pytest.raises(RuntimeError) as excinfo:
            client._run_prompt("hi", model="grok-4.6")
    assert "CURSOR_API_KEY" in str(excinfo.value)


def test_cursor_sdk_model_honors_catalog_variants():
    low = cursor_sdk_model("cursor-grok-4.6-low-fast")
    assert low["id"] == "grok-4.6"
    assert {p["id"]: p["value"] for p in low["params"]} == {
        "fast": "true",
        "reasoning": "low",
    }
    high_fast = cursor_sdk_model("cursor-grok-4.6-high-fast")
    assert {p["id"]: p["value"] for p in high_fast["params"]} == {
        "fast": "true",
        "reasoning": "high",
    }
    # Bare id keeps the tuned high / not-fast default.
    bare = cursor_sdk_model("grok-4.6")
    assert {p["id"]: p["value"] for p in bare["params"]} == {
        "fast": "false",
        "reasoning": "high",
    }
    assert cursor_sdk_model("composer-2.5") == "composer-2.5"


def test_normalize_cursor_model_id_strips_low_variant():
    assert normalize_cursor_model_id("cursor-grok-4.6-low-fast") == "grok-4.6"
    assert normalize_cursor_model_id("cursor-grok-4.6-low") == "grok-4.6"


def test_slot_key_prefers_explicit_session_id_over_process_env(monkeypatch):
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    monkeypatch.setenv("HERMES_SESSION_ID", "env-session")
    monkeypatch.delenv("HERMES_CURSOR_SESSION", raising=False)
    client = CursorSDKClient(api_key="crsr_test", session_id="agent-session")
    assert client._slot_key("grok-4.6").startswith("agent-session::")
    # Manual override env still wins over everything.
    monkeypatch.setenv("HERMES_CURSOR_SESSION", "pinned")
    assert client._slot_key("grok-4.6").startswith("pinned::")
    # No explicit binding falls back to the process env (CLI single session).
    plain = CursorSDKClient(api_key="crsr_test")
    monkeypatch.delenv("HERMES_CURSOR_SESSION", raising=False)
    assert plain._slot_key("grok-4.6").startswith("env-session::")


def test_slot_key_tracks_live_session_id_fn_after_init_and_rotation(monkeypatch):
    """session_id_fn is read at slot-key time, not frozen at construction.

    Mirrors agent_init (client built while session_id is still None) and
    /new (session_id rotates without rebuilding the client).
    """
    from agent import cursor_sdk_client as mod

    mod._slots.clear()
    monkeypatch.setenv("HERMES_SESSION_ID", "env-session")
    monkeypatch.delenv("HERMES_CURSOR_SESSION", raising=False)
    state = {"sid": None}
    client = CursorSDKClient(
        api_key="crsr_test",
        session_id=None,
        session_id_fn=lambda: state["sid"],
    )
    # Construction-time None must not pin the slot to the process env forever.
    assert client._slot_key("grok-4.6").startswith("env-session::")
    state["sid"] = "assigned-after-init"
    assert client._slot_key("grok-4.6").startswith("assigned-after-init::")
    state["sid"] = "rotated-by-new"
    assert client._slot_key("grok-4.6").startswith("rotated-by-new::")
    # Stale constructor snapshot must not win over the live getter.
    stale = CursorSDKClient(
        api_key="crsr_test",
        session_id="frozen-at-init",
        session_id_fn=lambda: state["sid"],
    )
    assert stale._slot_key("grok-4.6").startswith("rotated-by-new::")
    # Manual override env still wins over the live getter.
    monkeypatch.setenv("HERMES_CURSOR_SESSION", "pinned")
    assert client._slot_key("grok-4.6").startswith("pinned::")


def test_create_openai_client_live_session_id_survives_init_order_and_new(monkeypatch):
    """Wiring: create_openai_client before session_id, then /new, same client."""
    from agent.agent_runtime_helpers import create_openai_client

    monkeypatch.setenv("HERMES_SESSION_ID", "process-global")
    monkeypatch.delenv("HERMES_CURSOR_SESSION", raising=False)

    kwargs = {"api_key": "crsr_test", "base_url": "cursor-sdk://local"}
    agent_a = SimpleNamespace(
        provider="cursor",
        session_id=None,
        _client_log_context=lambda: "",
    )
    agent_b = SimpleNamespace(
        provider="cursor",
        session_id=None,
        _client_log_context=lambda: "",
    )
    with patch("agent.agent_runtime_helpers._ra") as ra:
        ra.return_value.logger.info = lambda *a, **k: None
        client_a = create_openai_client(
            agent_a, kwargs, reason="agent_init", shared=True
        )
        client_b = create_openai_client(
            agent_b, kwargs, reason="agent_init", shared=True
        )

    assert isinstance(client_a, CursorSDKClient)
    assert isinstance(client_b, CursorSDKClient)
    # Snapshot at agent_init is None — process env would bleed both slots.
    assert client_a._session_id is None
    assert client_b._session_id is None
    assert client_a._slot_key("grok-4.6").startswith("process-global::")

    # agent_init then assigns session_id (lines ~1577–1584) without rebuild.
    agent_a.session_id = "session-a"
    agent_b.session_id = "session-b"
    key_a = client_a._slot_key("grok-4.6")
    key_b = client_b._slot_key("grok-4.6")
    assert key_a.startswith("session-a::")
    assert key_b.startswith("session-b::")
    assert key_a != key_b

    # /new rotates agent.session_id on the same client instance.
    agent_a.session_id = "session-a-after-new"
    assert client_a._slot_key("grok-4.6").startswith("session-a-after-new::")
    assert client_b._slot_key("grok-4.6").startswith("session-b::")


_TERMINAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def test_cursor_client_maps_invoke_xml_to_hermes_tool_calls():
    client = CursorSDKClient(api_key="crsr_test")
    xml = (
        '<function_calls>\n'
        '<invoke name="terminal">\n'
        "<parameter name=\"command\">uname</parameter>\n"
        "</invoke>\n"
        "</function_calls>"
    )
    with patch.object(client, "_run_turn", return_value=xml) as run:
        completion = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "uname"}],
            tools=_TERMINAL_TOOLS,
        )
    assert run.call_count == 1
    assert completion.choices[0].finish_reason == "tool_calls"
    call = completion.choices[0].message.tool_calls[0]
    assert call.function.name == "terminal"
    assert json.loads(call.function.arguments) == {"command": "uname"}


def test_cursor_custom_tools_capture_becomes_hermes_tool_calls():
    client = CursorSDKClient(api_key="crsr_test")

    def _run(*_a, **_k):
        client._last_captures = [("terminal", {"command": "uname"}, "call_x")]
        return ""

    with patch.object(client, "_run_turn", side_effect=_run) as run:
        completion = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "uname"}],
            tools=_TERMINAL_TOOLS,
        )
    assert run.call_count == 1
    assert completion.choices[0].finish_reason == "tool_calls"
    call = completion.choices[0].message.tool_calls[0]
    assert call.function.name == "terminal"
    assert call.id == "call_x"
    assert json.loads(call.function.arguments) == {"command": "uname"}


def test_untranslatable_tool_not_found_fails_once_without_second_run_turn():
    from agent.copilot_acp_client import BRIDGE_WRONG_TOOL_FORMAT

    client = CursorSDKClient(api_key="crsr_test")
    not_found = "Tool not found: mystery_tool\nAvailable tools:"
    with patch.object(client, "_run_turn", return_value=not_found) as run:
        completion = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "uname"}],
            tools=_TERMINAL_TOOLS,
        )
    assert run.call_count == 1
    assert completion.choices[0].finish_reason == "stop"
    assert completion.choices[0].message.tool_calls in (None, [], ())
    content = completion.choices[0].message.content
    assert content == BRIDGE_WRONG_TOOL_FORMAT
    assert "Available tools:" not in content


def test_run_turn_registers_custom_tools_deferral_shim_and_keeps_builtins_off():
    from agent import cursor_sdk_client as mod
    from agent.copilot_acp_client import _CUSTOM_TOOL_DEFERRAL

    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    captured_options: dict = {}

    class _Agent:
        def send(self, _msg):
            custom = captured_options["local"]["custom_tools"]
            assert captured_options["tools"] == []
            assert captured_options["mcp_servers"] == {}
            assert "terminal" in custom
            defer = custom["terminal"]["execute"](
                {"command": "uname"}, SimpleNamespace(tool_call_id="c1")
            )
            assert defer == _CUSTOM_TOOL_DEFERRAL
            return SimpleNamespace(
                text=_CUSTOM_TOOL_DEFERRAL,
                wait=lambda: SimpleNamespace(
                    result=_CUSTOM_TOOL_DEFERRAL, status="finished"
                ),
                cancel=lambda: None,
            )

    def _create(**kwargs):
        captured_options.update(kwargs.get("options") or {})
        return _Agent()

    fake = SimpleNamespace(create=_create, prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        text = client._run_turn(
            "hi",
            model="grok-4.6",
            resume=False,
            tools=_TERMINAL_TOOLS,
        )
    assert text == _CUSTOM_TOOL_DEFERRAL
    assert client._last_captures[0][0] == "terminal"
    assert client._last_captures[0][1] == {"command": "uname"}
    assert client._last_captures[0][2] == "c1"


def test_untranslatable_markup_drops_agent_so_next_turn_is_not_resume(tmp_path, monkeypatch):
    from agent import cursor_sdk_client as mod
    from agent.copilot_acp_client import BRIDGE_WRONG_TOOL_FORMAT

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    mod._slots.clear()
    client = CursorSDKClient(api_key="crsr_test")
    creates = {"n": 0}
    not_found = "Tool not found: terminal\nAvailable tools:"

    class _Agent:
        def send(self, _msg):
            return SimpleNamespace(
                text=not_found,
                wait=lambda: SimpleNamespace(result=not_found, status="finished"),
                cancel=lambda: None,
            )

    def _create(**_k):
        creates["n"] += 1
        return _Agent()

    fake = SimpleNamespace(create=_create, prompt=lambda *a, **k: None)
    with patch.dict("sys.modules", {"cursor_sdk": SimpleNamespace(Agent=fake)}):
        first = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "uname"}],
            tools=_TERMINAL_TOOLS,
        )
        assert first.choices[0].message.content == BRIDGE_WRONG_TOOL_FORMAT
        assert creates["n"] == 1
        slot = client._slot_key("grok-4.6")
        rec = mod._slots.get(slot)
        assert rec is None or rec.get("agent") is None
        second = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "uname again"}],
            tools=_TERMINAL_TOOLS,
        )
    assert second.choices[0].message.content == BRIDGE_WRONG_TOOL_FORMAT
    assert creates["n"] == 2
