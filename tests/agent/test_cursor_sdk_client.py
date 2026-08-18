"""Cursor native provider — catalog parse, client shim, no live network."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.cursor_sdk_client import (
    CursorSDKClient,
    cursor_sdk_model,
    extract_cursor_images,
    expand_cursor_model_ids,
    format_hermes_cursor_prompt,
    hermes_tools_spec,
    normalize_cursor_model_id,
    window_cursor_messages,
)


def test_normalize_cursor_model_id_strips_catalog_affixes():
    assert normalize_cursor_model_id("cursor-grok-4.6-high-fast") == "grok-4.6"
    assert normalize_cursor_model_id("grok-4.6") == "grok-4.6"
    assert normalize_cursor_model_id("composer-2.5") == "composer-2.5"


def test_cursor_sdk_model_uses_high_without_fast_for_grok():
    sel = cursor_sdk_model("grok-4.6")
    assert sel["id"] == "grok-4.6"
    params = {p["id"]: p["value"] for p in sel["params"]}
    assert params["fast"] == "false"
    assert params["reasoning"] == "high"
    assert cursor_sdk_model("composer-2.5") == "composer-2.5"


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
