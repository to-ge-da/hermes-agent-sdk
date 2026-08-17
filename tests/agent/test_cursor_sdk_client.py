"""Cursor native provider — catalog parse, client shim, no live network."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.cursor_sdk_client import (
    CursorSDKClient,
    expand_cursor_model_ids,
    normalize_cursor_model_id,
)


def test_normalize_cursor_model_id_strips_catalog_affixes():
    assert normalize_cursor_model_id("cursor-grok-4.6-high-fast") == "grok-4.6"
    assert normalize_cursor_model_id("grok-4.6") == "grok-4.6"
    assert normalize_cursor_model_id("composer-2.5") == "composer-2.5"


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


def test_cursor_client_maps_text_and_tool_calls():
    client = CursorSDKClient(api_key="crsr_test")
    tool_text = (
        "<tool_call>"
        '{"id":"call_1","type":"function",'
        '"function":{"name":"terminal","arguments":"{\\"command\\":\\"uname\\"}"}}'
        "</tool_call>"
    )
    with patch.object(client, "_run_prompt", return_value=tool_text):
        completion = client.chat.completions.create(
            model="grok-4.6",
            messages=[{"role": "user", "content": "uname"}],
            tools=[{"type": "function", "function": {"name": "terminal"}}],
        )
    assert completion.choices[0].finish_reason == "tool_calls"
    assert completion.choices[0].message.tool_calls[0].function.name == "terminal"

    with patch.object(client, "_run_prompt", return_value="NATIVE_CURSOR_OK"):
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
