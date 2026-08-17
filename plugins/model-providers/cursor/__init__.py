"""Cursor Agent SDK provider profile.

Cursor dashboard keys (`crsr_…`, env ``CURSOR_API_KEY``) authenticate the
official Cloud Agents / Agent SDK. There is no public OpenAI-compatible
``POST /v1/chat/completions`` on api.cursor.com — inference goes through
``cursor-sdk`` (lazy import, not a core dependency).

Hermes stays the control plane: the client flattens the Hermes transcript
and tool schema into a prompt and maps ``<tool_call>`` blocks back onto
Hermes tools, same shape as Copilot ACP.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

log = logging.getLogger(__name__)

CURSOR_MARKER_BASE_URL = "cursor-sdk://local"
CURSOR_MODELS_URL = "https://api.cursor.com/v0/models"


class CursorProfile(ProviderProfile):
    """Cursor Agent SDK — catalog via /v0/models, inference via cursor-sdk."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """GET https://api.cursor.com/v0/models (items[].id), not /v1/models."""
        import urllib.request

        from hermes_cli.urllib_security import open_credentialed_url

        # Ignore the cursor-sdk:// marker — the catalog is always first-party.
        url = CURSOR_MODELS_URL
        parsed = urlparse(str(base_url or "").strip())
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            url = str(base_url).rstrip("/") + "/models"

        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", f"hermes-cli/{_HERMES_VERSION}")
        for key, value in self.default_headers.items():
            req.add_header(key, value)

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            log.debug("fetch_models(cursor): %s", exc)
            return None

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("items") or data.get("data") or data.get("models") or []
        else:
            return None

        raw_ids: list[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                raw_ids.append(item.strip())
            elif isinstance(item, dict):
                mid = item.get("id") or item.get("name")
                if isinstance(mid, str) and mid.strip():
                    raw_ids.append(mid.strip())
        from agent.cursor_sdk_client import expand_cursor_model_ids

        return expand_cursor_model_ids(raw_ids) or None

    def build_openai_client(self, **kwargs: Any) -> Any:
        from agent.cursor_sdk_client import CursorSDKClient

        return CursorSDKClient(**kwargs)


cursor = CursorProfile(
    name="cursor",
    aliases=("cursor-sdk", "cursor-composer"),
    display_name="Cursor",
    description="Cursor Agent SDK — CURSOR_API_KEY (crsr_… dashboard key)",
    signup_url="https://cursor.com/dashboard/api",
    env_vars=("CURSOR_API_KEY",),
    base_url=CURSOR_MARKER_BASE_URL,
    models_url=CURSOR_MODELS_URL,
    auth_type="api_key",
    default_headers={"User-Agent": f"HermesAgent/{_HERMES_VERSION}"},
    fallback_models=(
        "grok-4.6",
        "composer-2.5",
        "composer-2",
    ),
)

register_provider(cursor)
