"""Issue #8: no auxiliary LLM provider must not deadlock auto-compression.

When no summarizer provider is configured, ``_generate_summary`` used to arm the
600s summary-failure cooldown. While armed,
``_automatic_compression_blocked_locally()`` returns True and ``should_compress()``
returns False, so the deterministic (non-LLM) fallback — the only shrink
available in that state — never runs and the session stalls above the threshold.

"No provider configured" is not transient: nothing will clear it within the
cooldown. Real transient failures (429/timeout/malformed/stream-closed) must
still arm the cooldown; that is the #11529 thrash protection.
"""

import time
from unittest.mock import patch

from agent.context_compressor import ContextCompressor

# The exact shape auxiliary_client.py raises when no provider is resolvable.
_NO_PROVIDER_ERROR = RuntimeError(
    "No LLM provider configured for task=compression provider=auto. "
    "Run `hermes setup` to configure one."
)


def _make_compressor(**kwargs) -> ContextCompressor:
    """Build a real ContextCompressor with a deterministic context window."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(
            model="test-model",
            quiet_mode=True,
            protect_first_n=1,
            protect_last_n=2,
            **kwargs,
        )


def _turns(n: int = 6):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "x" * 200}
        for i in range(n)
    ]


def test_no_aux_provider_does_not_arm_cooldown():
    """Issue #8: the no-provider branch must not block automatic compaction."""
    comp = _make_compressor()

    with patch("agent.context_compressor.call_llm", side_effect=_NO_PROVIDER_ERROR):
        summary = comp._generate_summary(_turns())

    assert summary is None
    assert comp._summary_failure_cooldown_until <= time.monotonic(), (
        "no-aux provider must not arm the summary-failure cooldown"
    )
    assert comp._automatic_compression_blocked_locally() is False, (
        "automatic compaction must stay unblocked so the deterministic "
        "fallback can run"
    )
    # The condition is still surfaced to the user via _last_summary_error.
    assert comp._last_summary_error == "no auxiliary LLM provider configured"
    # Abort guards untouched — no auth/network failure was observed.
    assert comp._last_summary_auth_failure is False
    assert comp._last_summary_network_failure is False


def test_transient_summary_failure_still_arms_cooldown():
    """#11529 boundary: real transient failures keep the cooldown."""
    for error in (
        RuntimeError("Request timed out after 60s"),
        RuntimeError("429 Too Many Requests"),
    ):
        comp = _make_compressor()
        with patch("agent.context_compressor.call_llm", side_effect=error):
            summary = comp._generate_summary(_turns())

        assert summary is None
        assert comp._summary_failure_cooldown_until > time.monotonic(), (
            f"transient failure {error!r} must still arm the cooldown"
        )
        assert comp._automatic_compression_blocked_locally() is True


def test_no_aux_provider_runs_deterministic_fallback():
    """With no cooldown armed, compress() reaches the static fallback."""
    comp = _make_compressor()
    messages = [{"role": "system", "content": "System prompt"}] + _turns(12)

    with patch("agent.context_compressor.call_llm", side_effect=_NO_PROVIDER_ERROR):
        result = comp.compress(messages)

    assert len(result) < len(messages), "deterministic fallback must shrink the transcript"
    assert comp._last_summary_fallback_used is True
    assert comp._last_compress_aborted is False
    assert comp._automatic_compression_blocked_locally() is False, (
        "the next turn must not be blocked by a cooldown no provider can clear"
    )
