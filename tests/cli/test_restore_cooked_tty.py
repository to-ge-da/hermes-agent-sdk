"""Regression: /exit must restore cooked+echo after the TUI healer.

The SDK fork's ``_heal_cooked_mode_drift`` strips ECHO/ICANON while the TUI
is alive. After a longer session that healer (or a nested run_in_terminal
restore) can race past prompt_toolkit's unwind and leave the parent shell
with echo off — commands still print, but typing is invisible until SSH is
killed. Quick /exit does not reproduce because the healer never ran.
"""

import os
import sys

import pytest

if sys.platform == "win32":  # pragma: no cover
    pytest.skip("cooked tty restore is POSIX-only", allow_module_level=True)

import termios
import tty as _tty

import cli as cli_mod


@pytest.fixture()
def pty_fd():
    master, slave = os.openpty()
    try:
        yield slave
    finally:
        os.close(master)
        os.close(slave)


@pytest.fixture()
def tty_globals():
    snap = (
        cli_mod._orig_tty_attrs,
        cli_mod._orig_tty_fd,
        cli_mod._tui_exiting,
        cli_mod._tui_input_modes_active,
        cli_mod._cleanup_done,
        cli_mod._cleanup_in_progress,
    )
    cli_mod._orig_tty_attrs = None
    cli_mod._orig_tty_fd = None
    cli_mod._tui_exiting = False
    cli_mod._tui_input_modes_active = False
    yield
    (
        cli_mod._orig_tty_attrs,
        cli_mod._orig_tty_fd,
        cli_mod._tui_exiting,
        cli_mod._tui_input_modes_active,
        cli_mod._cleanup_done,
        cli_mod._cleanup_in_progress,
    ) = snap


def _is_cooked(fd: int) -> bool:
    lflag = termios.tcgetattr(fd)[3]
    return bool(lflag & termios.ECHO) and bool(lflag & termios.ICANON)


def test_restore_uses_captured_pre_tui_attrs(pty_fd, tty_globals):
    assert _is_cooked(pty_fd)
    cli_mod._capture_tty_attrs(pty_fd)
    _tty.setraw(pty_fd)
    assert not _is_cooked(pty_fd)

    cli_mod._restore_cooked_tty(pty_fd)

    assert _is_cooked(pty_fd)


def test_restore_fallback_without_snapshot(pty_fd, tty_globals):
    _tty.setraw(pty_fd)
    assert not _is_cooked(pty_fd)

    cli_mod._restore_cooked_tty(pty_fd)

    assert _is_cooked(pty_fd)


def test_restore_idempotent_when_already_cooked(pty_fd, tty_globals):
    before = termios.tcgetattr(pty_fd)
    cli_mod._restore_cooked_tty(pty_fd)
    assert termios.tcgetattr(pty_fd) == before


def test_healer_then_restore_recovers_echo(pty_fd, tty_globals):
    """The live failure: healer puts the tty raw after PT already restored."""
    cli_mod._capture_tty_attrs(pty_fd)
    assert cli_mod._heal_cooked_mode_drift(pty_fd) is True
    assert not _is_cooked(pty_fd)

    cli_mod._restore_cooked_tty(pty_fd)

    assert _is_cooked(pty_fd)


def test_reset_on_exit_restores_cooked_even_when_flag_already_cleared(
    pty_fd, tty_globals
):
    cli_mod._capture_tty_attrs(pty_fd)
    _tty.setraw(pty_fd)
    cli_mod._tui_input_modes_active = False

    cli_mod._reset_terminal_input_modes_on_exit()
    cli_mod._restore_cooked_tty(pty_fd)

    assert _is_cooked(pty_fd)
    assert cli_mod._tui_exiting is True


def test_healer_skips_after_should_exit(tty_globals, monkeypatch):
    called = []
    monkeypatch.setattr(
        cli_mod, "_heal_cooked_mode_drift", lambda fd: called.append(fd) or True
    )

    class _Stub:
        _should_exit = True
        _app = type("A", (), {"_is_running": True, "_running_in_terminal": False})()
        _last_termios_drift_check = 0.0

    cli_mod.HermesCLI._check_termios_drift(_Stub())
    assert called == []


def test_healer_skips_when_tui_exiting(tty_globals, monkeypatch):
    cli_mod._tui_exiting = True
    called = []
    monkeypatch.setattr(
        cli_mod, "_heal_cooked_mode_drift", lambda fd: called.append(fd) or True
    )

    class _Stub:
        _should_exit = False
        _app = type("A", (), {"_is_running": True, "_running_in_terminal": False})()
        _last_termios_drift_check = 0.0

    cli_mod.HermesCLI._check_termios_drift(_Stub())
    assert called == []


def test_mark_tui_active_captures_attrs(pty_fd, tty_globals):
    # Capture via the public TUI-start hook by feeding the pty fd.
    cli_mod._capture_tty_attrs(pty_fd)
    cli_mod._mark_tui_input_modes_active()
    assert cli_mod._tui_input_modes_active is True
    assert cli_mod._orig_tty_attrs is not None
    # First snapshot wins even if the tty later goes raw.
    _tty.setraw(pty_fd)
    cli_mod._capture_tty_attrs(pty_fd)
    cli_mod._restore_cooked_tty(pty_fd)
    assert _is_cooked(pty_fd)
