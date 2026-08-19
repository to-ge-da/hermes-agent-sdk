"""Tests for warn_deprecated_cwd_env_vars() migration warning."""


class TestDeprecatedCwdWarning:
    """Warn when MESSAGING_CWD or TERMINAL_CWD is set in .env."""

    def test_messaging_cwd_triggers_warning(self, monkeypatch, capsys):
        monkeypatch.setenv("MESSAGING_CWD", "/some/path")
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        from hermes_cli.config import warn_deprecated_cwd_env_vars
        warn_deprecated_cwd_env_vars(config={})

        captured = capsys.readouterr()
        assert "MESSAGING_CWD" in captured.err
        assert "deprecated" in captured.err.lower()
        assert "config.yaml" in captured.err


    def test_both_deprecated_vars_warn(self, monkeypatch, capsys):
        monkeypatch.setenv("MESSAGING_CWD", "/msg/path")
        monkeypatch.setenv("TERMINAL_CWD", "/term/path")

        from hermes_cli.config import warn_deprecated_cwd_env_vars
        warn_deprecated_cwd_env_vars(config={})

        captured = capsys.readouterr()
        assert "MESSAGING_CWD" in captured.err
        assert "TERMINAL_CWD" in captured.err

    def test_terminal_cwd_autmigrates_on_startup(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_MANAGED", raising=False)
        monkeypatch.setenv("TERMINAL_CWD", "/home/testnet")
        monkeypatch.delenv("MESSAGING_CWD", raising=False)

        from hermes_cli.config import read_raw_config, warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        captured = capsys.readouterr()
        assert "TERMINAL_CWD" not in captured.err
        assert read_raw_config().get("terminal", {}).get("cwd") == "/home/testnet"
