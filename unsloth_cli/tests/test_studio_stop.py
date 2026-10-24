# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth studio stop must stop every server started; one PID file was overwritten per launch."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


# Captured before _install stubs it, for the tests that exercise it.
_REAL_IS_STUDIO_SERVER = _studio()._pid_is_studio_server


def _install(
    monkeypatch,
    tmp_path,
    *,
    alive,
    killed = None,
):
    """Point the CLI at tmp_path and fake process liveness."""
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)

    live = set(alive)
    killed = killed if killed is not None else []

    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: pid in live)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)

    def fake_kill(pid, _sig):
        killed.append(pid)
        live.discard(pid)

    monkeypatch.setattr(studio_mod.os, "kill", fake_kill)
    monkeypatch.setattr(sys, "platform", "linux")
    return studio_mod, live, killed


def _write_pid(tmp_path, name, pid):
    (tmp_path / name).write_text(str(pid), encoding = "utf-8")


def _run_stop(studio_mod):
    import typer as _typer

    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    return CliRunner().invoke(app, ["studio", "stop"])


def test_stop_kills_every_recorded_server(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550, 8600})
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio-8902-8600.pid", 8600)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert sorted(killed) == [8550, 8600]
    assert not list(tmp_path.glob("studio-*.pid"))


def test_stop_does_not_leave_the_older_instance_running(monkeypatch, tmp_path):
    studio_mod, live, _killed = _install(monkeypatch, tmp_path, alive = {8550, 8600})
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio-8902-8600.pid", 8600)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert live == set()


def test_stop_signals_each_server_once(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)
    killed = []
    monkeypatch.setattr(studio_mod.os, "kill", lambda pid, _sig: killed.append(pid))
    monkeypatch.setattr(sys, "platform", "linux")
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == [8550]
    assert result.output.lower().count("sent shutdown signal") == 1


def test_stop_removes_every_stale_file_for_one_pid(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = set())
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not list(tmp_path.glob("*.pid"))


def test_stop_does_not_signal_a_reused_pid(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): False)
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 1, result.output
    assert killed == []
    assert (tmp_path / "studio-8901-8550.pid").exists()


def test_stop_signals_a_live_server_whose_pid_has_a_stale_record(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))
    (tmp_path / "studio-8888-8550.pid").write_text("8550\n111.5", encoding = "utf-8")
    (tmp_path / "studio-9000-8550.pid").write_text("8550\n999.0", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == [8550]
    assert not list(tmp_path.glob("studio-*.pid"))


def test_a_bare_run_py_command_line_is_not_rejected(monkeypatch):
    studio_mod = _studio()

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return ["python", "run.py", "--port", "8901"]

        def create_time(self):
            return 111.5

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert studio_mod._pid_is_studio_server(8550) is True


def test_an_untimed_record_is_trusted(monkeypatch):
    studio_mod = _studio()

    assert studio_mod._pid_is_studio_server(8550) is True
    assert studio_mod._pid_is_studio_server(8550, [None]) is True


def test_an_unverifiable_record_is_still_stopped(monkeypatch):
    studio_mod = _studio()
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert studio_mod._pid_is_studio_server(8550, [111.5]) is True
    assert studio_mod._pid_is_studio_server(8550, [None]) is True


def test_stop_signals_a_timestamped_record_without_psutil(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)
    monkeypatch.setitem(sys.modules, "psutil", None)
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n111.5", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == [8550]
    assert not (tmp_path / "studio-8901-8550.pid").exists()


def test_the_untimed_legacy_record_does_not_cancel_a_timed_one(monkeypatch):
    studio_mod = _studio()

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert studio_mod._pid_is_studio_server(8550, [111.5, None]) is False
    assert studio_mod._pid_is_studio_server(8550, [111.5]) is False
    assert studio_mod._pid_is_studio_server(8550, [111.5, 999.0]) is True
    assert studio_mod._pid_is_studio_server(8550, [None, None]) is True


def test_stop_does_not_signal_a_reused_pid_recorded_in_both_files(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n111.5\n127.0.0.1", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 1, result.output
    assert killed == []
    assert len(list(tmp_path.glob("*.pid"))) == 2


def test_pid_identity_check_trusts_the_record_without_psutil(monkeypatch):
    studio_mod = _studio()
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert studio_mod._pid_is_studio_server(8550) is True


def test_pid_identity_check_uses_the_recorded_start_time(monkeypatch):
    studio_mod = _studio()

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 111.5

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert studio_mod._pid_is_studio_server(8550, [111.5]) is True
    assert studio_mod._pid_is_studio_server(8550, [999.0]) is False


def test_stop_keeps_a_live_record_whose_start_time_no_longer_matches(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n111.5", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 1, result.output
    assert killed == []
    assert (tmp_path / "studio-8901-8550.pid").exists()
    assert "invalid pid file" not in result.output.lower()


def test_stop_reads_the_legacy_single_pid_file(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {4242})
    _write_pid(tmp_path, "studio.pid", 4242)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == [4242]
    assert not (tmp_path / "studio.pid").exists()


def test_stop_reports_nothing_running_without_pid_files(monkeypatch, tmp_path):
    studio_mod, _live, _killed = _install(monkeypatch, tmp_path, alive = set())

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert "no running unsloth server" in result.output.lower()


def test_stop_cleans_stale_pid_files_without_claiming_a_stop(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = set())
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not (tmp_path / "studio-8901-8550.pid").exists()
    assert "stopped" not in result.output.lower()


def test_stop_does_not_claim_a_stop_while_a_server_is_still_alive(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)
    monkeypatch.setattr(studio_mod.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(sys, "platform", "linux")
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert "shutting down" in result.output.lower()
    assert "stopped" not in result.output.lower()
    assert (tmp_path / "studio-8901-8550.pid").exists()


def test_stop_continues_after_one_server_fails_to_stop(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)
    live = {8550, 8600}
    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: pid in live)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)

    def fake_kill(pid, _sig):
        if pid == 8550:
            raise PermissionError("not permitted")
        live.discard(pid)

    monkeypatch.setattr(studio_mod.os, "kill", fake_kill)
    monkeypatch.setattr(sys, "platform", "linux")
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio-8902-8600.pid", 8600)

    result = _run_stop(studio_mod)

    combined = result.output or ""
    assert result.exit_code == 1, combined
    assert 8600 not in live
    assert "8550" in combined


def test_stop_never_signals_pid_zero_or_init(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {0, 1})
    _write_pid(tmp_path, "studio-8901-0.pid", 0)
    _write_pid(tmp_path, "studio-8902-1.pid", 1)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not list(tmp_path.glob("*.pid"))


def test_signal_stop_refuses_pid_zero_or_init(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {0, 1})

    assert studio_mod._signal_stop(0) is not None
    assert studio_mod._signal_stop(1) is not None
    assert killed == []


def test_stop_discards_a_corrupt_pid_file(monkeypatch, tmp_path):
    studio_mod, _live, _killed = _install(monkeypatch, tmp_path, alive = set())
    (tmp_path / "studio-8901-8550.pid").write_text("not-a-pid", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "studio-8901-8550.pid").exists()


def test_stop_keeps_a_record_it_cannot_read(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    path = tmp_path / "studio-8901-8550.pid"
    path.write_text("8550", encoding = "utf-8")
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self == path:
            raise PermissionError(13, "Permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)

    result = _run_stop(studio_mod)

    assert path.exists(), "an unreadable record must not be deleted"
    assert "cannot read" in result.output.lower()


def test_stop_does_not_claim_success_when_the_only_record_is_unreadable(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    path = tmp_path / "studio-8901-8550.pid"
    path.write_text("8550", encoding = "utf-8")
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self == path:
            raise PermissionError(13, "Permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)

    result = _run_stop(studio_mod)

    assert result.exit_code == 1, "an unreachable server is not a successful stop"
    output = result.output
    assert "no running unsloth server" not in output.lower()
    assert killed == []


def test_stop_reports_failure_when_one_record_is_unreadable_but_another_stops(
    monkeypatch, tmp_path
):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550, 8600})
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    hidden = tmp_path / "studio-8902-8600.pid"
    hidden.write_text("8600", encoding = "utf-8")
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self == hidden:
            raise PermissionError(13, "Permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)

    result = _run_stop(studio_mod)

    assert killed == [8550], "the readable server must still be stopped"
    assert result.exit_code == 1
    assert hidden.exists()


def test_stop_reaches_every_server_when_one_record_cannot_be_removed(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8600})
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio-8902-8600.pid", 8600)
    real_unlink = Path.unlink

    def deny(self, *args, **kwargs):
        if self.name == "studio-8901-8550.pid":
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny)

    result = _run_stop(studio_mod)

    assert killed == [8600], "the live server must still be signalled"
    assert result.exit_code == 0, result.output


def test_a_record_whose_pid_is_not_ascii_digits_is_discarded(monkeypatch, tmp_path):
    studio_mod, _live, _killed = _install(monkeypatch, tmp_path, alive = set())
    (tmp_path / "studio-8901-1.pid").write_text("²", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "studio-8901-1.pid").exists()


@pytest.mark.parametrize(
    "body, actual",
    [
        ("8550\n1000.0\n127.0.0.1\nlinux:boot:123\n", 4600.0),
        ("8550\n1000.0\n127.0.0.1", 999.0),
    ],
    ids = ["stable-clock-shift", "legacy-one-second-boundary"],
)
def test_stop_recognizes_a_live_record_after_an_epoch_shift(monkeypatch, tmp_path, body, actual):
    from studio.backend.utils import pid_records

    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)
    monkeypatch.setattr(pid_records, "linux_process_identity", lambda pid: "linux:boot:123")
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process = lambda pid: SimpleNamespace(create_time = lambda: actual)),
    )
    (tmp_path / "studio-8901-8550.pid").write_text(body, encoding = "utf-8")
    result = _run_stop(studio_mod)
    assert result.exit_code == 0, result.output
    assert killed == [8550]


def test_windows_failed_liveness_probe_preserves_the_record(monkeypatch):
    import subprocess

    studio_mod = _studio()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout = "", returncode = 1)
    )
    assert studio_mod._pid_alive(8550) is True


@pytest.mark.skipif(sys.platform != "linux", reason = "needs /proc for the stable identity")
def test_stop_drops_a_record_whose_stable_identity_proves_pid_reuse(monkeypatch, tmp_path):
    import os

    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {os.getpid()})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding = "ascii").strip()
    record = tmp_path / f"studio-8901-{os.getpid()}.pid"
    record.write_text(f"{os.getpid()}\n111.5\n127.0.0.1\nlinux:{boot}:1\n", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not record.exists()


def _install_windows(monkeypatch, tmp_path, *, shutdown_ok, exits):
    """win32 stop with taskkill and the shutdown POST recorded, nothing reaching a real process."""
    studio_mod, live, _killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []

    def fake_run(argv, **_kwargs):
        calls.append(("taskkill", argv))
        live.discard(int(argv[2]))

    def fake_shutdown(pid, paths):
        calls.append(("shutdown", sorted(p.name for p in paths)))
        if shutdown_ok and exits:
            live.discard(8550)
        return 0.0 if shutdown_ok else None

    monkeypatch.setattr(studio_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(studio_mod, "_request_graceful_shutdown", fake_shutdown, raising = False)
    monkeypatch.setattr(studio_mod, "_GRACEFUL_STOP_WAIT_S", 0.0, raising = False)
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    return studio_mod, calls


def test_windows_stop_asks_the_server_to_shut_down_before_any_kill(monkeypatch, tmp_path):
    # taskkill /F skips the SIGTERM path that stops and saves a running training job.
    studio_mod, calls = _install_windows(monkeypatch, tmp_path, shutdown_ok = True, exits = True)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert calls == [("shutdown", ["studio-8901-8550.pid"])]


def test_windows_stop_falls_back_to_taskkill_when_the_server_does_not_exit(monkeypatch, tmp_path):
    studio_mod, calls = _install_windows(monkeypatch, tmp_path, shutdown_ok = True, exits = False)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert calls == [
        ("shutdown", ["studio-8901-8550.pid"]),
        ("taskkill", ["taskkill", "/PID", "8550", "/T", "/F"]),
    ]


def test_graceful_shutdown_waits_past_the_budget_the_server_reports(monkeypatch, tmp_path):
    # The server saves for up to its own UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S; /F before that
    # cuts the save, and stop may run from a shell that never set the variable.
    from unsloth_cli import _inference

    studio_mod = _studio()
    record = tmp_path / "studio-8901-8550.pid"
    record.write_text("8550\n1700000000.0\n127.0.0.1", encoding = "utf-8")
    monkeypatch.delenv("UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S", raising = False)
    reply = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return reply[0]

    monkeypatch.setattr(_inference, "_studio_token", lambda: "tok")
    monkeypatch.setattr(_inference, "urlopen_no_redirect", lambda request, timeout: _Response())
    monkeypatch.setattr(_inference, "verify_studio_identity", lambda base: True)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)

    reply[:] = [b'{"status": "shutting_down", "save_budget_s": 300}']
    assert studio_mod._request_graceful_shutdown(8550, [record]) > 300
    # A server from before the field: this environment's value, else the fixed floor.
    reply[:] = [b'{"status": "shutting_down"}']
    assert studio_mod._request_graceful_shutdown(8550, [record]) == 150.0
    monkeypatch.setenv("UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S", "300")
    assert studio_mod._request_graceful_shutdown(8550, [record]) > 300


def test_windows_stop_kills_when_no_server_answers(monkeypatch, tmp_path):
    studio_mod, calls = _install_windows(monkeypatch, tmp_path, shutdown_ok = False, exits = False)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert [c[0] for c in calls] == ["shutdown", "taskkill"]


def test_graceful_shutdown_sends_the_token_only_to_a_verified_server(monkeypatch, tmp_path):
    from unsloth_cli import _inference

    studio_mod = _studio()
    record = tmp_path / "studio-8901-8550.pid"
    record.write_text("8550\n1700000000.0\n127.0.0.1", encoding = "utf-8")
    posted = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_open(request, timeout):
        posted.append((request.full_url, request.get_method(), request.headers["Authorization"]))
        return _Response()

    monkeypatch.setattr(_inference, "_studio_token", lambda: "tok")
    monkeypatch.setattr(_inference, "urlopen_no_redirect", fake_open)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)

    monkeypatch.setattr(_inference, "verify_studio_identity", lambda base: False)
    assert studio_mod._request_graceful_shutdown(8550, [record]) is None
    assert posted == []

    monkeypatch.setattr(_inference, "verify_studio_identity", lambda base: True)
    assert studio_mod._request_graceful_shutdown(8550, [record]) is not None
    assert posted == [("http://127.0.0.1:8901/api/shutdown", "POST", "Bearer tok")]
    # The legacy studio.pid has no port, so there is nothing to ask.
    assert studio_mod._request_graceful_shutdown(8550, [tmp_path / "studio.pid"]) is None


def test_windows_stop_never_kills_a_pid_reused_after_the_server_exited(monkeypatch, tmp_path):
    # The 150 s wait outlives the identity check stop ran before asking, so it has to hold for the kill too.
    studio_mod, calls = _install_windows(monkeypatch, tmp_path, shutdown_ok = True, exits = False)
    reused = []
    real_shutdown = studio_mod._request_graceful_shutdown

    def shutdown_then_reuse(pid, paths):
        reused.append(True)
        return real_shutdown(pid, paths)

    monkeypatch.setattr(studio_mod, "_request_graceful_shutdown", shutdown_then_reuse)
    monkeypatch.setattr(
        studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): not reused
    )

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert calls == [("shutdown", ["studio-8901-8550.pid"])]


def test_windows_stop_never_kills_a_pid_reused_during_a_failed_probe(monkeypatch, tmp_path):
    # The identity check and POST can take seconds; a server that exits meanwhile frees its PID.
    studio_mod, calls = _install_windows(monkeypatch, tmp_path, shutdown_ok = False, exits = False)
    probed = []
    real_shutdown = studio_mod._request_graceful_shutdown

    def probe_then_reuse(pid, paths):
        probed.append(True)
        return real_shutdown(pid, paths)

    monkeypatch.setattr(studio_mod, "_request_graceful_shutdown", probe_then_reuse)
    monkeypatch.setattr(
        studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): not probed
    )

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert calls == [("shutdown", ["studio-8901-8550.pid"])]


def test_graceful_shutdown_skips_a_stale_record_grouped_under_the_pid(monkeypatch, tmp_path):
    # The stale port may now belong to another Unsloth server, which would answer for this PID.
    from unsloth_cli import _inference

    studio_mod = _studio()
    stale = tmp_path / "studio-8901-8550.pid"
    stale.write_text("8550\n1600000000.0\n127.0.0.1", encoding = "utf-8")
    live = tmp_path / "studio-8902-8550.pid"
    live.write_text("8550\n1700000000.0\n127.0.0.1", encoding = "utf-8")
    posted = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_open(request, timeout):
        posted.append(request.full_url)
        return _Response()

    monkeypatch.setattr(_inference, "_studio_token", lambda: "tok")
    monkeypatch.setattr(_inference, "urlopen_no_redirect", fake_open)
    monkeypatch.setattr(_inference, "verify_studio_identity", lambda base: True)
    monkeypatch.setattr(
        studio_mod,
        "_pid_is_studio_server",
        lambda pid, created_times = (): 1700000000.0 in created_times,
    )

    assert studio_mod._request_graceful_shutdown(8550, [stale, live]) is not None
    assert posted == ["http://127.0.0.1:8902/api/shutdown"]
