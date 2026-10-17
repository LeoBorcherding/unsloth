# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Everything around the hold loop is stubbed; the switch back writes a marker instead of config.toml.
_CHILD = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
marker = Path(sys.argv[2])
import unsloth_cli.commands.start as start

class Target:
    path = marker.parent / "config.toml"

stubs = {
    "_codex_app_target": lambda: Target(),
    "_open_app_config": lambda target, base, key: (marker.parent / "state.json", {}, "", "k", None),
    "_codex_app_build": lambda text, base, key, entry, previous: ("new", {}),
    "_parse_app_config": lambda path, text: None,
    "_write_app_file": lambda path, text: None,
    "_codex_model_catalog": lambda entry, kind: {},
    "_codex_app_owner": lambda: {},
    "_app_file_mode": lambda path: None,
    "_save_app_config": lambda *args: None,
    "_open_codex_app": lambda: (marker.parent / "ready").write_text("1"),
    "_studio_healthy": lambda base, timeout = 3.0: True,
    "_switch_codex_app_back": lambda path, state: marker.write_text("switched back") or [],
    "_shutdown_auto_served": lambda: None,
}
for name, stub in stubs.items():
    setattr(start, name, stub)
start._codex_app_session("http://127.0.0.1:1", None, {"id": "m"})
"""


@pytest.mark.skipif(os.name != "nt", reason = "Windows console control events")
def test_codex_app_switches_back_on_a_windows_console_event(tmp_path):
    # CTRL_BREAK is the console event a test can send; close, logoff and shutdown come the same way.
    marker = tmp_path / "switched"
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", _CHILD, str(_REPO_ROOT), str(marker)],
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "ready").exists():
            assert child.poll() is None, child.stderr.read().decode(errors = "replace")
            assert time.monotonic() < deadline, "the session never reached the hold loop"
            time.sleep(0.1)
        time.sleep(0.5)
        sent = time.monotonic()
        os.kill(child.pid, signal.CTRL_BREAK_EVENT)
        child.wait(30)
    finally:
        if child.poll() is None:
            child.kill()
    assert marker.exists(), "config.toml was left on Unsloth"
    # A close event gives about 5 s, so the hold loop must not finish its sleep first.
    assert time.monotonic() - sent < 4
