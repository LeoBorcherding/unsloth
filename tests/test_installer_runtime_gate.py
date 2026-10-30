# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install.sh holds the runtime gate Unsloth Desktop and the CLI honour, so a Desktop opened
mid-install waits instead of repairing over the same venv."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

# POSIX only; an import at the top would fail collection on Windows before the skip below applies.
fcntl = pytest.importorskip("fcntl")

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
GATE_PY = REPO_ROOT / "unsloth_cli" / "_studio_runtime_gate.py"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "install.sh is POSIX only",
)


def _gate_module():
    spec = importlib.util.spec_from_file_location("_studio_runtime_gate_under_test", GATE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate_functions() -> str:
    source = INSTALL_SH.read_text(encoding = "utf-8")
    blocks = []
    for name in ("_take_studio_runtime_gate", "_release_studio_runtime_gate"):
        match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", source, re.S | re.M)
        assert match, f"{name} is missing from install.sh"
        blocks.append(match.group(0))
    return "_STUDIO_RUNTIME_GATE_OPEN=false\n" + "".join(blocks)


def _harness(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "harness.sh"
    script.write_text(
        "set -e\n"
        'step() { printf "%s: %s\\n" "$1" "$2"; }\n'
        'substep() { printf "%s\\n" "$1"; }\n'
        "tauri_log() { :; }\n"
        'C_ERR=""\n'
        f"STUDIO_HOME='{tmp_path / 'home'}'\n" + _gate_functions() + body
    )
    (tmp_path / "home").mkdir(exist_ok = True)
    return script


def _env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF"}
    env.update(extra)
    return env


def _hold(home: Path):
    handle = (home / ".studio-runtime.lock").open("a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)


def test_the_installer_holds_the_gate_the_cli_and_desktop_take(tmp_path):
    gate = _gate_module()
    home = tmp_path / "home"
    taken, released = tmp_path / "taken", tmp_path / "released"
    script = _harness(
        tmp_path,
        "_take_studio_runtime_gate\n"
        f"touch '{taken}'\n"
        f"while [ ! -e '{tmp_path / 'go'}' ]; do sleep 0.05; done\n"
        "_release_studio_runtime_gate\n"
        f"touch '{released}'\n"
        "sleep 30\n",
    )
    proc = subprocess.Popen(["sh", str(script)], env = _env())
    try:
        _wait_for(taken)
        # What a Desktop launch, probe or repair (and `unsloth studio update`) would hit mid-install.
        with pytest.raises(gate.StudioRuntimeGateBusy):
            with gate.studio_runtime_launch_guard(home):
                pass
        (tmp_path / "go").touch()
        _wait_for(released)
        # Released before the launch/service step, while the installer itself is still running.
        with gate.studio_runtime_launch_guard(home) as acquired:
            assert acquired
    finally:
        proc.kill()
        proc.wait()


def test_a_second_installer_stops_before_touching_the_venv(tmp_path):
    script = _harness(tmp_path, "_take_studio_runtime_gate\necho moved-the-venv\n")
    holder = _hold(tmp_path / "home")
    try:
        result = subprocess.run(["sh", str(script)], capture_output = True, text = True, env = _env())
    finally:
        holder.close()
    assert result.returncode == 1
    assert "already running" in result.stdout
    assert "moved-the-venv" not in result.stdout


@pytest.mark.skipif(shutil.which("perl") is None, reason = "needs perl")
def test_without_flock_the_perl_fallback_still_sees_the_holder(tmp_path):
    # macOS ships perl but no flock(1).
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in ("sh", "touch", "perl", "printf"):
        found = shutil.which(tool)
        if found:
            (tools / tool).symlink_to(found)
    script = _harness(tmp_path, "_take_studio_runtime_gate\necho moved-the-venv\n")
    holder = _hold(tmp_path / "home")
    try:
        busy = subprocess.run(
            [str(tools / "sh"), str(script)],
            capture_output = True,
            text = True,
            env = _env(PATH = str(tools)),
        )
    finally:
        holder.close()
    assert busy.returncode == 1 and "already running" in busy.stdout

    free = subprocess.run(
        [str(tools / "sh"), str(script)], capture_output = True, text = True, env = _env(PATH = str(tools))
    )
    assert free.returncode == 0 and "moved-the-venv" in free.stdout


def test_a_handed_down_gate_is_not_taken_again(tmp_path):
    script = _harness(tmp_path, "_take_studio_runtime_gate\necho moved-the-venv\n")
    holder = _hold(tmp_path / "home")
    try:
        result = subprocess.run(
            ["sh", str(script)],
            capture_output = True,
            text = True,
            env = _env(_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = "1"),
        )
    finally:
        holder.close()
    assert result.returncode == 0 and "moved-the-venv" in result.stdout


def test_the_gate_spans_the_venv_replacement_and_setup_but_not_the_launch():
    source = INSTALL_SH.read_text(encoding = "utf-8")

    def at(needle: str) -> int:
        index = source.find(needle)
        assert index != -1, needle
        return index

    take = at('mkdir -p "$STUDIO_HOME"\n_take_studio_runtime_gate\n')
    release = at("\n_release_studio_runtime_gate\n")
    assert take < at('if ! _start_studio_venv_replacement "$VENV_DIR"; then')
    assert at('bash "$SETUP_SH"') < at('exit "$_SETUP_EXIT"') < release
    assert release < at('if [ "$TAURI_MODE" = true ]; then\n    tauri_log "DONE"')
    assert release < at("\n    _install_systemd_user_service\n")
    assert release < at('exec "$VENV_DIR/bin/unsloth" studio -p 8888')
