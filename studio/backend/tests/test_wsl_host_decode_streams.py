# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""wsl.exe's own warnings are UTF-16LE on stderr while the guest writes UTF-8 on stdout: one
must not decide how the other is read. subprocess.run is stubbed; no wsl.exe runs."""

import subprocess

import pytest

from core.inference import wsl_host

GUEST = "0, GPU-1111aaaa-2222-3333-4444-555566667777\n"
WARNING = "wsl: A localhost proxy configuration was detected but not mirrored into WSL.\r\n"


@pytest.fixture
def wsl_returns(monkeypatch):
    def install(code):
        result = subprocess.CompletedProcess(
            [], code, stdout = GUEST.encode("utf-8"), stderr = WARNING.encode("utf-16-le")
        )
        monkeypatch.setattr(wsl_host, "wsl_exe", lambda: "wsl.exe")
        monkeypatch.setattr(wsl_host.subprocess, "run", lambda *_a, **_k: result)

    return install


def test_a_wsl_warning_does_not_garble_the_guest_output(wsl_returns):
    wsl_returns(0)
    code, output = wsl_host.run(["-d", "x", "--exec", "nvidia-smi"])
    assert code == 0
    assert output == GUEST


def test_a_failure_still_reports_both_streams_readably(wsl_returns):
    wsl_returns(1)
    _code, output = wsl_host.run(["--status"])
    assert GUEST in output and "localhost proxy" in output


def test_guest_gpu_indices_find_the_card_past_a_warning(wsl_returns, monkeypatch):
    wsl_returns(0)
    monkeypatch.setattr(wsl_host, "gpu_uuids", lambda _ids: ["GPU-1111aaaa-2222-3333-4444-555566667777"])
    monkeypatch.setattr(wsl_host, "guest_command", lambda argv, env = None: (["wsl.exe", *argv], None))
    assert wsl_host.guest_gpu_indices([0]) == [0]
