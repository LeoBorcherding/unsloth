# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A llama-server that loses its port before binding is respawned on a fresh one.

``_find_free_port`` releases the port it picked, and nothing holds it through the
GPU fit and the spawn. When another process takes it, the child exits with
"couldn't bind HTTP server socket", and every retry used to reuse the same busy
``self._port``. No real llama-server is started: Popen is stubbed.
"""

from __future__ import annotations

import itertools
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_REAL_POPEN = subprocess.Popen
_BIND_FAILED = [
    "main: loading model\n",
    "srv    start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 50001\n",
]
_OTHER_CRASH = ["llama_model_load: error loading model: unexpected end of file\n"]


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _port_of(cmd) -> int:
    cmd = [str(c) for c in cmd]
    return int(cmd[cmd.index("--port") + 1])


class _Harness:
    """CPU-only backend; each llama-server spawn exits with the next queued output."""

    def __init__(self, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_text("x")
        b = LlamaCppBackend()
        b._get_gpu_memory = lambda _binary = None, **kw: []
        b._get_gpu_free_memory = lambda _binary = None, **kw: []
        b._read_gguf_metadata = lambda _path: None
        b._can_estimate_kv = lambda: False
        b._get_gguf_size_bytes = lambda _path: 1024
        b._mmproj_vram_bytes = lambda _path: 0
        b._resolve_launch_mmproj_path = lambda **kwargs: None
        b._apu_ram_shortfall_message = lambda *args, **kwargs: None
        b._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
        b._find_llama_server_binary = lambda include_denied = False: str(binary)
        b._is_vulkan_backend = lambda _binary = None: False
        b._detect_audio_type_strict = lambda: None
        b._apply_detected_audio = lambda _detected: True
        ports = itertools.count(50001)
        b._find_free_port = lambda: next(ports)

        def crashed(timeout, **_kw):
            if b._stdout_thread is not None:
                b._stdout_thread.join(timeout = 2)
            return False

        b._wait_for_health = crashed
        self.backend = b
        self.gguf = _write_gguf(tmp_path / "model.gguf")
        self.outputs: list = []
        self.ports: list[int] = []

    def load(self) -> None:
        def fake_popen(cmd, **kwargs):
            if not cmd or "--port" not in [str(c) for c in cmd]:
                return _REAL_POPEN(cmd, **kwargs)
            self.ports.append(_port_of(cmd))
            lines = self.outputs.pop(0) if self.outputs else _OTHER_CRASH
            return type(
                "Process",
                (),
                {
                    "pid": 123,
                    "returncode": 1,
                    "stdout": iter(lines),
                    "poll": lambda self: 1,
                    "terminate": lambda self: None,
                    "wait": lambda self, timeout = None: 1,
                    "kill": lambda self: None,
                },
            )()

        with (
            patch.object(subprocess, "Popen", side_effect = fake_popen),
            pytest.raises(RuntimeError),
        ):
            self.backend.load_model(
                GgufLoadIntent(gguf_path = str(self.gguf), model_identifier = "t")
            )


@pytest.fixture
def harness(tmp_path):
    LlamaCppBackend._sched_reserve_abort_keys.clear()
    yield _Harness(tmp_path)
    LlamaCppBackend._sched_reserve_abort_keys.clear()


def test_a_lost_port_is_retried_on_a_fresh_one(harness):
    harness.outputs = [_BIND_FAILED]
    harness.load()
    assert len(harness.ports) >= 2, harness.ports
    assert harness.ports[0] == 50001
    assert harness.ports[1] != 50001, harness.ports
    # every later fallback spawn follows the new port, and health polls the same one
    assert 50001 not in harness.ports[1:], harness.ports
    assert harness.backend._port == harness.ports[-1]


def test_an_ordinary_crash_keeps_its_port(harness):
    harness.outputs = [_OTHER_CRASH]
    harness.load()
    assert set(harness.ports) == {50001}, harness.ports


def test_with_port_rewrites_only_unsloths_own_flag():
    cmd = ["llama-server", "-m", "m.gguf", "--port", "1", "--parallel", "1"]
    assert LlamaCppBackend._with_port(cmd, 7) == [
        "llama-server", "-m", "m.gguf", "--port", "7", "--parallel", "1",
    ]
    assert cmd[4] == "1"
    assert LlamaCppBackend._with_port(["x"], 7) == ["x"]
