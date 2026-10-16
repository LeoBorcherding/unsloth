"""The CUDA block of `unsloth/_gpu_init.py`, run with `os`, `torch` and `bitsandbytes` stubbed:
a missing bitsandbytes is not a CUDA linking failure, so a root import must not run ldconfig or
warn, and the /usr/local/cuda-* pick must read versions like 12.4.1. Nothing touches the host."""

from __future__ import annotations

import ast
import re
import sys
import textwrap
import types
from pathlib import Path

import pytest
from packaging.version import Version

GPU_INIT = Path(__file__).resolve().parents[2] / "unsloth" / "_gpu_init.py"


def _cuda_block() -> str:
    src = GPU_INIT.read_text(encoding = "utf-8")
    for node in ast.parse(src).body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "cuda"
        ):
            # Only the cuda arm, not the elif chain behind it.
            lines = src.splitlines()[node.body[0].lineno - 1 : node.body[-1].end_lineno]
            return textwrap.dedent("\n".join(lines))
    raise AssertionError("no `if DEVICE_TYPE == \"cuda\":` block in _gpu_init.py")


def _run(monkeypatch, *, bnb_lib, listing = "", lib64_nvidia = False):
    calls, warned = [], []
    fake_os = types.SimpleNamespace(
        geteuid = lambda: 0,
        system = calls.append,
        path = types.SimpleNamespace(
            exists = lambda p: p == "/usr/local" or (lib64_nvidia and p == "/usr/lib64-nvidia")
        ),
    )
    if bnb_lib is None:
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)
        monkeypatch.setitem(sys.modules, "bitsandbytes.functional", None)
    else:
        functional = types.ModuleType("bitsandbytes.functional")
        functional.lib = bnb_lib
        bnb = types.ModuleType("bitsandbytes")
        bnb.functional = functional
        monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
        monkeypatch.setitem(sys.modules, "bitsandbytes.functional", functional)
    namespace = {
        "os": fake_os,
        "torch": types.SimpleNamespace(cuda = types.SimpleNamespace(is_available = lambda: True)),
        "triton": None,
        "Version": Version,
        "re": re,
        "importlib": types.SimpleNamespace(reload = lambda m: m),
        "subprocess": types.SimpleNamespace(check_output = lambda *a, **k: listing.encode()),
        "warnings": types.SimpleNamespace(warn = lambda msg, *a, **k: warned.append(str(msg))),
        "np": pytest.importorskip("numpy"),
    }
    exec(compile(_cuda_block(), str(GPU_INIT), "exec"), namespace)
    return calls, warned


def test_root_import_without_bitsandbytes_leaves_the_linker_cache_alone(monkeypatch):
    calls, warned = _run(monkeypatch, bnb_lib = None, lib64_nvidia = True)
    assert calls == []
    assert not any("not linked properly" in w for w in warned)


def test_the_newest_local_cuda_is_picked_by_version_parts(monkeypatch):
    listing = "\n".join(
        f"drwxr-xr-x 1 root root 4096 Jan 1 00:00 {name}"
        for name in ("cuda-12.4.1", "cuda-12.10", "cuda-12.9")
    )
    # bitsandbytes present but its CUDA library unresolved: the real linking-failure path.
    calls, _warned = _run(monkeypatch, bnb_lib = types.SimpleNamespace(), listing = listing)
    assert calls == ["ldconfig /usr/local/cuda-12.10"]
