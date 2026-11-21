# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Windows ROCm: which ``libbitsandbytes_rocm<NN>.dll`` becomes BNB_ROCM_VERSION, at every site
that picks one. Fake DLL folders and a fake ``torch.version.hip``; no torch, no GPU."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SHARED_PATH = _REPO / "studio" / "backend" / "utils" / "bnb_rocm_version.py"
_IMPORT_FIXES_PATH = _REPO / "unsloth" / "import_fixes.py"
_STACK_PATH = _REPO / "studio" / "install_python_stack.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def import_fixes():
    return _load("unsloth_import_fixes_bnb_rank", _IMPORT_FIXES_PATH)


@pytest.fixture(scope = "module")
def stack():
    return _load("studio_install_python_stack_bnb_rank", _STACK_PATH)


@pytest.fixture(scope = "module")
def shared():
    return _load("studio_bnb_rocm_version_under_test", _SHARED_PATH)


def _fake_bnb(tmp_path, monkeypatch, suffixes, hip):
    pkg = tmp_path / "bitsandbytes"
    pkg.mkdir()
    for s in suffixes:
        (pkg / f"libbitsandbytes_rocm{s}.dll").write_text("")
    (pkg / "libbitsandbytes_cpu.dll").write_text("")
    fake_spec = types.SimpleNamespace(submodule_search_locations = [str(pkg)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: fake_spec)
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(version = types.SimpleNamespace(hip = hip))
    )


# (installed suffixes, torch.version.hip, expected pick)
_CASES = [
    # bitsandbytes 0.50.3.dev0 on gr33n (torch 2.11.0+rocm7.14.0): today's pick, kept.
    (["72", "714", "715"], "7.14.60850", "715"),
    (["72", "714"], "7.13.0", "714"),
    (["72", "713"], None, "713"),
    (["72"], "7.2.0", "72"),
    # ROCm 10: "101" is 10.1, not 1.01 below 7.x.
    (["715", "101"], "10.1.0", "101"),
    (["72", "714", "101", "102"], "10.2.0", "102"),
    (["715", "101"], "7.14.0", "715"),
    # Across 6.x/7.x the integer max picked "613" on a 7.x torch.
    (["613", "72"], "7.2.0", "72"),
    # No DLL for torch's major: highest overall, as before.
    (["72", "714"], "10.1.0", "714"),
    ([], "7.14.0", None),
]


@pytest.mark.parametrize("suffixes, hip, expected", _CASES)
def test_shared_pick(shared, suffixes, hip, expected):
    assert shared.pick_bnb_rocm_dll_suffix(suffixes, hip) == expected


@pytest.mark.parametrize("suffixes, hip, expected", _CASES)
def test_import_fixes_pick_matches_shared(import_fixes, shared, suffixes, hip, expected):
    assert import_fixes._pick_bnb_rocm_dll_suffix(suffixes, hip) == expected
    assert import_fixes._pick_bnb_rocm_dll_suffix(suffixes, hip) == shared.pick_bnb_rocm_dll_suffix(
        suffixes, hip
    )


def test_torch_hip_version_read_without_import(shared, tmp_path, monkeypatch):
    pkg = tmp_path / "torch"
    pkg.mkdir()
    (pkg / "version.py").write_text(
        "from typing import Optional\n"
        "__version__ = '2.11.0+rocm7.14.0'\n"
        "cuda: Optional[str] = None\n"
        "hip: Optional[str] = '7.14.60850'\n"
    )
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(submodule_search_locations = [str(pkg)]),
    )
    assert shared.torch_hip_version() == "7.14.60850"
    assert "torch" not in sys.modules


@pytest.mark.parametrize(
    "suffixes, hip, expected",
    [
        (["72", "714", "715"], "7.14.60850", "715"),
        (["715", "101"], "10.1.0", "101"),
    ],
    ids = ["rocm7-today", "rocm10-cross-major"],
)
def test_install_python_stack_detect(stack, tmp_path, monkeypatch, suffixes, hip, expected):
    _fake_bnb(tmp_path, monkeypatch, suffixes, hip)
    assert stack._detect_bnb_rocm_dll_ver() == expected


@pytest.mark.parametrize(
    "suffixes, hip, expected",
    [
        (["72", "714", "715"], "7.14.60850", "715"),
        (["715", "101"], "10.1.0", "101"),
    ],
    ids = ["rocm7-today", "rocm10-cross-major"],
)
def test_import_fixes_detect(import_fixes, tmp_path, monkeypatch, suffixes, hip, expected):
    _fake_bnb(tmp_path, monkeypatch, suffixes, hip)
    assert import_fixes._detect_installed_bnb_rocm_version() == expected


@pytest.mark.parametrize(
    "rel", ["studio/backend/main.py", "studio/backend/core/training/worker.py"]
)
def test_backend_sites_rank_by_version(rel):
    # These pick inline at startup, so check they route through the shared helper.
    src = (_REPO / rel).read_text(encoding = "utf-8")
    assert "pick_bnb_rocm_dll_suffix(" in src
    assert "key = lambda v: int(v)" not in src
