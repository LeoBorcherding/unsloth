# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The backend must not put the venv's unsigned hipInfo.exe on PATH (Smart App Control blocks it)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FILES = [_BACKEND / "main.py", _BACKEND / "core" / "training" / "worker.py"]


def _path_writes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and ast.unparse(t) == "os.environ['PATH']":
                    yield node


def _cuda_specs_filters(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addFilter"
            and "bitsandbytes.cuda_specs" in ast.unparse(node.func.value)
        ):
            yield eval(compile(ast.Expression(node.args[0]), "<filter>", "eval"))


@pytest.mark.parametrize("path", _FILES, ids = lambda p: p.name)
def test_no_hipinfo_path_injection(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    assert not list(_path_writes(tree))


@pytest.mark.parametrize("path", _FILES, ids = lambda p: p.name)
def test_arch_probe_log_dropped_others_kept(path):
    filters = list(_cuda_specs_filters(ast.parse(path.read_text(encoding = "utf-8"))))
    assert len(filters) == 1

    def record(func):
        r = logging.LogRecord("bitsandbytes.cuda_specs", logging.ERROR, "x", 1, "m", None, None)
        r.funcName = func
        return r

    assert not filters[0](record("get_rocm_gpu_arch"))
    assert filters[0](record("get_cuda_specs"))
