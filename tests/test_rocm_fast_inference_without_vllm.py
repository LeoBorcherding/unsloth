# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastLlamaModel keeps fast_inference off on ROCm once it has decided vLLM is unavailable."""

import ast
import textwrap
import types
from pathlib import Path

import pytest

LLAMA = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "llama.py"


def _torch(capability = (9, 0)):
    return types.SimpleNamespace(
        cuda = types.SimpleNamespace(get_device_capability = lambda: capability)
    )


def _fast_inference_block():
    lines = LLAMA.read_text(encoding = "utf-8").splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == "if fast_inference:" and "is_vLLM_available" in lines[i + 1]
    )
    end = next(i for i in range(start, len(lines)) if "if unsloth_vllm_standby" in lines[i])
    return textwrap.dedent("\n".join(lines[start:end]))


def _run_block(device_type, vllm_available):
    namespace = {
        "fast_inference": True,
        "is_vLLM_available": lambda: vllm_available,
        "DEVICE_TYPE": device_type,
        "torch": _torch(),
    }
    exec(_fast_inference_block(), namespace)
    return namespace["fast_inference"]


def _will_load_weights(device_type, vllm_available):
    tree = ast.parse(LLAMA.read_text(encoding = "utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vllm_will_load_weights"
    )
    namespace = {
        "DEVICE_TYPE": device_type,
        "is_vLLM_available": lambda: vllm_available,
        "torch": _torch(),
    }
    exec(compile(ast.Module(body = [node], type_ignores = []), str(LLAMA), "exec"), namespace)
    return namespace["_vllm_will_load_weights"]


@pytest.mark.parametrize("device_type", ["hip", "cuda", "xpu"])
def test_missing_vllm_turns_fast_inference_off(device_type, capsys):
    assert _run_block(device_type, vllm_available = False) is False
    assert "Will use Unsloth inference" in capsys.readouterr().out


@pytest.mark.parametrize("device_type", ["hip", "cuda"])
def test_installed_vllm_keeps_fast_inference(device_type):
    assert _run_block(device_type, vllm_available = True) is True


@pytest.mark.parametrize("device_type", ["hip", "cuda"])
def test_revision_is_kept_when_vllm_is_missing(device_type):
    assert _will_load_weights(device_type, vllm_available = False)(True) is False


@pytest.mark.parametrize("device_type", ["hip", "cuda"])
def test_vllm_owns_the_load_when_installed(device_type):
    will_load = _will_load_weights(device_type, vllm_available = True)
    assert will_load(True) is True
    assert will_load(True, num_labels = 2) is False
