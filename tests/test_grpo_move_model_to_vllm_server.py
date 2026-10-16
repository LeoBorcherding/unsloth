"""GRPO on trl's legacy vLLM path: Unsloth rewrote `_move_model_to_vllm` to `return None`, which
is only right when fast_inference shares the weights. In server mode (trl vllm-serve) nothing then
pushed the trained weights, so every rollout came from the step-0 policy. The rewrite is applied to
the installed trl's real method source; vLLM, the model and the client are stubs."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import textwrap
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = ROOT / "unsloth" / "models" / "rl_replacements.py"


def _function(path: Path, name: str, cls: str | None = None) -> str:
    src = path.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next((n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        pytest.skip(f"{name} not in {path.name}")
    lines = src.splitlines(keepends = True)
    # As inspect.getsource hands a method to the rewrite: indented, decorators included.
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def _rewritten_trl_method(tmp: Path) -> str:
    spec = importlib.util.find_spec("trl")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("trl not installed")
    grpo = Path(spec.submodule_search_locations[0]) / "trainer" / "grpo_trainer.py"
    original = _function(grpo, "_move_model_to_vllm", cls = "GRPOTrainer")
    ns = {"re": re, "inspect": inspect}
    rewrite = textwrap.dedent(_function(REPLACEMENTS, "grpo_trainer__move_model_to_vllm"))
    # On a real file, so inspect.getsource inside the rewrite can read its own source back.
    path = tmp / "rewrite.py"
    path.write_text(rewrite, encoding = "utf-8")
    exec(compile(rewrite, str(path), "exec"), ns)
    return ns["grpo_trainer__move_model_to_vllm"]("_move_model_to_vllm", original)


def _trainer(mode: str, shared: bool):
    pushed = []
    client = types.SimpleNamespace(
        update_named_param = lambda name, data: pushed.append(name),
        reset_prefix_cache = lambda: None,
    )
    param = types.SimpleNamespace(data = "tensor")
    trainer = types.SimpleNamespace(
        vllm_mode = mode,
        llm = types.SimpleNamespace(shared_weights = shared, reset_prefix_cache = lambda: None),
        vllm_client = client,
        accelerator = types.SimpleNamespace(
            state = types.SimpleNamespace(deepspeed_plugin = None), is_main_process = True
        ),
        is_fsdp_enabled = False,
        model = types.SimpleNamespace(named_parameters = lambda: [("lm_head.weight", param)]),
        _fix_param_name_to_vllm = lambda name, **_k: name,
    )
    return trainer, pushed


def _run(tmp: Path, mode: str, shared: bool) -> list:
    source = textwrap.dedent(_rewritten_trl_method(tmp))
    ns = {"nullcontext": nullcontext, "is_peft_model": lambda _m: False}
    exec(source, ns)
    trainer, pushed = _trainer(mode, shared)
    ns["_move_model_to_vllm"](trainer)
    return pushed


def test_server_mode_pushes_the_trained_weights(tmp_path):
    assert _run(tmp_path, "server", shared = False) == ["lm_head.weight"]


@pytest.mark.parametrize("mode, shared", [("colocate", True), ("colocate", False), ("server", True)])
def test_shared_weights_and_colocate_still_skip(tmp_path, mode, shared):
    assert _run(tmp_path, mode, shared) == []
