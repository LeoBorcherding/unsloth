# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The trainer patcher guards TRL's vLLM weight pushes on fast_inference instead of deleting them."""

import ast
import importlib.util
import re
import sys
import textwrap
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


RL_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
RL_REPLACEMENTS_PATH = RL_PATH.with_name("rl_replacements.py")


@pytest.fixture(autouse = True)
def _sync_unsafe_module(monkeypatch):
    # The guard imports _unsloth_vllm_sync_unsafe at call time; serve it without importing unsloth.
    text = RL_REPLACEMENTS_PATH.read_text(encoding = "utf-8")
    nodes = [
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_vllm_sync_unsafe"
    ]
    module = types.ModuleType("unsloth.models.rl_replacements")
    code = compile(ast.Module(body = nodes, type_ignores = []), str(RL_REPLACEMENTS_PATH), "exec")
    exec(code, module.__dict__)
    monkeypatch.setitem(sys.modules, "unsloth.models.rl_replacements", module)


def _load_guard():
    text = RL_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text)
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_guard_vllm_weight_loads")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_VLLM_WEIGHT_LOAD_LINE" for t in node.targets)
        )
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(RL_PATH), "exec"), namespace)
    return namespace["_guard_vllm_weight_loads"]


def _trl_rloo_methods():
    spec = importlib.util.find_spec("trl")
    if spec is None:
        pytest.skip("trl is not installed")
    path = Path(spec.origin).parent / "trainer" / "rloo_trainer.py"
    if not path.exists():
        pytest.skip("this trl has no rloo_trainer.py")
    text = path.read_text(encoding = "utf-8")
    cls = next(
        n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "RLOOTrainer"
    )
    methods = {
        n.name: ast.get_source_segment(text, n, padded = True)
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }
    for name in ("_move_model_to_vllm", "_sync_fsdp2_params_to_vllm"):
        if name not in methods:
            pytest.skip(f"this trl's RLOOTrainer has no {name}")
    return methods


# TRL 0.18 to 0.22 OnlineDPOTrainer._generate_vllm loads the weights inline, outside any sync method.
_ONLINE_DPO_GENERATE_VLLM = """
    def _generate_vllm(self, model, prompts):
        # Load the latest weights
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(model.state_dict().items())
        return self.llm.generate(prompts)
"""


def _patched(name, source):
    source = _load_guard()(name, source)
    decorators = [line for line in source.splitlines() if line.strip().startswith("@")]
    for line in decorators:
        source = source.replace(line + "\n", "", 1)
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "nullcontext": nullcontext,
        "is_peft_model": lambda model: hasattr(model, "merge_adapter"),
    }
    exec(textwrap.dedent(source), namespace)
    return namespace[name]


class _Recorder:
    def __init__(self):
        self.loaded = []
        self.resets = 0

    def load_weights(self, weights):
        self.loaded.extend(name for name, _ in weights)

    def reset_prefix_cache(self):
        self.resets += 1


class _SharedEngine:
    # Unsloth's fast_inference engine: no TRL-style driver_worker to load weights into.
    shared_weights = True

    def __init__(self):
        self.resets = 0

    def reset_prefix_cache(self):
        self.resets += 1

    def generate(self, prompts):
        return ["out"] * len(prompts)


class _Param:
    is_cpu = False

    def full_tensor(self):
        return self


class _PeftModel(torch.nn.Linear):
    prefix = "lora_"

    def __init__(self):
        super().__init__(2, 2)
        self.merges = 0

    def merge_adapter(self):
        self.merges += 1

    def unmerge_adapter(self):
        self.merges += 1


def _trainer(llm, model):
    return types.SimpleNamespace(
        accelerator = types.SimpleNamespace(
            state = types.SimpleNamespace(deepspeed_plugin = None), is_main_process = True
        ),
        is_fsdp_enabled = False,
        model = model,
        vllm_mode = "colocate",
        llm = llm,
        _fix_param_name_to_vllm = lambda name, extra_prefixes = None: name,
    )


def _trl_llm():
    rec = _Recorder()
    llm = types.SimpleNamespace(
        llm_engine = types.SimpleNamespace(
            model_executor = types.SimpleNamespace(
                driver_worker = types.SimpleNamespace(model_runner = types.SimpleNamespace(model = rec))
            )
        ),
        reset_prefix_cache = rec.reset_prefix_cache,
    )
    return llm, rec


def test_rloo_fsdp2_sync_keeps_state_dict_and_loads_into_trl_vllm():
    method = _patched("_sync_fsdp2_params_to_vllm", _trl_rloo_methods()["_sync_fsdp2_params_to_vllm"])
    llm, rec = _trl_llm()
    module = types.SimpleNamespace(state_dict = lambda: {"w": _Param(), "b": _Param()})
    method(_trainer(llm, None), module)
    assert rec.loaded == ["w", "b"]


def test_rloo_move_model_pushes_weights_to_trl_vllm():
    method = _patched("_move_model_to_vllm", _trl_rloo_methods()["_move_model_to_vllm"])
    llm, rec = _trl_llm()
    method(_trainer(llm, torch.nn.Linear(2, 2)))
    assert rec.loaded == ["weight", "bias"]
    assert rec.resets == 1


def test_rloo_move_model_is_skipped_when_fast_inference_shares_weights():
    method = _patched("_move_model_to_vllm", _trl_rloo_methods()["_move_model_to_vllm"])
    llm, model = _SharedEngine(), _PeftModel()
    method(_trainer(llm, model))
    assert model.merges == 0
    assert llm.resets == 0


def test_rloo_move_model_keeps_the_no_op_for_lora_on_trl_vllm(capsys):
    method = _patched("_move_model_to_vllm", _trl_rloo_methods()["_move_model_to_vllm"])
    llm, rec = _trl_llm()
    model = _PeftModel()
    model.peft_config = {"default": object()}
    trainer = _trainer(llm, model)
    method(trainer)
    method(trainer)
    assert model.merges == 0
    assert rec.loaded == []
    assert capsys.readouterr().out.count("vLLM weight sync is skipped") == 1


@pytest.mark.parametrize("shared", [False, True])
def test_inline_weight_load_outside_sync_methods(shared):
    method = _patched("_generate_vllm", _ONLINE_DPO_GENERATE_VLLM)
    if shared:
        llm = _SharedEngine()
        assert method(types.SimpleNamespace(llm = llm), torch.nn.Linear(2, 2), ["p"]) == ["out"]
    else:
        llm, rec = _trl_llm()
        llm.generate = lambda prompts: ["out"] * len(prompts)
        method(types.SimpleNamespace(llm = llm), torch.nn.Linear(2, 2), ["p"])
        assert rec.loaded == ["weight", "bias"]


def test_methods_without_vllm_weight_loads_are_untouched():
    source = """
    def save_state(self):
        # load_weights( in a comment stays as it is
        return self.model.state_dict()
"""
    assert _load_guard()("save_state", source) == source
