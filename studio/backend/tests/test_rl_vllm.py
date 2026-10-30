# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import dataclasses
import importlib.util
import sys
import types

import pytest
from pydantic import ValidationError

import core.training.rl as rl
from core.training.rl import (
    build_rl_trainer,
    vllm_fallback_message,
    vllm_load_kwargs,
    vllm_rollout_blocker,
)
from core.training.worker import _grpo_vllm_load_kwargs
from models.training import TrainingStartRequest

BASE = {"model_name": "unsloth/Qwen2.5-3B-Instruct", "training_type": "LoRA/QLoRA", "format_type": "auto"}
REWARD = {"name": "r", "rule": {"type": "length", "max_chars": 9, "score": {"over": -1.0, "under": 0.0}}}


def _fake_torch(*, cuda = True, hip = None, capability = (8, 0)):
    return types.SimpleNamespace(
        version = types.SimpleNamespace(hip = hip),
        cuda = types.SimpleNamespace(
            is_available = lambda: cuda, get_device_capability = lambda *a: capability
        ),
    )


@pytest.fixture
def host(monkeypatch):
    """Pin the runtime probe to a described host instead of asking this machine."""

    def set_host(platform = "linux", vllm = True, disabled = None, **torch_kw):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(**torch_kw))
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: (object() if vllm else None) if name == "vllm" else real(name, *a),
        )
        monkeypatch.setitem(
            sys.modules, "unsloth.import_fixes", types.SimpleNamespace(VLLM_DISABLED_REASON = disabled)
        )
        return rl._vllm_runtime_blocker()

    return set_host


def test_linux_cuda_with_vllm_runs_vllm(host):
    assert host() is None


def test_linux_rocm_with_a_vllm_build_runs_vllm(host):
    assert host(hip = "7.1") is None


@pytest.mark.parametrize(
    "described, expected",
    [
        ({"platform": "win32"}, "Linux only"),
        ({"platform": "darwin"}, "Linux only"),
        ({"cuda": False}, "CUDA or ROCm GPU"),
        ({"capability": (6, 1)}, "compute capability 7.0"),
        ({"vllm": False}, "vLLM is not installed"),
        ({"vllm": False, "hip": "7.1"}, "no vLLM build"),
        ({"disabled": "needs transformers 5"}, "disabled (needs transformers 5)"),
    ],
)
def test_runtime_blockers_name_the_reason(host, described, expected):
    assert expected in host(**described)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"full_finetuning": True}, "LoRA"),
        ({"vision": True}, "text-only"),
        ({"audio": True}, "not audio"),
        ({"diffusion": True}, "text-diffusion"),
        ({"offload_layers": "auto"}, "offloading"),
        ({"gpu_count": 2}, "single GPU"),
    ],
)
def test_run_shapes_vllm_cannot_take_never_probe_the_runtime(kwargs, expected):
    def runtime():
        raise AssertionError("probed the runtime")

    assert expected in vllm_rollout_blocker(runtime = runtime, **kwargs)


def test_a_supported_run_defers_to_the_runtime():
    assert vllm_rollout_blocker(runtime = lambda: None) is None
    assert vllm_rollout_blocker(runtime = lambda: "vLLM runs on Linux only") == "vLLM runs on Linux only"


def test_fallback_message_says_what_generates_instead():
    message = vllm_fallback_message("vLLM runs on Linux only")
    assert message.startswith("vLLM rollouts are off: vLLM runs on Linux only.")
    assert "Hugging Face generate" in message


def test_load_kwargs_follow_the_notebooks():
    assert vllm_load_kwargs({}, 64) == {
        "fast_inference": True,
        "gpu_memory_utilization": 0.5,
        "max_lora_rank": 64,
    }
    # vLLM rejects an adapter above max_lora_rank at the first sync, so never size it below the run's rank.
    assert vllm_load_kwargs({"vllm_gpu_memory_utilization": 0.7}, 8)["max_lora_rank"] == 16
    assert vllm_load_kwargs({"vllm_gpu_memory_utilization": 0.7}, 8)["gpu_memory_utilization"] == 0.7


@pytest.mark.parametrize(
    "config, wanted",
    [
        ({"objective": "sft", "rl_settings": {"use_vllm": True}}, False),
        ({"objective": "dpo", "rl_settings": {"use_vllm": True}}, False),
        ({"objective": "grpo", "rl_settings": {}}, False),
        ({"objective": "grpo"}, False),
        ({"objective": "grpo", "rl_settings": {"use_vllm": True}, "lora_r": 32}, True),
    ],
)
def test_worker_asks_for_vllm_only_for_an_opted_in_grpo_run(config, wanted):
    kwargs = _grpo_vllm_load_kwargs(config)
    assert bool(kwargs) == wanted
    if wanted:
        assert kwargs["fast_inference"] is True and kwargs["max_lora_rank"] == 32


@dataclasses.dataclass
class _Config:
    output_dir: str = "out"
    use_vllm: bool = False
    beta: float = 0.0
    temperature: float = 1.0
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 256
    reward_weights: list = None
    log_completions: bool = False
    loss_type: str = "dapo"
    importance_sampling_level: str = "token"
    mask_truncated_completions: bool = False
    epsilon: float = 0.2
    epsilon_high: float = None
    generation_kwargs: dict = None


@pytest.mark.parametrize("engine", [True, False])
def test_grpo_config_uses_vllm_exactly_when_the_model_has_an_engine(monkeypatch, engine):
    # fp16-only card: the HF-generate logits guard must not reach vLLM's SamplingParams.
    monkeypatch.setattr(rl, "_fp16_only_gpu", lambda: True)
    monkeypatch.setitem(
        sys.modules, "trl", types.SimpleNamespace(GRPOConfig = _Config, GRPOTrainer = lambda **kw: kw)
    )
    model = types.SimpleNamespace(vllm_engine = object()) if engine else types.SimpleNamespace()
    kw = build_rl_trainer(
        "grpo",
        model = model,
        tokenizer = None,
        train_dataset = None,
        eval_dataset = None,
        config_args = {"output_dir": "o", "max_seq_length": 1024},
        settings = {"use_vllm": True},
        reward_specs = [REWARD],
    )
    assert kw["args"].use_vllm is engine
    assert kw["args"].generation_kwargs == (None if engine else {"remove_invalid_values": True})


def test_request_keeps_vllm_off_unless_asked():
    req = TrainingStartRequest(**BASE, objective = "grpo", grpo_rewards = [{"name": "r"}])
    assert req.grpo_use_vllm is False and req.grpo_vllm_gpu_memory_utilization == 0.5


@pytest.mark.parametrize("fraction", [0.05, 0.99])
def test_request_refuses_a_memory_fraction_that_leaves_nothing_for_one_side(fraction):
    with pytest.raises(ValidationError):
        TrainingStartRequest(
            **BASE,
            objective = "grpo",
            grpo_rewards = [{"name": "r"}],
            grpo_vllm_gpu_memory_utilization = fraction,
        )


def test_resume_lifts_the_stored_vllm_settings():
    req = TrainingStartRequest(
        **BASE,
        objective = "grpo",
        rl_settings = {"use_vllm": True, "vllm_gpu_memory_utilization": 0.6},
        reward_specs = [{"name": "r", "weight": 1.0}],
    )
    assert req.grpo_use_vllm is True and req.grpo_vllm_gpu_memory_utilization == 0.6


@pytest.mark.parametrize(
    "shape, expected",
    [
        ({"diffusion_lm_profile": "dream"}, "text-diffusion"),
        ({"_audio_type": "snac"}, "not audio"),
        ({"is_audio_vlm": True}, "not audio"),
        ({"model_has_vision": True}, "text-only"),
    ],
)
def test_loader_warns_for_models_whose_branch_skips_vllm(monkeypatch, shape, expected):
    # These load through their own branch, so the warning has to come from the shared check.
    from core.training.trainer import UnslothTrainer

    monkeypatch.setattr(rl, "_vllm_runtime_blocker", lambda: None)
    warnings = []
    trainer = types.SimpleNamespace(
        **{
            "model_has_vision": False,
            "diffusion_lm_profile": None,
            "_audio_type": None,
            "is_audio_vlm": False,
            "_offload_layers": 0,
            **shape,
        },
        _record_warning = warnings.append,
    )
    kwargs = UnslothTrainer._vllm_kwargs_or_warn(trainer, {"fast_inference": True}, False, 1)
    assert kwargs == {} and len(warnings) == 1 and expected in warnings[0]


def test_loader_passes_vllm_kwargs_for_a_plain_text_model(monkeypatch):
    from core.training.trainer import UnslothTrainer

    monkeypatch.setattr(rl, "_vllm_runtime_blocker", lambda: None)
    trainer = types.SimpleNamespace(
        model_has_vision = False,
        diffusion_lm_profile = None,
        _audio_type = None,
        is_audio_vlm = False,
        _offload_layers = 0,
        _record_warning = lambda m: pytest.fail(m),
    )
    requested = {"fast_inference": True, "max_lora_rank": 64}
    assert UnslothTrainer._vllm_kwargs_or_warn(trainer, requested, False, 1) == requested
    assert UnslothTrainer._vllm_kwargs_or_warn(trainer, None, False, 1) == {}


@pytest.mark.parametrize(
    "requested, engine, warned",
    [(True, True, False), (True, False, True), (False, False, False)],
)
def test_a_load_that_comes_back_without_an_engine_warns(requested, engine, warned):
    # Unsloth turns fast_inference off for some models and GPUs and returns a plain model.
    from core.training.trainer import UnslothTrainer

    warnings = []
    model = types.SimpleNamespace(vllm_engine = object()) if engine else types.SimpleNamespace()
    trainer = types.SimpleNamespace(model = model, _record_warning = warnings.append)
    UnslothTrainer._check_vllm_engine(trainer, requested)
    assert bool(warnings) == warned
    if warned:
        assert warnings[0].startswith("vLLM rollouts are off: Unsloth loaded this model without vLLM")
