# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""A ModelOpt NVFP4 config with no ignore list takes its quantized set from hf_quant_config.json
`parameter_groups` (thinkingmachines/Inkling-2-Mini), so 16bit Linears are not loaded as NVFP4."""

import json
import re

import pytest

transformers = pytest.importorskip("transformers")

from unsloth.models.modelopt_nvfp4 import modelopt_nvfp4_plan


def _group(transform, tensors):
    return {"weight_transform": transform, "tensors": tensors}


def _config(tmp_path, groups):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"model": {"parameter_groups": groups}, "quant_method": "inkling2"})
    )
    config = transformers.PretrainedConfig(
        quantization_config = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    )
    config._name_or_path = str(tmp_path)
    return config


def _ignored(plan, name):
    for entry in plan["ignore"]:
        if entry == name or (entry.startswith("re:") and re.match(entry[3:], name)):
            return True
    return False


def test_experts_only_nvfp4_keeps_every_linear_16bit(tmp_path):
    plan = modelopt_nvfp4_plan(
        _config(
            tmp_path,
            [
                _group("cast", ["model.layers.0.self_attn.q_proj.linear.weight", "lm_head.weight"]),
                _group("nvfp4_nearest", ["model.layers.0.mlp.experts.w13_weight"]),
                _group("mxfp8_nearest", ["model.layers.1.mlp.experts.w2_weight"]),
            ],
        )
    )
    assert plan is not None
    for name in ("model.layers.0.self_attn.q_proj.linear", "lm_head", "model.visual.proj_in"):
        assert _ignored(plan, name)


def test_quantized_linears_stay_targeted(tmp_path):
    plan = modelopt_nvfp4_plan(
        _config(
            tmp_path,
            [
                _group("cast", ["model.layers.0.self_attn.o_proj.weight", "lm_head.weight"]),
                _group("nvfp4_nearest", ["model.layers.0.self_attn.q_proj.weight"]),
            ],
        )
    )
    assert _ignored(plan, "model.layers.0.self_attn.o_proj")
    assert _ignored(plan, "lm_head")
    assert not _ignored(plan, "model.layers.0.self_attn.q_proj")


def test_no_hf_quant_config_keeps_the_old_plan(tmp_path):
    config = transformers.PretrainedConfig(
        quantization_config = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    )
    config._name_or_path = str(tmp_path)
    assert modelopt_nvfp4_plan(config)["ignore"] == []
