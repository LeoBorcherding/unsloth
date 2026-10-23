# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
"""A mamba-ssm kernel whose `causal_conv1d_cuda` import failed must not switch on the fused path."""

import sys
import types

import pytest

transformers = pytest.importorskip("transformers")
hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
from unsloth.import_fixes import fix_transformers_mamba_ssm_without_causal_conv1d_cuda


def _fake_kernels(monkeypatch, conv_cuda):
    # The shape of kernels-community/mamba-ssm (and mamba-ssm 2.x) after its guarded import ran.
    root = "fake_hub_mamba_ssm"
    mods = {}
    for name in (root, f"{root}.ops", f"{root}.ops.triton"):
        mods[name] = types.ModuleType(name)
        mods[name].__path__ = []
    for sub in ("ops.triton.ssd_combined", "ops.selective_scan_interface"):
        mods[f"{root}.{sub}"] = types.ModuleType(f"{root}.{sub}")
        mods[f"{root}.{sub}"].causal_conv1d_cuda = conv_cuda
    fn = lambda *a, **k: None
    for attr in (
        "selective_state_update",
        "mamba_chunk_scan_combined",
        "mamba_split_conv1d_scan_combined",
    ):
        setattr(mods[f"{root}.ops.triton.ssd_combined"], attr, fn)
    mods[f"{root}.ops.triton.selective_state_update"] = types.ModuleType(
        f"{root}.ops.triton.selective_state_update"
    )
    mods[f"{root}.ops.triton.selective_state_update"].selective_state_update = fn
    mods[root].ops = mods[f"{root}.ops"]
    mods[f"{root}.ops"].triton = mods[f"{root}.ops.triton"]
    mods[f"{root}.ops"].selective_scan_interface = mods[f"{root}.ops.selective_scan_interface"]
    mods[f"{root}.ops.triton"].ssd_combined = mods[f"{root}.ops.triton.ssd_combined"]
    mods[f"{root}.ops.triton"].selective_state_update = mods[
        f"{root}.ops.triton.selective_state_update"
    ]
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    conv = types.ModuleType("fake_hub_causal_conv1d")
    conv.causal_conv1d_fn = conv.causal_conv1d_update = fn
    mapping = hub_kernels._KERNEL_MODULE_MAPPING
    monkeypatch.setitem(mapping, "mamba-ssm", mods[root])
    monkeypatch.setitem(mapping, "causal-conv1d", conv)
    return mods[root]


def _nemotron_h_fast_path():
    try:
        from transformers.models.nemotron_h import modeling_nemotron_h as m
        from transformers import NemotronHConfig
    except Exception:
        pytest.skip("no Nemotron-H on this transformers")
    if not hasattr(m, "is_fast_path_available"):
        pytest.skip("this transformers picks the Mamba kernels without lazy_load_kernel")
    config = NemotronHConfig(
        hidden_size = 64,
        mamba_num_heads = 4,
        mamba_head_dim = 16,
        ssm_state_size = 8,
        n_groups = 1,
        num_hidden_layers = 1,
        vocab_size = 32,
    )
    m.NemotronHMamba2Mixer(config, layer_idx = 0)
    return m.is_fast_path_available


@pytest.fixture
def restore_lazy_load_kernel(monkeypatch):
    # `import unsloth` (conftest) may have applied the fix already: start every test from the original.
    from transformers import integrations

    original = getattr(hub_kernels, "lazy_load_kernel", None)
    if original is None:
        pytest.skip("no lazy_load_kernel on this transformers")
    if getattr(original, "_unsloth_mamba_ssm_guard", False):
        original = original.__wrapped__
    monkeypatch.setattr(hub_kernels, "lazy_load_kernel", original)
    try:
        from transformers.models.nemotron_h import modeling_nemotron_h as m
        if hasattr(m, "lazy_load_kernel"):
            monkeypatch.setattr(m, "lazy_load_kernel", original)
    except Exception:
        pass
    if "lazy_load_kernel" in vars(integrations):
        monkeypatch.setattr(integrations, "lazy_load_kernel", original)


def test_kernel_without_causal_conv1d_cuda_takes_the_naive_path(
    monkeypatch, restore_lazy_load_kernel
):
    _fake_kernels(monkeypatch, conv_cuda = None)
    assert (
        _nemotron_h_fast_path()
    ), "without the fix the fused path is chosen and asserts in training"
    fix_transformers_mamba_ssm_without_causal_conv1d_cuda()
    assert not _nemotron_h_fast_path()


def test_kernel_with_causal_conv1d_cuda_keeps_the_fused_path(monkeypatch, restore_lazy_load_kernel):
    kernel = _fake_kernels(monkeypatch, conv_cuda = types.ModuleType("causal_conv1d_cuda"))
    fix_transformers_mamba_ssm_without_causal_conv1d_cuda()
    assert hub_kernels.lazy_load_kernel("mamba-ssm") is kernel
    assert _nemotron_h_fast_path()


def test_fix_is_idempotent(restore_lazy_load_kernel):
    fix_transformers_mamba_ssm_without_causal_conv1d_cuda()
    first = hub_kernels.lazy_load_kernel
    fix_transformers_mamba_ssm_without_causal_conv1d_cuda()
    assert hub_kernels.lazy_load_kernel is first


class _LazyAliasModule(types.ModuleType):
    # Like transformers 5.5's `image_processing_<x>_fast` aliases: any missing attribute imports a processor.
    accesses = 0

    def __getattr__(self, name):
        type(self).accesses += 1
        raise RuntimeError("operator torchvision::nms does not exist")


def test_sweep_skips_lazy_aliases_and_rebinds_modeling_modules(
    monkeypatch, restore_lazy_load_kernel
):
    original = hub_kernels.lazy_load_kernel
    alias = _LazyAliasModule("transformers.models.fake_alias.image_processing_fake_alias_fast")
    modeling = types.ModuleType("transformers.models.fake_mamba.modeling_fake_mamba")
    modeling.lazy_load_kernel = original
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    monkeypatch.setattr(_LazyAliasModule, "accesses", 0)
    fix_transformers_mamba_ssm_without_causal_conv1d_cuda()
    assert _LazyAliasModule.accesses == 0
    assert modeling.lazy_load_kernel is hub_kernels.lazy_load_kernel
    assert modeling.lazy_load_kernel is not original
