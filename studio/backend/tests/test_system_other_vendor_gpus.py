# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vendor-aware GPU ids on /api/system: an NVIDIA + AMD host lists both cards (#12248)."""

from types import SimpleNamespace

import pytest

import main

_MASKS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

_RTX_4090 = {
    "vendor": "nvidia",
    "index": 0,
    "name": "NVIDIA GeForce RTX 4090",
    "memory_total_gb": 23.99,
    "source": "nvidia-smi",
}
# sysfs publishes no name, only the VRAM size.
_R9700 = {
    "vendor": "amd",
    "index": 0,
    "name": None,
    "memory_total_gb": 31.86,
    "source": "sysfs-drm",
}


def _refuse(*args, **kwargs):
    raise AssertionError("this probe must not run")


def _host(
    monkeypatch,
    *,
    torch_backend,
    torch_cards,
    vendors,
    inventory,
    amd_supported = True,
    env = None,
):
    """Pins torch's view, the vendor check and the torch-free inventory; nothing is read off the runner."""
    import utils.hardware as hardware
    import utils.hardware.hardware as hw
    import utils.llama_cpp_path_settings as ps
    from core.inference.llama_cpp import LlamaCppBackend

    for name in _MASKS:
        monkeypatch.delenv(name, raising = False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    cards = [
        {
            "index": i,
            "index_kind": "physical",
            "visible_ordinal": i,
            "name": name,
            "memory_total_gb": total,
        }
        for i, (name, total) in enumerate(torch_cards)
    ]
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": bool(cards),
            "backend": torch_backend,
            "index_kind": "physical",
            "devices": cards,
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {
            "available": bool(cards),
            "backend": torch_backend,
            "devices": [{"index": d["index"], "vram_used_gb": 1.0} for d in cards],
        },
    )
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "get_cross_vendor_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "get_mixed_vulkan_inference_gpu_info", lambda info: None)
    monkeypatch.setattr(hardware, "amd_driver_warning_report", lambda: {})
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: False))
    monkeypatch.setattr(ps, "host_gpu_vendors", lambda: set(vendors))
    if inventory is None:
        monkeypatch.setattr(hw, "get_physical_gpu_inventory", _refuse)
    else:
        monkeypatch.setattr(
            hw,
            "get_physical_gpu_inventory",
            lambda block = True: {"available": True, "devices": [dict(d) for d in inventory]},
        )
    monkeypatch.setattr(hw, "_amd_device_can_establish_a_mismatch", lambda device: amd_supported)
    monkeypatch.setattr(main, "_system_gpu_cache", None)


def _system_gpu():
    gpu, _ = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))
    return gpu


def test_nvidia_only_host_is_unchanged_but_for_the_vendor_tags(monkeypatch):
    _host(
        monkeypatch,
        torch_backend = "cuda",
        torch_cards = [("NVIDIA GeForce RTX 4090", 23.99)],
        vendors = {"nvidia"},
        inventory = None,  # a single-vendor host never starts the inventory probe
    )
    gpu = _system_gpu()

    assert "other_vendor_devices" not in gpu
    (card,) = gpu["devices"]
    assert card["index"] == 0
    assert card["vendor"] == "nvidia"
    assert card["qualified_id"] == "cuda:0"


def test_amd_only_host_is_unchanged_but_for_the_vendor_tags(monkeypatch):
    _host(
        monkeypatch,
        torch_backend = "rocm",
        torch_cards = [("AMD Radeon AI PRO R9700", 31.86)],
        vendors = {"amd"},
        inventory = None,
    )
    gpu = _system_gpu()

    assert "other_vendor_devices" not in gpu
    (card,) = gpu["devices"]
    assert (card["index"], card["vendor"], card["qualified_id"]) == (0, "amd", "rocm:0")


def test_mixed_host_with_cuda_torch_lists_the_amd_card_beside_devices(monkeypatch):
    """The Discord report: RTX 4090 + R9700, torch 2.13.0+cu130, only the 4090 was listed."""
    _host(
        monkeypatch,
        torch_backend = "cuda",
        torch_cards = [("NVIDIA GeForce RTX 4090", 23.99)],
        vendors = {"nvidia", "amd"},
        inventory = [_RTX_4090, _R9700],
    )
    gpu = _system_gpu()

    # devices stays the torch-usable list fit budgets and the pin read.
    assert [d["qualified_id"] for d in gpu["devices"]] == ["cuda:0"]
    assert gpu["other_vendor_devices"] == [
        {
            "vendor": "amd",
            "backend": "rocm",
            "index": 0,
            "index_kind": "physical",
            "qualified_id": "rocm:0",
            "name": None,
            "memory_total_gb": 31.86,
            "source": "sysfs-drm",
            "trainable": False,
            "reason": "torch_backend_mismatch",
        }
    ]


def test_mixed_host_with_rocm_torch_lists_the_nvidia_card_beside_devices(monkeypatch):
    _host(
        monkeypatch,
        torch_backend = "rocm",
        torch_cards = [("AMD Radeon AI PRO R9700", 31.86)],
        vendors = {"nvidia", "amd"},
        inventory = [_RTX_4090, _R9700],
    )
    gpu = _system_gpu()

    assert [d["qualified_id"] for d in gpu["devices"]] == ["rocm:0"]
    (other,) = gpu["other_vendor_devices"]
    assert other["qualified_id"] == "cuda:0"
    assert other["name"] == "NVIDIA GeForce RTX 4090"
    assert other["trainable"] is False


def test_emptied_cuda_mask_on_a_mixed_host_hides_both_cards(monkeypatch):
    """HIP reads CUDA_VISIBLE_DEVICES when HIP_ and ROCR_VISIBLE_DEVICES are unset, so the AMD card is hidden too."""
    _host(
        monkeypatch,
        torch_backend = "cuda",
        torch_cards = [],
        vendors = {"nvidia", "amd"},
        inventory = None,
        env = {"CUDA_VISIBLE_DEVICES": ""},
    )
    gpu = _system_gpu()

    assert gpu["devices"] == []
    assert "other_vendor_devices" not in gpu


def test_emptied_cuda_mask_with_a_hip_mask_still_lists_the_amd_card(monkeypatch):
    _host(
        monkeypatch,
        torch_backend = "cuda",
        torch_cards = [],
        vendors = {"nvidia", "amd"},
        inventory = [_RTX_4090, _R9700],
        env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "0"},
    )
    gpu = _system_gpu()

    assert gpu["devices"] == []
    assert [d["qualified_id"] for d in gpu["other_vendor_devices"]] == ["rocm:0"]


def test_an_amd_card_the_installers_leave_on_cpu_torch_is_not_listed(monkeypatch):
    _host(
        monkeypatch,
        torch_backend = "cuda",
        torch_cards = [("NVIDIA GeForce RTX 4090", 23.99)],
        vendors = {"nvidia", "amd"},
        inventory = [_RTX_4090, _R9700],
        amd_supported = False,
    )
    assert "other_vendor_devices" not in _system_gpu()


def test_a_cpu_torch_leaves_the_cards_to_the_mismatch_report(monkeypatch):
    """physical_devices already lists every card when torch can open none; no second list."""
    _host(
        monkeypatch,
        torch_backend = "cpu",
        torch_cards = [],
        vendors = {"nvidia", "amd"},
        inventory = None,
    )
    assert "other_vendor_devices" not in _system_gpu()


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, {"nvidia", "amd"}),
        ({"CUDA_VISIBLE_DEVICES": ""}, set()),
        ({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "0"}, {"amd"}),
        ({"ROCR_VISIBLE_DEVICES": "-1"}, {"nvidia"}),
    ],
)
def test_reachable_gpu_vendors_applies_the_masks(monkeypatch, env, expected):
    import utils.llama_cpp_path_settings as ps

    for name in _MASKS:
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ps, "host_gpu_vendors", lambda: {"nvidia", "amd"})

    assert ps.reachable_gpu_vendors() == expected


def test_vulkan_rows_carry_no_vendor_tags():
    from utils.hardware import tag_gpu_vendor
    rows = [{"index": 0, "index_kind": "vulkan", "name": "Vulkan0"}]
    assert tag_gpu_vendor(rows, "vulkan") == rows


@pytest.mark.parametrize(
    "platform, expected", [("linux", {"nvidia"}), ("win32", {"nvidia", "amd"})]
)
def test_an_emptied_rocr_mask_hides_amd_only_where_rocr_exists(monkeypatch, platform, expected):
    """Windows HIP has no ROCr layer, so ROCR_VISIBLE_DEVICES cannot hide its cards there."""
    import utils.llama_cpp_path_settings as ps

    for name in _MASKS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    monkeypatch.setattr(ps.sys, "platform", platform)
    monkeypatch.setattr(ps, "host_gpu_vendors", lambda: {"nvidia", "amd"})

    assert ps.reachable_gpu_vendors() == expected
