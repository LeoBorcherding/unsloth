# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Managed engines on AMD ROCm hosts: profile selection, the support gate and the VRAM budget."""

import pytest

from core.inference import engine_install as install
from utils.hardware import amd, hardware


@pytest.fixture
def rocm_linux(monkeypatch):
    from core.inference import wsl_host

    monkeypatch.setattr(hardware, "IS_ROCM", True)
    monkeypatch.setattr(wsl_host, "active", lambda: False)
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install.platform, "libc_ver", lambda: ("glibc", "2.39"))
    monkeypatch.setattr(amd, "amd_kfd_gpu_node_count", lambda: 1)


def test_rocm_host_selects_the_rocm_lock(rocm_linux):
    profile = install.profile("vllm")
    assert profile["lock"] == "vllm-linux-rocm723"
    assert install.python_version("vllm") == (3, 12)
    assert "wheels.vllm.ai/rocm" in profile["index"]
    lock = install.requirements("vllm").read_text(encoding = "utf-8")
    assert "--python-version 3.12 --python-platform x86_64-manylinux_2_39" in lock
    assert "vllm==0.30.0+rocm723" in lock
    assert "nvidia-" not in lock


def test_nvidia_host_keeps_the_cuda_lock(monkeypatch):
    monkeypatch.setattr(hardware, "IS_ROCM", False)
    assert install.profile("vllm")["lock"].startswith("vllm-linux-cu130")
    assert "index" not in install.profile("vllm")
    assert install.python_version("vllm") == install.PYTHON


def test_rocm_engine_never_shares_studios_torch(rocm_linux, monkeypatch):
    monkeypatch.setattr(install, "_studio_packages", lambda: {"torch": "2.12.0+git6bbd260"})
    assert install.install_plan("vllm")["shared"] is False


def test_rocm_gate_passes_on_native_linux_with_a_kfd_gpu(rocm_linux):
    assert install.support_reason("vllm") is None
    assert install.support_reason("vllm", wait = False) is None


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda m, w: m.setattr(w, "active", lambda: True), "native Linux"),
        (lambda m, w: m.setattr(install.platform, "system", lambda: "Windows"), "native Linux"),
        (
            lambda m, w: m.setattr(install.platform, "libc_ver", lambda: ("glibc", "2.35")),
            "glibc 2.39",
        ),
        (lambda m, w: m.setattr(amd, "amd_kfd_gpu_node_count", lambda: 0), "/dev/kfd"),
    ],
)
def test_rocm_gate_refusals(rocm_linux, monkeypatch, change, expected):
    from core.inference import wsl_host

    monkeypatch.setattr(amd, "amd_node_permission_hint", lambda **_: None)
    change(monkeypatch, wsl_host)
    assert expected in install.support_reason("vllm")


def test_sglang_stays_refused_on_rocm(rocm_linux):
    assert install.support_reason("sglang") == "sglang does not support AMD GPUs yet."


def test_rocm_memory_budget_maps_amd_smi_ids_to_hip_ids(monkeypatch):
    from core.inference import engine_adapters
    from utils import vram_budget_settings

    monkeypatch.setattr(hardware, "IS_ROCM", True)
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: 0.97)
    # amd-smi 0 is HIP 1 and the other way round; free, total in MiB.
    monkeypatch.setattr(
        amd, "get_gpu_vram_report", lambda: ({0: (8192, 32768), 1: (30000, 32768)}, [0, 1])
    )
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: {0: 1, 1: 0})
    assert engine_adapters.gpu_memory_fraction([0], 512) == int((30000 - 512) / 32768 * 1000) / 1000
    assert engine_adapters.gpu_memory_fraction([1], 512) == int((8192 - 512) / 32768 * 1000) / 1000


def test_rocm_memory_budget_refuses_an_unmappable_multi_gpu_host(monkeypatch):
    from core.inference import engine_adapters

    monkeypatch.setattr(hardware, "IS_ROCM", True)
    monkeypatch.setattr(
        amd, "get_gpu_vram_report", lambda: ({0: (30000, 32768), 1: (30000, 32768)}, [0, 1])
    )
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: None)
    with pytest.raises(RuntimeError, match = "every selected GPU"):
        engine_adapters.gpu_memory_fraction([0])
    monkeypatch.setattr(amd, "get_gpu_vram_report", lambda: ({0: (30000, 32768)}, [0]))
    assert engine_adapters.gpu_memory_fraction([0], 512) > 0.5
