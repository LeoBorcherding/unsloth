# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generic rocm7.2 index admits torch 2.11-2.14 (#9932) in the update path too; the AMD
per-arch leaves and the Windows pins stay on 2.11, and a rocm7.2 pin accepts the whole window."""

import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "install_python_stack_9932", REPO / "studio" / "install_python_stack.py"
)
stack = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stack
_spec.loader.exec_module(stack)

_TRIO_211 = (
    "torch>=2.11.0,<2.12.0",
    "torchvision>=0.26.0,<0.27.0",
    "torchaudio>=2.11.0,<2.12.0",
)


def _install_sh_case(leaf_pattern: str) -> tuple[str, str, str]:
    """The (torch, torchvision, torchaudio) install.sh sets in the arm named by leaf_pattern."""
    src = (REPO / "install.sh").read_text(encoding = "utf-8")
    block = src[src.index("# rocm7.2 and per-gfx indexes ship torch 2.11.0") :]
    block = block[: block.index("\nesac\n")]
    arm = block[block.index(f"    {leaf_pattern})") :]
    arm = arm[: arm.index(";;")]
    return tuple(
        re.search(rf'{name}_CONSTRAINT="([^"]+)"', arm).group(1)
        for name in ("TORCH", "TORCHVISION", "TORCHAUDIO")
    )


def test_generic_rocm72_spec_matches_install_sh():
    assert stack._ROCM_TORCH_PKG_SPECS["rocm7.2"] == _install_sh_case("rocm7.2")
    assert stack._ROCM_TORCH_PKG_SPECS["rocm7.2"] == (
        "torch>=2.11.0,<2.15.0",
        "torchvision>=0.26.0,<0.30.0",
        "torchaudio>=2.11.0,<2.12.0",
    )


def test_per_arch_spec_matches_install_sh_and_stays_211():
    assert stack._ROCM_TORCH211_PKG_SPEC == _TRIO_211
    assert (
        _install_sh_case("gfx120x-all|gfx1151|gfx1150|gfx1152|gfx103x-all|gfx110x-all") == _TRIO_211
    )


def test_windows_pins_unchanged():
    assert set(stack._WINDOWS_ROCM_TORCH_PKG_SPECS.values()) == {_TRIO_211}


def test_ceiling_constant_matches_spec():
    hi = re.search(r"<(\d+)\.(\d+)\.0", stack._ROCM_TORCH_PKG_SPECS["rocm7.2"][0])
    assert (int(hi.group(1)), int(hi.group(2))) == stack._ROCM72_TORCH_CEILING


@pytest.mark.parametrize(
    "installed, mismatch",
    [
        ("2.11.0+rocm7.2", False),
        ("2.12.1+rocm7.2", False),
        ("2.14.1+rocm7.2", False),
        ("2.10.0+rocm7.2", True),
        ("2.15.0+rocm7.2", True),
        ("2.14.1+rocm7.1", True),
        ("2.14.1", True),
    ],
)
def test_rocm72_pin_accepts_the_window(installed, mismatch, monkeypatch):
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    assert (
        stack._rocm_pin_family_mismatch("https://download.pytorch.org/whl/rocm7.2", installed)
        is mismatch
    )


# WSL stays on 2.11: 2.12+ bundles a librocprofiler-sdk that aborts on DXG at the first CUDA call.
def test_install_sh_rocm72_wsl_branch_matches_211_spec():
    src = (REPO / "install.sh").read_text(encoding = "utf-8")
    arm = src[src.index("# rocm7.2 and per-gfx indexes ship torch 2.11.0") :]
    arm = arm[arm.index("    rocm7.2)") : arm.index(";;")]
    wsl_branch = arm[arm.index('if [ "${OS:-}" = "wsl" ]; then') : arm.index("        fi")]
    got = [re.search(rf'{n}_CONSTRAINT="([^"]+)"', wsl_branch) for n in ("TORCH", "TORCHVISION")]
    assert tuple(m.group(1) for m in got) == _TRIO_211[:2]
    assert _install_sh_case("rocm7.2")[2] == _TRIO_211[2]


@pytest.mark.parametrize(
    "installed, mismatch",
    [
        ("2.11.0+rocm7.2", False),
        ("2.11.5+rocm7.2", False),
        ("2.12.0+rocm7.2", True),
        ("2.14.1+rocm7.2", True),
        ("2.10.0+rocm7.2", True),
    ],
)
def test_rocm72_pin_on_wsl_accepts_211_only(installed, mismatch, monkeypatch):
    monkeypatch.setattr(stack, "_is_wsl", lambda: True)
    monkeypatch.setattr(stack, "_installed_rocm_wheel_family", lambda: None)
    assert (
        stack._rocm_pin_family_mismatch("https://download.pytorch.org/whl/rocm7.2", installed)
        is mismatch
    )


@pytest.mark.parametrize("wsl", [True, False])
def test_update_rocm72_pin_over_2141(wsl, monkeypatch):
    """A rocm7.2 pin over 2.14.1 reinstalls the 2.11 trio on WSL; natively 2.14.1 is kept."""
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "rocm7.2")
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    probe = MagicMock(
        returncode = 0, stdout = stack._TORCH_PROBE_MARKER + "2.14.1+rocm7.2|7.2.12345|\n"
    )
    with (
        patch.object(stack, "_is_wsl", return_value = wsl),
        patch.object(stack, "_installed_rocm_wheel_family", lambda: None),
        patch.object(stack, "IS_MACOS", False),
        patch.object(stack, "IS_WINDOWS", False),
        patch("platform.machine", return_value = "x86_64"),
        patch.object(stack, "pip_install_try", return_value = True),
        patch.object(stack, "pip_install") as pip,
        patch.object(stack, "_has_usable_nvidia_gpu", return_value = False),
        patch.object(stack, "_has_rocm_gpu", return_value = True),
        patch.object(stack, "_detect_rocm_version", return_value = (7, 2)),
        patch("os.path.isdir", return_value = True),
        patch("subprocess.run", return_value = probe),
    ):
        stack._ensure_rocm_torch()
    torch_calls = [
        c for c in pip.call_args_list if any(str(a).startswith("torch>=") for a in c.args)
    ]
    if wsl:
        assert len(torch_calls) == 1
        assert all(spec in torch_calls[0].args for spec in _TRIO_211)
    else:
        assert torch_calls == []


@pytest.mark.parametrize(
    "installed, unsafe",
    [
        ("2.11.0+rocm7.2", False),
        ("2.12.0+rocm7.2", True),
        ("2.14.1+rocm7.2", True),
        ("2.14.1+rocm7.20", False),
        ("2.11.0+rocm7.13.0", False),
        ("2.14.1+cpu", False),
    ],
)
def test_wsl_rocm72_torch_unsafe(installed, unsafe, monkeypatch):
    monkeypatch.setattr(stack, "_is_wsl", lambda: True)
    assert stack._wsl_rocm72_torch_unsafe(installed) is unsafe
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    assert stack._wsl_rocm72_torch_unsafe(installed) is False


@pytest.mark.parametrize("wsl", [True, False])
def test_update_unpinned_2141_on_wsl(wsl, monkeypatch):
    """No pin: an existing 2.14.1+rocm7.2 is reinstalled to the 2.11 trio on WSL, kept natively."""
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    probe = MagicMock(
        returncode = 0, stdout = stack._TORCH_PROBE_MARKER + "2.14.1+rocm7.2|7.2.12345|\n"
    )
    with (
        patch.object(stack, "_is_wsl", return_value = wsl),
        patch.object(stack, "_installed_rocm_wheel_family", lambda: None),
        patch.object(stack, "IS_MACOS", False),
        patch.object(stack, "IS_WINDOWS", False),
        patch("platform.machine", return_value = "x86_64"),
        patch.object(stack, "pip_install_try", return_value = True),
        patch.object(stack, "pip_install") as pip,
        patch.object(stack, "_has_usable_nvidia_gpu", return_value = False),
        patch.object(stack, "_has_rocm_gpu", return_value = True),
        patch.object(stack, "_detect_rocm_version", return_value = (7, 2)),
        patch("os.path.isdir", return_value = True),
        patch("subprocess.run", return_value = probe),
    ):
        stack._ensure_rocm_torch()
    torch_calls = [
        c for c in pip.call_args_list if any(str(a).startswith("torch>=") for a in c.args)
    ]
    if wsl:
        assert len(torch_calls) == 1
        assert all(spec in torch_calls[0].args for spec in _TRIO_211)
    else:
        assert torch_calls == []
