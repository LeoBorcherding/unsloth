# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""UNSLOTH_ROCM_TORCH opts a Windows RDNA install into AMD's ROCm 10.1 multi-arch torch (unslothai/unsloth#9932).

The default stays 2.11.0+rocm7.14.0. The installed torch is never read from this host: every test pins it.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
_SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"


def _load_stack_module():
    spec = importlib.util.spec_from_file_location("studio_install_python_stack_rocm_opt_in", _STACK_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()

_MULTIARCH = "https://repo.amd.com/rocm/whl-multi-arch/"
_NEXT = "https://stable.repo.amd.com/rocm/whl-next/"
_DEFAULT_TRIO = (
    "torch[device-gfx1201]==2.11.0+rocm7.14.0",
    "torchvision[device-gfx1201]==0.26.0+rocm7.14.0",
    "torchaudio==2.11.0+rocm7.14.0",
)
_ROCM101_TRIO = (
    "torch[device-gfx1201]==2.14.0+rocm10.1.0",
    "torchvision[device-gfx1201]==0.29.0a0+rocm10.1.0",
    "torchaudio==2.11.0.3+rocm10.1.0",
)


@pytest.fixture
def host(monkeypatch):
    """No mirror, no request, and a torch tag the test sets (default: none installed)."""
    for name in ("UNSLOTH_ROCM_TORCH", "UNSLOTH_ROCM_WINDOWS_MIRROR", "UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(stack_mod, "_ROCM_WINDOWS_INDEX_BASE", "https://repo.amd.com/rocm/whl")
    monkeypatch.setattr(stack_mod, "_ROCM_WINDOWS_MULTIARCH_INDEX_BASE", "https://repo.amd.com/rocm/whl-multi-arch")
    monkeypatch.setattr(stack_mod, "_ROCM_TORCH_SET_WARNED", False)
    state = SimpleNamespace(tag = "")
    monkeypatch.setattr(stack_mod, "_installed_torch_local_tag", lambda: state.tag)
    monkeypatch.setattr(stack_mod, "_multiarch_device_pack_installed", lambda gfx: True)
    monkeypatch.setattr(stack_mod, "_detect_windows_gfx_arch", lambda: "gfx1201")
    state.mp = monkeypatch
    return state


class TestSelection:
    def test_default_is_unchanged(self, host):
        assert stack_mod._windows_rocm_index_url("gfx1201") == _MULTIARCH
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1201") == _DEFAULT_TRIO

    @pytest.mark.parametrize("value", ["rocm10.1", "ROCm10.1.0", " rocm10.1 "])
    def test_opt_in_installs_the_exact_10_1_trio_from_whl_next(self, host, value):
        host.mp.setenv("UNSLOTH_ROCM_TORCH", value)
        assert stack_mod._windows_rocm_index_url("gfx1201") == _NEXT
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1201") == _ROCM101_TRIO

    def test_an_installed_10_1_trio_is_kept_without_the_variable(self, host):
        host.tag = "rocm10.1.0"
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1201") == _ROCM101_TRIO
        assert not stack_mod._rocm_windows_torch_set_switch_pending("rocm10.1.0")

    @pytest.mark.parametrize("value", ["default", "rocm7.14", "rocm7.14.0"])
    def test_default_switches_an_opted_in_venv_back(self, host, value):
        host.tag = "rocm10.1.0"
        host.mp.setenv("UNSLOTH_ROCM_TORCH", value)
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1201") == _DEFAULT_TRIO
        assert stack_mod._rocm_windows_torch_set_switch_pending("rocm10.1.0")

    def test_an_unknown_value_warns_once_and_keeps_the_default(self, host, capsys):
        host.mp.setenv("UNSLOTH_ROCM_TORCH", "rocm10.2")
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1201") == _DEFAULT_TRIO
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1201") == _DEFAULT_TRIO
        assert capsys.readouterr().out.count("UNSLOTH_ROCM_TORCH=rocm10.2 is not a known set") == 1

    def test_an_older_default_tag_is_left_to_the_existing_repairs(self, host):
        # Default users must not get a new reinstall from this change.
        assert not stack_mod._rocm_windows_torch_set_switch_pending("rocm7.13.0")
        assert not stack_mod._rocm_windows_torch_set_switch_pending("rocm7.14.0")

    def test_opting_in_marks_a_default_venv_for_reinstall(self, host):
        host.mp.setenv("UNSLOTH_ROCM_TORCH", "rocm10.1")
        assert stack_mod._rocm_windows_torch_set_switch_pending("rocm7.14.0")

    def test_a_multiarch_mirror_serves_the_opt_in_set_too(self, host):
        host.mp.setenv("UNSLOTH_ROCM_TORCH", "rocm10.1")
        mirror = "https://mirror.example/amd?token=abc"
        host.mp.setattr(
            stack_mod,
            "_ROCM_WINDOWS_OPT_IN_TORCH_SETS",
            tuple(s._replace(index_base = mirror) for s in stack_mod._ROCM_WINDOWS_OPT_IN_TORCH_SETS),
        )
        url = stack_mod._windows_rocm_index_url("gfx1201")
        assert url == "https://mirror.example/amd/?token=abc"
        assert stack_mod._windows_rocm_torch_pkg_specs_for(url, "gfx1201") == _ROCM101_TRIO

    def test_whl_next_counts_as_a_multiarch_index(self, host):
        assert stack_mod._index_is_multiarch(_NEXT)
        assert stack_mod._index_is_multiarch("https://other.example/rocm/whl-next")


class TestFastPath:
    @pytest.fixture
    def windows(self, host):
        host.mp.setattr(stack_mod, "IS_WINDOWS", True)
        host.mp.setattr(stack_mod, "NO_TORCH", False)
        host.mp.setattr(stack_mod, "_TORCH_BACKEND", "")
        for name in (
            "_explicit_unknown_family_torch_index_url",
            "_explicit_rocm_torch_index_url",
        ):
            host.mp.setattr(stack_mod, name, lambda: None)
        host.mp.setattr(stack_mod, "_explicit_torch_index_is_unusable", lambda: False)
        host.mp.setattr(stack_mod, "_has_usable_nvidia_gpu", lambda: False)
        return host

    def _installed(self, windows, version):
        import importlib.metadata as md

        windows.tag = version.rpartition("+")[2]
        windows.mp.setattr(md, "version", lambda name: version)

    def test_opting_in_sends_a_default_venv_through_the_dependency_pass(self, windows):
        self._installed(windows, "2.11.0+rocm7.14.0")
        assert not stack_mod._windows_rocm_device_packs_need_dependency_pass()
        windows.mp.setenv("UNSLOTH_ROCM_TORCH", "rocm10.1")
        assert stack_mod._windows_rocm_device_packs_need_dependency_pass()

    def test_an_opted_in_venv_keeps_the_fast_path(self, windows):
        self._installed(windows, "2.14.0+rocm10.1.0")
        assert not stack_mod._windows_rocm_device_packs_need_dependency_pass()


class TestFusedAttentionGate:
    def _gate(self, host, works):
        calls = []
        host.mp.setattr(stack_mod, "_rocm_fused_sdpa_works", lambda: works)
        host.mp.setattr(stack_mod, "pip_install_try", lambda *a, **k: calls.append(a) or True)
        return stack_mod._rocm_windows_opt_in_fused_sdpa_gate("gfx1201"), calls

    def test_the_default_trio_is_never_probed(self, host):
        host.tag = "rocm7.14.0"
        host.mp.setattr(stack_mod, "_rocm_fused_sdpa_works", lambda: pytest.fail("probed"))
        assert stack_mod._rocm_windows_opt_in_fused_sdpa_gate("gfx1201")

    def test_working_fused_attention_keeps_the_opt_in(self, host):
        host.tag = "rocm10.1.0"
        assert self._gate(host, True) == (True, [])

    def test_no_answer_keeps_the_opt_in(self, host):
        host.tag = "rocm10.1.0"
        assert self._gate(host, None) == (True, [])

    def test_failing_fused_attention_puts_the_default_back(self, host, capsys):
        host.tag = "rocm10.1.0"
        host.mp.setenv("UNSLOTH_ROCM_TORCH", "rocm10.1")
        kept, calls = self._gate(host, False)
        assert not kept
        assert len(calls) == 1
        assert calls[0][-3:] == _DEFAULT_TRIO
        assert _MULTIARCH in calls[0]
        assert "fused attention" in capsys.readouterr().out


class TestFusedAttentionProbe:
    @pytest.mark.parametrize(
        "stdout, expected",
        [
            ("UNSLOTH_FUSED_SDPA=start\nUNSLOTH_FUSED_SDPA=ok\n", True),
            ("UNSLOTH_FUSED_SDPA=start\nUNSLOTH_FUSED_SDPA=fail:FLASH_ATTENTION\n", False),
            # Reached the GPU and died inside a fused kernel (the 7.14.1 hipErrorInvalidValue shape).
            ("UNSLOTH_FUSED_SDPA=start\n", False),
            # No GPU, or torch never imported: no answer.
            ("", None),
        ],
    )
    def test_reads_the_probe(self, monkeypatch, stdout, expected):
        monkeypatch.setattr(stack_mod.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = stdout))
        assert stack_mod._rocm_fused_sdpa_works() is expected

    def test_a_timeout_is_no_answer(self, monkeypatch):
        def boom(*a, **k):
            raise stack_mod.subprocess.TimeoutExpired("python", 180)

        monkeypatch.setattr(stack_mod.subprocess, "run", boom)
        assert stack_mod._rocm_fused_sdpa_works() is None


def _ps_sets(path):
    text = path.read_text(encoding = "utf-8")
    start = text.index("function Get-RocmWindowsTorchSet")
    body = text[start : text.index("\n}", start) if path.name == "setup.ps1" else text.index("\n    }", start)]
    return {
        m.group(1): (m.group(2), m.group(3), m.group(4), m.group(5))
        for m in re.finditer(
            r'Name = "([^"]+)"; IndexBase = [^;]+; Tag = "([^"]+)"; Torch = "([^"]+)"; Vision = "([^"]+)"; Audio = "([^"]+)"',
            body,
        )
    }


class TestPowerShellTablesMatch:
    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
    def test_same_sets_as_python(self, path):
        expected = {
            s.name: (s.tag, s.torch, s.torchvision, s.torchaudio)
            for s in (stack_mod._rocm_windows_default_torch_set(), *stack_mod._ROCM_WINDOWS_OPT_IN_TORCH_SETS)
        }
        assert _ps_sets(path) == expected
