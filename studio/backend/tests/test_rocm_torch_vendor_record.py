"""A ROCm torch chosen on a mixed NVIDIA+AMD host survives update, Repair and an install.sh re-run.

UNSLOTH_FORCE_ROCM_TORCH (or a ROCm index pin) is exported for one run; the manifest's
rocm_torch_requested key carries it to the next one. The variable still wins, and =0 clears it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_rocm_torch_explicit_request import (  # noqa: F401
    _MIXED_HOST,
    _NO_AMD_CARD,
    _STACK,
    _clean_env,
    _index_url,
    _mixed_host,
    _no_real_install,
    _shell_function,
    stack,
)

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "studio"))
import install_manifest  # noqa: E402


def _venv(tmp_path: Path, **keys) -> Path:
    """A venv root holding a manifest a previous install wrote."""
    install_manifest.write_manifest(root = tmp_path, req_root = tmp_path, **keys)
    return tmp_path


def _fresh_stack(monkeypatch, root: Path):
    """install_python_stack as an update in the venv at ``root`` imports it."""
    # conftest already points venv_root at a per-test directory; this test names its own.
    monkeypatch.setattr(sys.modules["install_manifest"], "venv_root", lambda: root)
    spec = importlib.util.spec_from_file_location("_unsloth_stack_vendor_record", _STACK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "pip_install", lambda *a, **k: pytest.fail("reached an install"))
    return module


_FORCED = dict(expected_torch_tag = "rocm", rocm_torch_requested = True)
_LEGACY_PIN = dict(expected_torch_tag = "rocm", expected_torch_tag_pinned = True)


# The record itself.


def test_the_key_round_trips(tmp_path):
    assert install_manifest.recorded_rocm_torch_requested(_venv(tmp_path, **_FORCED)) is True
    assert install_manifest.read_manifest(tmp_path)["rocm_torch_requested"] is True


@pytest.mark.parametrize("value", [False, None])
def test_no_request_writes_no_key(tmp_path, value):
    _venv(tmp_path, expected_torch_tag = "rocm", rocm_torch_requested = value)
    assert "rocm_torch_requested" not in install_manifest.read_manifest(tmp_path)
    assert install_manifest.recorded_rocm_torch_requested(tmp_path) is False


def test_a_pinned_rocm_flavor_from_an_older_manifest_counts(tmp_path):
    assert install_manifest.recorded_rocm_torch_requested(_venv(tmp_path, **_LEGACY_PIN)) is True


def test_an_auto_selected_rocm_flavor_is_not_a_request(tmp_path):
    """A pure AMD host gets ROCm by detection; that is no choice to keep."""
    assert (
        install_manifest.recorded_rocm_torch_requested(_venv(tmp_path, expected_torch_tag = "rocm"))
        is False
    )


def test_an_update_killed_mid_pass_keeps_the_choice(tmp_path):
    _venv(tmp_path, **_FORCED)
    install_manifest.remove_manifest(tmp_path)
    assert install_manifest.read_manifest(tmp_path) is None
    assert install_manifest.recorded_rocm_torch_requested(tmp_path) is True


def test_extras_cannot_forge_the_key(tmp_path):
    _venv(tmp_path, extra = {"rocm_torch_requested": True})
    assert install_manifest.recorded_rocm_torch_requested(tmp_path) is False


# The update path: setup.sh's --cuda-torch-needs-dependency-pass probe and the repair behind it.


def _update_on_mixed_host(
    monkeypatch,
    tmp_path,
    *,
    env = None,
    **host,
):
    module = _fresh_stack(monkeypatch, tmp_path)
    _mixed_host(module, monkeypatch, request = False, torch = "2.11.0+rocm7.0", **host)
    if env is not None:
        monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", env)
    return module


@pytest.mark.parametrize("record", [_FORCED, _LEGACY_PIN], ids = ["forced", "rocm-pin"])
def test_an_update_keeps_a_recorded_rocm_torch(monkeypatch, tmp_path, capsys, record):
    """Fails before the fix: the probe answered "ROCm build on an NVIDIA host" and the pass
    force-reinstalled CUDA torch."""
    _venv(tmp_path, **record)
    module = _update_on_mixed_host(monkeypatch, tmp_path)
    assert module._rocm_torch_explicitly_requested() is True
    assert module._cuda_torch_needs_dependency_pass() is False
    module._ensure_cuda_torch()
    assert "keeping ROCm torch, recorded at install" in capsys.readouterr().out


def test_force_zero_switches_back_to_cuda(monkeypatch, tmp_path):
    _venv(tmp_path, **_FORCED)
    module = _update_on_mixed_host(monkeypatch, tmp_path, env = "0")
    assert module._rocm_torch_explicitly_requested() is False
    assert module._cuda_torch_needs_dependency_pass() is True


def test_a_record_whose_amd_card_is_gone_is_ignored(monkeypatch, tmp_path):
    _venv(tmp_path, **_FORCED)
    module = _update_on_mixed_host(monkeypatch, tmp_path, rocm_gpu = False, archs = [])
    assert module._cuda_torch_needs_dependency_pass() is True


def test_no_record_still_repairs_rocm_on_nvidia(monkeypatch, tmp_path):
    """The KFD-poisoned venv this repair exists for: ROCm torch nobody asked for."""
    _venv(tmp_path, expected_torch_tag = "rocm")
    module = _update_on_mixed_host(monkeypatch, tmp_path)
    assert module._cuda_torch_needs_dependency_pass() is True


def test_a_pure_nvidia_host_is_unchanged(monkeypatch, tmp_path):
    _venv(tmp_path, expected_torch_tag = "cu130")
    module = _fresh_stack(monkeypatch, tmp_path)
    _mixed_host(module, monkeypatch, request = False, rocm_gpu = False, archs = [])
    assert module._rocm_torch_explicitly_requested() is False
    assert module._cuda_torch_needs_dependency_pass() is False


# What the pass writes back.


@pytest.mark.parametrize(
    "env, pin, resident, expected",
    [
        ("1", None, "2.11.0+rocm7.0", True),
        ("1", None, "2.11.0+cu130", False),  # the request found no AMD card and fell to CUDA
        ("0", None, "2.11.0+rocm7.0", False),
        (None, "rocm7.0", "2.11.0+rocm7.0", True),
        (None, None, "2.11.0+rocm7.0", False),
    ],
    ids = ["forced", "forced-fell-to-cuda", "force-zero", "rocm-pin", "auto-rocm"],
)
def test_the_pass_records_only_a_deliberate_resident_rocm(
    stack, monkeypatch, env, pin, resident, expected
):
    _mixed_host(stack, monkeypatch, request = False, torch = resident, pin = pin)
    monkeypatch.setattr(stack, "_RECORDED_ROCM_TORCH_REQUESTED", False)
    monkeypatch.setattr(stack, "_RECORDED_TORCH_TAG_PINNED", False)
    if env is not None:
        monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", env)
    assert stack._rocm_torch_request_to_record() is expected


def test_the_pass_carries_the_record_forward(stack, monkeypatch):
    _mixed_host(stack, monkeypatch, request = False, torch = "2.11.0+rocm7.0")
    monkeypatch.setattr(stack, "_RECORDED_ROCM_TORCH_REQUESTED", True)
    assert stack._rocm_torch_request_to_record() is True


# install.sh: a re-run and Unsloth Desktop's Repair.


def _recorded_by_shell(root: Path) -> str:
    script = "\n".join(
        [_shell_function("_recorded_rocm_torch_request"), f"_recorded_rocm_torch_request '{root}'"]
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, env = _clean_env()
    ).stdout.strip()


@pytest.mark.parametrize(
    "keys, expected",
    [
        (_FORCED, "true"),
        (_LEGACY_PIN, "true"),
        (dict(expected_torch_tag = "rocm"), "false"),
        (dict(expected_torch_tag = "cu130", expected_torch_tag_pinned = True), "false"),
    ],
    ids = ["forced", "rocm-pin", "auto-rocm", "cuda-pin"],
)
def test_the_shell_reader_agrees_with_the_python_one(tmp_path, keys, expected):
    _venv(tmp_path, **keys)
    assert _recorded_by_shell(tmp_path) == expected
    assert str(install_manifest.recorded_rocm_torch_requested(tmp_path)).lower() == expected


def test_the_shell_reader_reads_the_parked_copy_and_no_manifest(tmp_path):
    assert _recorded_by_shell(tmp_path) == "false"
    _venv(tmp_path, **_FORCED)
    install_manifest.remove_manifest(tmp_path)
    assert _recorded_by_shell(tmp_path) == "true"


def test_a_rerun_selects_rocm_from_the_record(stack):
    """Fails before the fix: with the variable unset the re-run picked cu* and the flavor check
    reinstalled CUDA torch over the ROCm one."""
    out = _index_url("_ROCM_TORCH_RECORDED=true", _MIXED_HOST)
    assert "rocm" in out or "gfx" in out, out


def test_a_rerun_with_force_zero_selects_cuda_over_the_record(stack):
    out = _index_url("UNSLOTH_FORCE_ROCM_TORCH=0 _ROCM_TORCH_RECORDED=true", _MIXED_HOST)
    assert "rocm" not in out and "gfx" not in out, out


def test_a_rerun_with_the_record_but_no_amd_card_selects_cuda(stack):
    out = _index_url("_ROCM_TORCH_RECORDED=true", _NO_AMD_CARD)
    assert "rocm" not in out and "gfx" not in out, out


def test_install_sh_reads_the_record_before_moving_the_venv_aside():
    """install.sh moves the old venv aside before it picks an index, so a read after that finds
    nothing; the fresh venv's pass then needs the choice handed over in the environment."""
    text = (_REPO / "install.sh").read_text(encoding = "utf-8")
    read = text.index('_ROCM_TORCH_RECORDED=$(_recorded_rocm_torch_request "$VENV_DIR")')
    moved = text.index('_start_studio_venv_replacement "$VENV_DIR"; then')
    picked = text.index("TORCH_INDEX_URL=$(get_torch_index_url)")
    handover = text.index("export UNSLOTH_FORCE_ROCM_TORCH=1")
    assert read < moved < picked < handover


def test_the_manifest_names_agree_with_install_sh():
    text = (_REPO / "install.sh").read_text(encoding = "utf-8")
    assert install_manifest.MANIFEST_NAME in text
    assert install_manifest.PREVIOUS_MANIFEST_NAME in text
    assert json.dumps("rocm_torch_requested") in text
