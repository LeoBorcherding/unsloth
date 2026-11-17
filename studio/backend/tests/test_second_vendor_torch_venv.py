# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UNSLOTH_SECOND_VENDOR_TORCH=1 adds a venv with the other vendor's torch on a mixed host (#12248)."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from utils.second_torch_venv import RECORD_NAME, second_torch_venv

_HERE = Path(__file__).resolve().parent
_INSTALL_SH = _HERE.parents[2] / "install.sh"


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"_unsloth_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The get_torch_index_url harness and host stubs the UNSLOTH_FORCE_ROCM_TORCH tests already use.
_rocm = _load_sibling("test_rocm_torch_explicit_request")
_fn = _rocm._shell_function


def _run(script: str, env: "dict | None" = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, env = _rocm._clean_env(env)
    )


@pytest.mark.parametrize("value", ["1", "true", " YES ", "on"])
def test_the_request_is_recognised(value):
    out = _run(
        _fn("_second_vendor_torch_requested") + "\n_second_vendor_torch_requested",
        {"UNSLOTH_SECOND_VENDOR_TORCH": value},
    )
    assert out.returncode == 0


@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_anything_else_is_not_a_request(value):
    """The default install must never see the flag as set."""
    env = dict(os.environ)
    env.pop("UNSLOTH_SECOND_VENDOR_TORCH", None)
    if value is not None:
        env["UNSLOTH_SECOND_VENDOR_TORCH"] = value
    out = subprocess.run(
        ["bash", "-c", _fn("_second_vendor_torch_requested") + "\n_second_vendor_torch_requested"],
        env = env,
    )
    assert out.returncode == 1


def test_the_install_step_is_gated_on_the_request():
    """The only call site sits behind the flag, so a default install runs none of it."""
    text = _INSTALL_SH.read_text(encoding = "utf-8")
    calls = [l for l in text.splitlines() if "_install_second_vendor_torch " in l and "()" not in l]
    assert len(calls) == 1, calls
    gate = text.splitlines()[text.splitlines().index(calls[0]) - 1]
    assert "_second_vendor_torch_requested" in gate, gate


_LEAF_DEFS = "\n".join(
    [
        _fn("_torch_index_url_leaf"),
        _fn("_torch_index_url_is_rocm"),
        _fn("_second_vendor_for_index"),
    ]
)


@pytest.mark.parametrize(
    "main_index, expected",
    [
        ("https://download.pytorch.org/whl/cu128", "rocm"),
        ("https://download.pytorch.org/whl/cu130/", "rocm"),
        ("https://download.pytorch.org/whl/rocm7.0", "cuda"),
        ("https://repo.amd.com/rocm/whl/gfx120X-all/", "cuda"),
        ("https://download.pytorch.org/whl/cpu", None),
        ("https://download.pytorch.org/whl/xpu", None),
    ],
)
def test_the_second_vendor_is_the_one_the_main_venv_did_not_get(main_index, expected):
    out = _run(_LEAF_DEFS + f"\n_second_vendor_for_index '{main_index}'")
    if expected is None:
        assert out.returncode != 0, out.stdout
    else:
        assert out.returncode == 0 and out.stdout.strip() == expected, out


def _second_index(
    vendor: str,
    stubs: str,
    env: str = "",
) -> "str | None":
    """_second_vendor_torch_index_url under a stubbed host; None when it declines."""
    script = "\n".join(
        [
            'uname() { case "${1:-}" in -m) echo x86_64 ;; *) echo Linux ;; esac; }',
            _fn("_rocm_torch_explicitly_requested"),
            stubs,
            *_rocm._wheel_route_defs(arch_family = False),
            _rocm._shell_constant("_ROCM_BNB_GENERIC_FLOOR_TAG"),
            _fn("_rocm_bnb_compatible_generic_tag"),
            _fn("_trim_index_path_slashes"),
            _fn("get_torch_index_url"),
            _LEAF_DEFS,
            _fn("_second_vendor_torch_index_url"),
            f"{env} _second_vendor_torch_index_url {vendor}",
        ]
    )
    out = _run(script)
    return out.stdout.strip() if out.returncode == 0 else None


def test_a_cuda_main_venv_gets_a_rocm_second_index_on_a_mixed_host():
    url = _second_index("rocm", _rocm._MIXED_HOST)
    assert url and "/rocm" in url, url


def test_a_rocm_main_venv_gets_a_cuda_second_index_on_a_mixed_host():
    """The main venv was forced to ROCm; the second one must still find the NVIDIA card's index."""
    url = _second_index("cuda", _rocm._MIXED_HOST, "UNSLOTH_FORCE_ROCM_TORCH=1")
    # The harness's nvidia-smi is a shell function _run_bounded cannot exec, so the leaf is the
    # cu126 fallback; the point is that it is a CUDA leaf at all.
    assert url and url.rsplit("/", 1)[-1].startswith("cu"), url


def test_the_main_venvs_index_pin_does_not_leak_into_the_second():
    """A cu128 pin for the main venv would otherwise come back verbatim as the 'ROCm' index."""
    url = _second_index(
        "rocm",
        _rocm._MIXED_HOST,
        "UNSLOTH_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 UNSLOTH_TORCH_INDEX_FAMILY=cu128",
    )
    assert url and "/rocm" in url, url


def test_no_amd_card_means_no_rocm_second_index():
    assert _second_index("rocm", _rocm._NO_AMD_CARD) is None


def test_no_nvidia_card_means_no_cuda_second_index():
    pure_amd = _rocm._MIXED_HOST.replace(
        "_has_usable_nvidia_gpu() { return 0; }", "_has_usable_nvidia_gpu() { return 1; }"
    )
    assert _second_index("cuda", pure_amd) is None


def test_the_second_index_override_is_used_verbatim():
    url = _second_index(
        "rocm",
        _rocm._NO_AMD_CARD,
        "UNSLOTH_SECOND_VENDOR_TORCH_INDEX_URL=' https://example.invalid/whl/rocm7.2/ '",
    )
    assert url == "https://example.invalid/whl/rocm7.2", url


def _fake_python(tmp_path: Path, version: str) -> Path:
    """A python that imports a stub torch; -I is dropped so the stub's PYTHONPATH applies."""
    pkg = tmp_path / "fake_site" / "torch"
    pkg.mkdir(parents = True)
    hip = "'7.0'" if "rocm" in version else "None"
    cuda = "'13.0'" if "+cu" in version else "None"
    (pkg / "__init__.py").write_text(
        f"__version__ = {version!r}\nclass version:\n    hip = {hip}\n    cuda = {cuda}\n"
    )
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text(
        "#!/bin/sh\n"
        '[ "$1" = -I ] && shift\n'
        f'PYTHONPATH="{tmp_path / "fake_site"}" exec "{sys.executable}" "$@"\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper


def _install(
    tmp_path: Path,
    *,
    torch_version: str,
    pip_fails: bool = False,
) -> subprocess.CompletedProcess:
    """_install_second_vendor_torch with uv and the index lookup stubbed, in a scratch STUDIO_HOME."""
    wrapper = _fake_python(tmp_path, torch_version)
    pip_rc = 1 if pip_fails else 0
    script = "\n".join(
        [
            f'STUDIO_HOME="{tmp_path / "studio"}"; mkdir -p "$STUDIO_HOME"',
            "_TORCH_CEILING=2.12.0; _TORCHVISION_CEILING=0.27.0; _TORCHAUDIO_CEILING=2.12.0; C_WARN=",
            'step() { echo "STEP $*"; }',
            'substep() { echo "SUB $1"; }',
            '_run_bounded() { shift 2; "$@"; }',
            _fn("_strip_index_url_credentials"),
            _LEAF_DEFS,
            "_second_vendor_torch_index_url() { echo https://download.pytorch.org/whl/rocm7.0; }",
            # uv venv: lay out bin/python as the stub python; uv pip install: succeed or fail.
            "run_install_cmd() { shift; [ \"$1 $2\" = 'uv venv' ] || return 1; "
            f'eval "_d=\\${{$#}}"; mkdir -p "$_d/bin"; ln -s "{wrapper}" "$_d/bin/python"; }}',
            f"run_install_cmd_retry() {{ return {pip_rc}; }}",
            _fn("_install_second_vendor_torch"),
            '_install_second_vendor_torch /main/bin/python https://download.pytorch.org/whl/cu128; echo "RC=$?"',
        ]
    )
    return _run(script)


def test_the_install_step_records_a_venv_the_helper_reads(tmp_path):
    out = _install(tmp_path, torch_version = "2.11.0+rocm7.0")
    assert "RC=0" in out.stdout, out
    studio = tmp_path / "studio"
    assert (studio / "unsloth_studio_rocm" / ".unsloth-second-torch-owned").is_file()
    record = json.loads((studio / RECORD_NAME).read_text())
    assert record["vendor"] == "rocm" and record["torch_version"] == "2.11.0+rocm7.0", record
    found = second_torch_venv(studio, main_vendor = "cuda")
    assert found is not None and found.vendor == "rocm"


def test_a_wrong_vendor_torch_is_not_recorded(tmp_path):
    """A cpu or CUDA wheel that slipped in from a mirror must not be handed out as the ROCm venv."""
    out = _install(tmp_path, torch_version = "2.11.0+cpu")
    assert "RC=0" in out.stdout and "not recording it" in out.stdout, out
    assert not (tmp_path / "studio" / RECORD_NAME).exists()


def test_a_failed_rebuild_drops_the_old_record_and_keeps_the_install_going(tmp_path):
    studio = tmp_path / "studio"
    studio.mkdir()
    (studio / RECORD_NAME).write_text('{"schema": 1}')
    out = _install(tmp_path, torch_version = "2.11.0+rocm7.0", pip_fails = True)
    assert "RC=0" in out.stdout and "main install is unaffected" in out.stdout, out
    assert not (studio / RECORD_NAME).exists()


def test_a_directory_the_installer_did_not_make_is_left_alone(tmp_path):
    keep = tmp_path / "studio" / "unsloth_studio_rocm" / "precious.txt"
    keep.parent.mkdir(parents = True)
    keep.write_text("mine")
    out = _install(tmp_path, torch_version = "2.11.0+rocm7.0")
    assert "RC=0" in out.stdout and "not made by this installer" in out.stdout, out
    assert keep.read_text() == "mine"


def _record(root: Path, **overrides) -> Path:
    py = root / "venv" / "bin" / "python"
    py.parent.mkdir(parents = True, exist_ok = True)
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    data = {
        "schema": 1,
        "vendor": "rocm",
        "python": str(py),
        "venv": str(root / "venv"),
        "torch_version": "2.11.0+rocm7.0",
    }
    data.update(overrides)
    (root / RECORD_NAME).write_text(json.dumps(data))
    return py


def test_the_helper_reads_a_valid_record(tmp_path):
    py = _record(tmp_path)
    found = second_torch_venv(tmp_path)
    assert found is not None
    assert (found.vendor, found.python, found.torch_version) == ("rocm", py, "2.11.0+rocm7.0")


def test_no_record_means_no_second_venv(tmp_path):
    assert second_torch_venv(tmp_path) is None


@pytest.mark.parametrize(
    "overrides",
    [{"schema": 2}, {"vendor": "xpu"}, {"python": None}, {"python": "/nonexistent/bin/python"}],
    ids = ["future-schema", "unknown-vendor", "no-python", "python-gone"],
)
def test_a_bad_record_is_ignored(tmp_path, overrides):
    _record(tmp_path, **overrides)
    assert second_torch_venv(tmp_path) is None


def test_a_malformed_record_is_ignored(tmp_path):
    (tmp_path / RECORD_NAME).write_text("{not json")
    assert second_torch_venv(tmp_path) is None


def test_a_record_for_the_main_venvs_own_vendor_is_stale(tmp_path):
    """The main venv was reinstalled for the AMD card; a ROCm 'second' venv is no longer the other one."""
    _record(tmp_path)
    assert second_torch_venv(tmp_path, main_vendor = "rocm") is None
    assert second_torch_venv(tmp_path, main_vendor = "cuda") is not None
