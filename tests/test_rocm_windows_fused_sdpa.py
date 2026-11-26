# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Import unsloth must disable exactly the SDPA backends that fail on Windows ROCm, and no others."""

import json
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")


def _on_windows_rocm():
    if sys.platform != "win32" or not torch.cuda.is_available():
        return False
    # Same detection as the fix: some AMD wheels tag only the torch version.
    from unsloth.import_fixes import _is_rocm_torch_build
    return bool(getattr(torch.version, "hip", None)) or _is_rocm_torch_build()


def _run(code, *args):
    p = subprocess.run(
        [sys.executable, "-c", code, *args], capture_output = True, text = True, timeout = 900
    )
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, p.stdout[-2000:] + p.stderr[-3000:]
    return json.loads(line[-1].split(" ", 1)[1])


def test_probe_is_windows_rocm_only():
    if _on_windows_rocm():
        pytest.skip(reason = "this host is the Windows ROCm one the probe targets")
    from unsloth.import_fixes import _rocm_windows_broken_sdpa_backends
    assert _rocm_windows_broken_sdpa_backends() == []


@pytest.mark.parametrize(
    "broken, expected",
    [
        ([], [True, True, True]),
        (["flash"], [False, True, True]),
        (["mem_efficient"], [True, False, True]),
        (["flash", "mem_efficient"], [False, False, True]),
    ],
)
def test_only_broken_backends_are_turned_off(monkeypatch, broken, expected):
    import unsloth.import_fixes as fixes

    cuda = torch.backends.cuda
    saved = (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled())
    monkeypatch.delenv("UNSLOTH_ALLOW_ROCM_FUSED_SDPA", raising = False)
    monkeypatch.setattr(fixes, "_rocm_windows_broken_sdpa_backends", lambda: list(broken))
    try:
        cuda.enable_flash_sdp(True)
        cuda.enable_mem_efficient_sdp(True)
        fixes.fix_rocm_windows_fused_sdpa()
        got = [cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled()]
    finally:
        cuda.enable_flash_sdp(saved[0])
        cuda.enable_mem_efficient_sdp(saved[1])
    assert got == expected


def test_opt_out_skips_the_probe(monkeypatch):
    import unsloth.import_fixes as fixes

    monkeypatch.setenv("UNSLOTH_ALLOW_ROCM_FUSED_SDPA", "1")
    monkeypatch.setattr(
        fixes,
        "_rocm_windows_broken_sdpa_backends",
        lambda: pytest.fail("probed despite the opt-out"),
    )
    fixes.fix_rocm_windows_fused_sdpa()


_PROBE_TWICE = textwrap.dedent(
    """
    import json
    import torch
    from unsloth.import_fixes import _rocm_windows_broken_sdpa_backends
    first = _rocm_windows_broken_sdpa_backends()
    with torch.inference_mode():  # an import inside inference_mode must probe the same
        second = _rocm_windows_broken_sdpa_backends()
    # Nothing may be left pending: a stale launch error would raise on this checked kernel.
    torch.ones(4, device = "cuda").float().sum().item()
    print("RESULT " + json.dumps([first, second]))
    """
)


def test_windows_rocm_probe_is_stable_and_leaves_no_pending_error():
    if not _on_windows_rocm():
        pytest.skip(reason = "needs a Windows ROCm GPU to reproduce the launch failure")
    first, second = _run(_PROBE_TWICE)
    assert first == second


_ATTENTION = textwrap.dedent(
    """
    import json
    import unsloth  # noqa: F401  (applies the import-time fixes)
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    g = torch.Generator(device = "cuda").manual_seed(0)
    # fp16: gfx10 claims bf16 it lacks, and fp16 exercises the same fused kernels.
    q, k, v = (torch.randn(2, 4, 24, 64, device = "cuda", dtype = torch.float16, generator = g)
               for _ in range(3))
    mask = torch.ones(24, 24, device = "cuda", dtype = torch.bool).tril()
    out = {}
    for name, grad, kw, nkv in (
        ("causal_fwd", False, {"is_causal": True}, 4),
        ("mask_fwd", False, {"attn_mask": mask}, 4),
        ("causal_bwd", True, {"is_causal": True}, 4),
        ("mask_bwd", True, {"attn_mask": mask}, 4),
        ("gqa_bwd", True, {"is_causal": True, "enable_gqa": True}, 2),
    ):
        def run(math):
            qq = q.clone().requires_grad_(grad)
            if math:
                with sdpa_kernel([SDPBackend.MATH]):
                    o = F.scaled_dot_product_attention(qq, k[:, :nkv], v[:, :nkv], **kw)
            else:
                o = F.scaled_dot_product_attention(qq, k[:, :nkv], v[:, :nkv], **kw)
            o = o.float()  # a checked launch: a failed attention launch raises here
            if grad:
                o.sum().backward()
                return o.detach(), qq.grad.float()
            return o, o
        try:
            o, d = run(False)
            ro, rd = run(True)
            out[name] = bool(torch.allclose(o, ro, atol = 2e-2, rtol = 2e-2)
                             and torch.allclose(d, rd, atol = 2e-2, rtol = 2e-2))
        except Exception as e:
            out[name] = f"{type(e).__name__}: {e}"[:200]
    print("RESULT " + json.dumps(out))
    """
)


def test_windows_rocm_attention_runs_after_import():
    if not _on_windows_rocm():
        pytest.skip(reason = "needs a Windows ROCm GPU to reproduce the launch failure")
    assert _run(_ATTENTION) == {
        "causal_fwd": True,
        "mask_fwd": True,
        "causal_bwd": True,
        "mask_bwd": True,
        "gqa_bwd": True,
    }


class _Props:
    def __init__(self, arch):
        self.gcnArchName = arch


def _mixed_host(
    monkeypatch,
    tmp_path,
    current,
    archs,
    arch_list,
    packs = None,
):
    """Pin a fake Windows ROCm host: torch device ``current`` among ``archs``, no real GPU read."""
    import unsloth.import_fixes as fixes

    root = tmp_path / "torch"
    root.mkdir()
    if packs is not None:
        (root / ".kpack").mkdir()
        for arch in packs:
            (root / ".kpack" / f"torch_{arch}.kpack").write_bytes(b"")
    monkeypatch.setattr(fixes.sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising = False)
    monkeypatch.setattr(torch, "__file__", str(root / "__init__.py"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(archs))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: _Props(archs[int(getattr(i, "index", i))])
    )
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arch_list))
    return fixes


def test_probe_skips_a_device_the_wheel_has_no_kernels_for(monkeypatch, tmp_path, caplog):
    # RX 5700 XT (gfx1010) is torch device 0; the installer picked the gfx1034 card's wheel.
    fixes = _mixed_host(monkeypatch, tmp_path, 0, ["gfx1010", "gfx1034"], ["gfx1030", "gfx1034"])
    # Any launch on the uncovered card is a native access violation, so the probe must not get there.
    monkeypatch.setattr(torch, "Generator", lambda *a, **k: pytest.fail("probed an uncovered GPU"))
    with caplog.at_level("WARNING"):
        assert fixes._rocm_windows_broken_sdpa_backends() == []
    assert "HIP_VISIBLE_DEVICES to 1" in caplog.text


def test_multi_arch_wheel_covers_only_installed_kernel_packs(monkeypatch, tmp_path):
    # Multi-arch wheels list gfx1010 in get_arch_list() but only ship the packs that were installed.
    fixes = _mixed_host(
        monkeypatch, tmp_path, 0, ["gfx1010", "gfx1034"], ["gfx1010", "gfx1034"], packs = ["gfx1034"]
    )
    device = torch.device("cuda", 0)
    assert fixes._rocm_device_lacks_torch_kernels(torch, device, "gfx1010")
    assert not fixes._rocm_device_lacks_torch_kernels(torch, device, "gfx1034:xnack-")


def test_coverage_fails_open_on_a_generic_arch_token(monkeypatch, tmp_path):
    fixes = _mixed_host(monkeypatch, tmp_path, 0, ["gfx1010"], ["gfx10-1-generic", "gfx1034"])
    assert not fixes._rocm_device_lacks_torch_kernels(torch, torch.device("cuda", 0), "gfx1010")
