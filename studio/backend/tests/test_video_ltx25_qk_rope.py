# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.x fused q/k RMSNorm + split RoPE: processor parity with diffusers (CPU = stock path inside the fused
processor), gating / fallbacks, the custom op's schema and fake, and the Triton kernel under the CPU interpreter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from core.inference import video_ltx25_qk_rope as Q

ltx2 = pytest.importorskip("diffusers.models.transformers.transformer_ltx2")

BACKEND = Path(__file__).resolve().parents[1]


def _rope(S, heads, half, seed):
    g = torch.Generator().manual_seed(seed)
    ang = (torch.rand(1, S, heads, half, generator = g, dtype = torch.float64) * 2 - 1) * 40
    return ang.cos().float().swapaxes(1, 2), ang.sin().float().swapaxes(1, 2)


def _attn(
    query_dim,
    heads,
    dim_head,
    cross = None,
    rope_type = "split",
    perturbed = False,
    seed = 0,
):
    torch.manual_seed(seed)
    proc = ltx2.LTX2PerturbedAttnProcessor() if perturbed else ltx2.LTX2AudioVideoAttnProcessor()
    a = ltx2.LTX2Attention(
        query_dim = query_dim,
        heads = heads,
        kv_heads = heads,
        dim_head = dim_head,
        cross_attention_dim = cross,
        rope_type = rope_type,
        apply_gated_attention = True,
        processor = proc,
    )
    with torch.no_grad():
        a.norm_q.weight.copy_(1 + 0.2 * torch.randn_like(a.norm_q.weight))
        a.norm_k.weight.copy_(1 + 0.2 * torch.randn_like(a.norm_k.weight))
    return a.to(torch.bfloat16).eval()


class LTX2VideoTransformer3DModel(torch.nn.Module):  # name-matched stand-in for the DiT
    def __init__(self, *mods):
        super().__init__()
        self.mods = torch.nn.ModuleList(mods)


class _Pipe:
    def __init__(self, dit):
        self.transformer = dit


def _fused_copy(a):
    import copy

    b = copy.deepcopy(a)
    assert (
        Q.install_ltx2_fused_qk_rope(_Pipe(LTX2VideoTransformer3DModel(b)), require_cuda = False) == 1
    )
    assert getattr(type(b.processor), "_unsloth_fused_qk_rope", False)
    return b


def test_audited_processors_present():
    classes = Q.processor_classes()
    assert set(classes) == {"LTX2AudioVideoAttnProcessor", "LTX2PerturbedAttnProcessor"}
    assert issubclass(classes["LTX2AudioVideoAttnProcessor"], ltx2.LTX2AudioVideoAttnProcessor)
    assert hasattr(torch.ops.unsloth_ltx2, "qk_norm_rope")


@pytest.mark.parametrize("perturbed", [False, True])
def test_processor_parity_cpu(perturbed):
    heads, dim_head, dim = 4, 16, 96
    S, Sa, St = 10, 6, 7
    a_self = _attn(dim, heads, dim_head, perturbed = perturbed)
    a_cross = _attn(dim, heads, dim_head, cross = 48, perturbed = perturbed, seed = 1)
    a_text = _attn(dim, heads, dim_head, cross = 40, perturbed = perturbed, seed = 2)
    f_self, f_cross, f_text = _fused_copy(a_self), _fused_copy(a_cross), _fused_copy(a_text)
    x = torch.randn(1, S, dim).to(torch.bfloat16)
    audio = torch.randn(1, Sa, 48).to(torch.bfloat16)
    text = torch.randn(1, St, 40).to(torch.bfloat16)
    rq = _rope(S, heads, dim_head // 2, 3)
    rk = _rope(Sa, heads, dim_head // 2, 4)
    bias = torch.zeros(1, St).to(torch.bfloat16)
    bias[:, 5:] = -10000.0
    # STG perturbs self-attention only (the block passes perturbation_mask to attn1 / audio_attn1)
    extra = (
        {"perturbation_mask": torch.tensor([1.0]).view(1, 1, 1).to(torch.bfloat16)}
        if perturbed
        else {}
    )
    before = dict(Q.STATS)
    with torch.no_grad():
        cases = [
            (a_self, f_self, dict(hidden_states = x, query_rotary_emb = rq, **extra)),
            (
                a_cross,
                f_cross,
                dict(
                    hidden_states = x,
                    encoder_hidden_states = audio,
                    query_rotary_emb = rq,
                    key_rotary_emb = rk,
                ),
            ),
            (
                a_text,
                f_text,
                dict(hidden_states = x, encoder_hidden_states = text, attention_mask = bias),
            ),
        ]
        for stock, fused, kw in cases:
            assert torch.equal(stock(**kw), fused(**kw))
        if perturbed:
            kw = dict(hidden_states = x, query_rotary_emb = rq, all_perturbed = True)
            assert torch.equal(a_self(**kw), f_self(**kw))
    # CPU: the gate keeps the stock chain in Python; the op never runs
    assert Q.STATS == before


def test_op_cpu_falls_back_and_opcheck():
    heads, D = 4, 16
    q = torch.randn(1, 9, heads * D).to(torch.bfloat16)
    k = torch.randn(1, 9, heads * D).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(heads * D)).to(torch.bfloat16)
    c, s = _rope(9, heads, D // 2, 5)
    Q.processor_classes()
    before = Q.STATS["stock"]
    oq, ok = torch.ops.unsloth_ltx2.qk_norm_rope(q, k, w, w, c, s, None, None, heads, 1e-6)
    assert Q.STATS["stock"] == before + 1
    rq, rk = Q.reference_qk_norm_rope(q, k, w, w, c, s, None, None, heads, 1e-6)
    assert (
        oq.shape == (1, 9, heads, D)
        and oq.is_contiguous()
        and torch.equal(oq, rq)
        and torch.equal(ok, rk)
    )
    torch.library.opcheck(
        torch.ops.unsloth_ltx2.qk_norm_rope.default,
        (q, k, w, w, c, s, None, None, heads, 1e-6),
        test_utils = ("test_schema", "test_faketensor"),
    )


def test_gates():
    heads, D = 4, 16
    q = torch.randn(1, 9, heads * D).to(torch.bfloat16)
    r = _rope(9, heads, D // 2, 6)
    w = torch.ones(heads * D, dtype = torch.bfloat16)
    assert not Q.supported(q, q, w, w, r, None, heads)  # CPU
    meta = q.to("meta")
    assert not Q.supported(meta, meta, w, w, r, None, heads)
    # interleaved and env-off never install
    a = _attn(64, heads, D, rope_type = "interleaved")
    assert (
        Q.install_ltx2_fused_qk_rope(_Pipe(LTX2VideoTransformer3DModel(a)), require_cuda = False) == 0
    )
    assert not getattr(type(a.processor), "_unsloth_fused_qk_rope", False)


def test_env_off_and_cuda_required(monkeypatch):
    a = _attn(64, 4, 16)
    monkeypatch.setenv(Q.QK_ROPE_ENV, "0")
    assert (
        Q.install_ltx2_fused_qk_rope(_Pipe(LTX2VideoTransformer3DModel(a)), require_cuda = False) == 0
    )
    monkeypatch.delenv(Q.QK_ROPE_ENV)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert (
        Q.install_ltx2_fused_qk_rope(_Pipe(LTX2VideoTransformer3DModel(a)), require_cuda = True) == 0
    )
    assert not getattr(type(a.processor), "_unsloth_fused_qk_rope", False)


def test_unaudited_processor_left_alone(monkeypatch):
    a = _attn(64, 4, 16)
    monkeypatch.setattr(Q, "processor_classes", lambda: {"SomethingElse": object})
    assert (
        Q.install_ltx2_fused_qk_rope(_Pipe(LTX2VideoTransformer3DModel(a)), require_cuda = False) == 0
    )
    assert type(a.processor) is ltx2.LTX2AudioVideoAttnProcessor


def test_backend_attribute_carried_over():
    a = _attn(64, 4, 16)
    a.processor._attention_backend = "native"
    b = _fused_copy(a)
    assert b.processor._attention_backend == "native"


def test_fullgraph_compile_cpu():
    a = _fused_copy(_attn(64, 4, 16))
    x = torch.randn(1, 8, 64).to(torch.bfloat16)
    r = _rope(8, 4, 8, 7)
    compiled = torch.compile(a, fullgraph = True, dynamic = False, backend = "eager")
    with torch.no_grad():
        assert torch.equal(
            compiled(hidden_states = x, query_rotary_emb = r), a(hidden_states = x, query_rotary_emb = r)
        )


_INTERPRET_SCRIPT = textwrap.dedent(
    """
    import json, torch
    import torch.nn.functional as F
    from core.inference import video_ltx25_qk_rope as Q
    from diffusers.models.transformers.transformer_ltx2 import apply_split_rotary_emb

    def rope_ref(x, heads, rope, round_norm, w, eps=1e-6):
        # Inductor-like (round_norm=False: normed value kept in fp32) or eager (bf16 round trip)
        xf = x.float()
        n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        n = n.to(torch.bfloat16) if round_norm else n
        out = apply_split_rotary_emb(n, rope).to(torch.bfloat16)
        return out.unflatten(2, (heads, -1))

    res = {}
    g = torch.Generator().manual_seed(0)
    for name, heads, D, Sq, Sk, cross in (("video_self", 32, 128, 6, 6, False), ("a2v", 32, 64, 5, 3, True),
                                          ("v2a", 32, 64, 3, 5, True)):
        C = heads * D
        q = (torch.randn(1, Sq, C, generator=g) * 2).to(torch.bfloat16)
        k = (torch.randn(1, Sk, C, generator=g) * 2).to(torch.bfloat16)
        wq = (1 + 0.2 * torch.randn(C, generator=g)).to(torch.bfloat16)
        wk = (1 + 0.2 * torch.randn(C, generator=g)).to(torch.bfloat16)
        def tab(S):
            a = (torch.rand(1, S, heads, D // 2, generator=g, dtype=torch.float64) * 2 - 1) * 40
            return a.cos().float().swapaxes(1, 2), a.sin().float().swapaxes(1, 2)
        rq = tab(Sq)
        rk = tab(Sk) if cross else None
        for rn in (False, True):
            oq, ok = Q._launch(q, k, wq, wk, *rq, *(rk if rk else (None, None)), heads, 1e-6, round_norm=rn)
            eq = rope_ref(q, heads, rq, rn, wq)
            ek = rope_ref(k, heads, rk if rk else rq, rn, wk)
            res[f"{name}_rn{int(rn)}"] = [float((oq.float() != eq.float()).float().mean()),
                                          float((ok.float() != ek.float()).float().mean()),
                                          float((oq.float() - eq.float()).abs().max()),
                                          float((ok.float() - ek.float()).abs().max())]
    print("RESULT" + json.dumps(res))
    """
)


def test_kernel_triton_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, "-c", _INTERPRET_SCRIPT],
        cwd = BACKEND,
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT")]
    assert proc.returncode == 0 and line, proc.stderr[-3000:]
    res = json.loads(line[0][len("RESULT") :])
    for key, (fq, fk, mq, mk) in res.items():
        # fp32 reduction-order / rsqrt flips of a single bf16 rounding only
        assert fq < 2e-3 and fk < 2e-3, (key, res[key])
        assert mq <= 0.07 and mk <= 0.07, (key, res[key])


_PROCESSOR_INTERPRET_SCRIPT = textwrap.dedent(
    """
    import copy, json, sys, torch
    sys.path.insert(0, "tests")
    import test_video_ltx25_qk_rope as T
    from core.inference import video_ltx25_qk_rope as Q

    # route the op to the kernel on CPU (interpreter) and let the gate pass off CUDA
    Q.supported = lambda *a: True
    def impl(q, k, wq, wk, qc, qs, kc, ks, heads, eps):
        Q.STATS["fused"] += 1
        return Q._launch(q, k, wq, wk, qc, qs, kc, ks, heads, eps)
    Q.qk_norm_rope_impl = impl

    heads, D, dim = 32, 8, 96
    S, Sa = 10, 6
    res = {}
    for perturbed in (False, True):
        a_self = T._attn(dim, heads, D, perturbed=perturbed)
        a_cross = T._attn(dim, heads, D, cross=48, perturbed=perturbed, seed=1)
        f_self, f_cross = T._fused_copy(a_self), T._fused_copy(a_cross)
        x = torch.randn(1, S, dim).to(torch.bfloat16)
        audio = torch.randn(1, Sa, 48).to(torch.bfloat16)
        rq, rk, ra = T._rope(S, heads, D // 2, 3), T._rope(Sa, heads, D // 2, 4), T._rope(Sa, heads, D // 2, 5)
        cases = {
            "self": (a_self, f_self, dict(hidden_states=x, query_rotary_emb=rq)),
            "a2v": (a_cross, f_cross, dict(hidden_states=x, encoder_hidden_states=audio, query_rotary_emb=rq,
                                           key_rotary_emb=rk)),
        }
        with torch.no_grad():
            for name, (stock, fused, kw) in cases.items():
                before = Q.STATS["fused"]
                ref = stock(**kw).float()
                got = fused(**kw).float()
                comp = torch.compile(fused, fullgraph=True, dynamic=False, backend="eager")(**kw).float()
                scale = ref.abs().max().item()
                res[f"{name}_p{int(perturbed)}"] = [(got - ref).abs().max().item() / scale,
                                                    (comp - got).abs().max().item(),
                                                    Q.STATS["fused"] - before]
    print("RESULT" + json.dumps(res))
    """
)


def test_processor_fused_path_triton_interpreter():
    """The fused processor with the op really launched (interpreter on CPU), eager and fullgraph-traced, vs stock."""
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, "-c", _PROCESSOR_INTERPRET_SCRIPT],
        cwd = BACKEND,
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT")]
    assert proc.returncode == 0 and line, proc.stderr[-3000:]
    res = json.loads(line[0][len("RESULT") :])
    for key, (rel, compiled_diff, launches) in res.items():
        assert rel < 2e-2, (
            key,
            res[key],
        )  # bf16 rounding of the normed value only (eager rounds, the op does not)
        assert compiled_diff == 0.0, (key, res[key])
        assert launches == 2, (key, res[key])  # eager call + compiled call, one launch each
