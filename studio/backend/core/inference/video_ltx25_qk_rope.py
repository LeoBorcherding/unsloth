# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.x fused q/k RMSNorm (across all heads) + split RoPE: one Triton launch per attention call.

diffusers' ``LTX2AudioVideoAttnProcessor`` / ``LTX2PerturbedAttnProcessor`` run ``norm_q`` / ``norm_k`` (RMSNorm over
``heads * head_dim``) and then ``apply_split_rotary_emb``, whose in-place ``addcmul_`` on slices makes Inductor emit ~3
pointwise kernels per site plus layout copies (~12% of the L4 DiT kernel time at 1280x704x121). The fused op reads
q / k once, keeps the normed value in float32 into the rotation (what Inductor emits, not eager's bf16 round trip),
rotates with the per-head float32 cos / sin, and writes bf16 ``[B, S, H, D]``, the layout the processor hands to
``dispatch_attention_fn``. Self-attention shares one cos / sin read between q and k.

Engaged on every RoPE site (video / audio self-attention, audio-to-video and video-to-audio cross-attention with
their separate q / k tables). The text cross-attention sites (norm, no RoPE) keep the stock chain. Off with
``UNSLOTH_LTX25_QK_ROPE=0``; anything unsupported (interleaved RoPE, non-bf16, CPU / MPS / ROCm, no Triton, grad,
unexpected shapes, a diffusers processor whose ``__call__`` differs from the audited one) runs the stock math.
``STATS`` counts calls per path (``fused`` / ``stock``) at run time, inside compiled graphs too.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap
import threading
from functools import lru_cache
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch
from .triton_closure_globals import publishes_closure_globals

QK_ROPE_ENV = "UNSLOTH_LTX25_QK_ROPE"
SPEED_OPTIM_KEY = "ltx2_fused_qk_rope"
_OP_NS = "unsloth_ltx2"
_OP_NAME = "qk_norm_rope"
_INSTALLED_FLAG = "_unsloth_ltx2_qk_rope"

# ast.dump of the stock ``__call__`` bodies this module's copies were taken from (diffusers 0.40.0.dev0 == 0.42.1).
KNOWN_CALL_FINGERPRINTS = {
    "LTX2AudioVideoAttnProcessor": frozenset({"7ea240cf62db1294"}),
    "LTX2PerturbedAttnProcessor": frozenset({"b30f8d7a159a57dc"}),
}

STATS = {"fused": 0, "stock": 0, "probe_failed": 0}
_LOCK = threading.Lock()
_VERDICTS: dict = {}

_LTX2: Any = None  # diffusers' transformer_ltx2 module, bound by processor_classes()

# Filled on first use with diffusers' ``dispatch_attention_fn``; a module global so Studio's dispatch routers
# (``install_dispatch_route``, sequence parallel) that rewrite module globals reach these processors too.
dispatch_attention_fn: Any = None


def qk_rope_enabled() -> bool:
    return (os.environ.get(QK_ROPE_ENV) or "1").strip().lower() not in ("0", "off", "false", "no")


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


# ---------------------------------------------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize = None)
@publishes_closure_globals
def _kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def _rne_bf16(x, EMU: tl.constexpr):
        # The CPU interpreter casts fp32 -> bf16 by truncation; round to nearest even first there (exact afterwards).
        if EMU:
            bits = x.to(tl.uint32, bitcast = True)
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
            x = bits.to(tl.float32, bitcast = True)
        return x.to(tl.bfloat16)

    @triton.jit
    def _norm_rows(
        x_ptr,
        w_ptr,
        off0,
        off1,
        eps,
        C: tl.constexpr,
        HAS_W: tl.constexpr,
        ROUND_NORM: tl.constexpr,
        EMU: tl.constexpr,
    ):
        x0 = tl.load(x_ptr + off0).to(tl.float32)
        x1 = tl.load(x_ptr + off1).to(tl.float32)
        ms = (tl.sum(tl.sum(x0 * x0, axis = 1), axis = 0) + tl.sum(tl.sum(x1 * x1, axis = 1), axis = 0)) / C
        rstd = tl.math.rsqrt(ms + eps)
        n0 = x0 * rstd
        n1 = x1 * rstd
        if HAS_W:
            n0 = n0 * tl.load(w_ptr + off0).to(tl.float32)
            n1 = n1 * tl.load(w_ptr + off1).to(tl.float32)
        if ROUND_NORM:  # eager's bf16 round trip between rms_norm and the rotation
            n0 = _rne_bf16(n0, EMU).to(tl.float32)
            n1 = _rne_bf16(n1, EMU).to(tl.float32)
        return n0, n1

    @triton.jit
    def _rope_store(o_ptr, off0, off1, n0, n1, cv, sv, EMU: tl.constexpr):
        # diffusers: out = x * cos; first.addcmul_(-sin, second_x); second.addcmul_(sin, first_x)  (float32)
        tl.store(o_ptr + off0, _rne_bf16(n0 * cv + (-sv) * n1, EMU))
        tl.store(o_ptr + off1, _rne_bf16(n1 * cv + sv * n0, EMU))

    @triton.jit
    def _ltx2_qk_norm_rope_kernel(
        q_ptr,
        k_ptr,
        qo_ptr,
        ko_ptr,
        qw_ptr,
        kw_ptr,
        qc_ptr,
        qs_ptr,
        kc_ptr,
        ks_ptr,
        Sq,
        Sk,
        q_sb,
        q_ss,
        k_sb,
        k_ss,
        qc_sb,
        qc_sh,
        qc_ss,
        qc_sr,
        kc_sb,
        kc_sh,
        kc_ss,
        kc_sr,
        eps,
        H: tl.constexpr,
        R: tl.constexpr,
        HAS_W: tl.constexpr,
        ROUND_NORM: tl.constexpr,
        EMU: tl.constexpr,
        SHARED: tl.constexpr,
    ):
        row = tl.program_id(0)
        which = tl.program_id(1)
        b = tl.program_id(2).to(tl.int64)
        C: tl.constexpr = H * 2 * R
        hh = tl.arange(0, H)[:, None]
        rr = tl.arange(0, R)[None, :]
        off0 = hh * (2 * R) + rr
        off1 = off0 + R
        if SHARED:
            # self-attention: one cos / sin read for the q row and the k row
            coff = b * qc_sb + row.to(tl.int64) * qc_ss + hh * qc_sh + rr * qc_sr
            cv = tl.load(qc_ptr + coff).to(tl.float32)
            sv = tl.load(qs_ptr + coff).to(tl.float32)
            n0, n1 = _norm_rows(
                q_ptr + b * q_sb + row.to(tl.int64) * q_ss,
                qw_ptr,
                off0,
                off1,
                eps,
                C,
                HAS_W,
                ROUND_NORM,
                EMU,
            )
            _rope_store(qo_ptr + (b * Sq + row) * C, off0, off1, n0, n1, cv, sv, EMU)
            n0, n1 = _norm_rows(
                k_ptr + b * k_sb + row.to(tl.int64) * k_ss,
                kw_ptr,
                off0,
                off1,
                eps,
                C,
                HAS_W,
                ROUND_NORM,
                EMU,
            )
            _rope_store(ko_ptr + (b * Sk + row) * C, off0, off1, n0, n1, cv, sv, EMU)
        else:
            if which == 0:
                x_ptr = q_ptr
                o_ptr = qo_ptr
                w_ptr = qw_ptr
                c_ptr = qc_ptr
                s_ptr = qs_ptr
                S = Sq
                sb = q_sb
                ss = q_ss
                cb = qc_sb
                ch = qc_sh
                cs = qc_ss
                cr = qc_sr
            else:
                x_ptr = k_ptr
                o_ptr = ko_ptr
                w_ptr = kw_ptr
                c_ptr = kc_ptr
                s_ptr = ks_ptr
                S = Sk
                sb = k_sb
                ss = k_ss
                cb = kc_sb
                ch = kc_sh
                cs = kc_ss
                cr = kc_sr
            if row < S:
                coff = b * cb + row.to(tl.int64) * cs + hh * ch + rr * cr
                cv = tl.load(c_ptr + coff).to(tl.float32)
                sv = tl.load(s_ptr + coff).to(tl.float32)
                n0, n1 = _norm_rows(
                    x_ptr + b * sb + row.to(tl.int64) * ss,
                    w_ptr,
                    off0,
                    off1,
                    eps,
                    C,
                    HAS_W,
                    ROUND_NORM,
                    EMU,
                )
                _rope_store(o_ptr + (b * S + row) * C, off0, off1, n0, n1, cv, sv, EMU)

    return _ltx2_qk_norm_rope_kernel


def _num_warps(C: int) -> int:
    return 8 if C >= 4096 else 4


def _launch(
    q: Any,
    k: Any,
    wq: Any,
    wk: Any,
    q_cos: Any,
    q_sin: Any,
    k_cos: Any,
    k_sin: Any,
    heads: int,
    eps: float,
    round_norm: bool = False,
    num_warps: int = 0,
) -> tuple:
    """q [B,Sq,H*D], k [B,Sk,H*D] bf16 (channel stride 1); cos / sin [B|1, H, S, D/2]. Returns contiguous
    ([B,Sq,H,D], [B,Sk,H,D]). ``k_cos is None`` = self-attention (k uses q's table; needs Sq == Sk)."""
    import torch

    B, Sq, C = q.shape
    Sk = k.shape[1]
    D = C // heads
    R = D // 2
    qo = torch.empty((B, Sq, heads, D), device = q.device, dtype = q.dtype)
    ko = torch.empty((B, Sk, heads, D), device = k.device, dtype = k.dtype)
    shared = k_cos is None
    if shared:
        k_cos, k_sin = q_cos, q_sin

    def strides(t: Any) -> list:
        return [t.stride(i) if t.shape[i] > 1 else 0 for i in range(4)]

    has_w = wq is not None
    _kernel()[(Sq, 1, B) if shared else (max(Sq, Sk), 2, B)](
        q,
        k,
        qo,
        ko,
        wq if has_w else q,
        wk if has_w else k,
        q_cos,
        q_sin,
        k_cos,
        k_sin,
        Sq,
        Sk,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        *strides(q_cos),
        *strides(k_cos),
        float(eps),
        H = heads,
        R = R,
        HAS_W = has_w,
        ROUND_NORM = bool(round_norm),
        EMU = _interpret(),
        SHARED = shared,
        num_warps = num_warps or _num_warps(C),
    )
    return qo, ko


# ---------------------------------------------------------------------------------------------------------------
# Reference (stock math) and support checks
# ---------------------------------------------------------------------------------------------------------------


def _ltx2_module() -> Any:
    from diffusers.models.transformers import transformer_ltx2
    return transformer_ltx2


def reference_qk_norm_rope(q, k, wq, wk, q_cos, q_sin, k_cos, k_sin, heads, eps) -> tuple:
    """The processor's chain: rms_norm over heads * head_dim, diffusers' split rotation, unflatten to heads."""
    import torch.nn.functional as F

    rope = _ltx2_module().apply_split_rotary_emb
    q = F.rms_norm(q, (q.shape[-1],), wq, eps)
    k = F.rms_norm(k, (k.shape[-1],), wk, eps)
    q = rope(q, (q_cos, q_sin))
    k = rope(k, (k_cos, k_sin) if k_cos is not None else (q_cos, q_sin))
    return q.unflatten(2, (heads, -1)), k.unflatten(2, (heads, -1))


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _table_ok(cos: Any, sin: Any, batch: int, heads: int, seq: int, half: int) -> bool:
    import torch
    return (
        isinstance(cos, torch.Tensor)
        and isinstance(sin, torch.Tensor)
        and cos.dim() == 4
        and tuple(cos.shape[1:]) == (heads, seq, half)
        and cos.shape[0] in (1, batch)
        and sin.shape == cos.shape
        and sin.stride() == cos.stride()
        and cos.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and sin.dtype == cos.dtype
        and cos.device == sin.device
    )


def supported(q: Any, k: Any, wq: Any, wk: Any, q_rope: Any, k_rope: Any, heads: int) -> bool:
    """Shape / dtype / device gate on tensor metadata only (traced by dynamo: no try / env / import in here). ROCm,
    no Triton and the env switch are decided at install time."""
    import torch

    if not (q.is_cuda and k.device == q.device):
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or q.dim() != 3 or k.dim() != 3:
        return False
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
        return False
    B, Sq, C = q.shape
    if k.shape[0] != B or k.shape[2] != C or q.stride(2) != 1 or k.stride(2) != 1 or C % heads:
        return False
    D = C // heads
    if D % 2 or not (_pow2(heads) and _pow2(D // 2)) or C > 8192:
        return False
    if (wq is None) != (wk is None):
        return False
    for w in (wq, wk):
        if w is not None and (
            tuple(w.shape) != (C,) or not w.is_contiguous() or w.device != q.device
        ):
            return False
    if not isinstance(q_rope, (tuple, list)) or len(q_rope) != 2:
        return False
    q_cos, q_sin = q_rope
    if not _table_ok(q_cos, q_sin, B, heads, Sq, D // 2) or q_cos.device != q.device:
        return False
    if k_rope is None:
        return Sq == k.shape[1]
    if not isinstance(k_rope, (tuple, list)) or len(k_rope) != 2:
        return False
    k_cos, k_sin = k_rope
    return _table_ok(k_cos, k_sin, B, heads, k.shape[1], D // 2) and k_cos.device == q.device


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _probe(device: Any, C: int, heads: int, table_dtype: Any, has_w: bool) -> bool:
    """One small launch vs a float64 oracle: the fused result is no further from it than the stock chain (+ slack)."""
    import torch

    g = torch.Generator(device = "cpu").manual_seed(7)
    D = C // heads
    S = 64
    q = (torch.randn(1, S, C, generator = g) * 2 + 0.3).to(torch.bfloat16).to(device)
    k = (torch.randn(1, S, C, generator = g) * 2 - 0.2).to(torch.bfloat16).to(device)
    w = (1 + 0.2 * torch.randn(C, generator = g)).to(torch.bfloat16).to(device) if has_w else None
    ang = (torch.rand(1, S, heads, D // 2, generator = g, dtype = torch.float64) * 2 - 1) * 60
    cos = ang.cos().to(table_dtype).to(device).swapaxes(1, 2)
    sin = ang.sin().to(table_dtype).to(device).swapaxes(1, 2)
    fq, fk = _launch(q, k, w, w, cos, sin, None, None, heads, 1e-6)
    rq, rk = reference_qk_norm_rope(q, k, w, w, cos, sin, None, None, heads, 1e-6)

    def oracle(x):
        xd = x.double()
        n = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim = True) + 1e-6)
        if w is not None:
            n = n * w.double()
        n = n.view(1, S, heads, D)
        c = cos.double().transpose(1, 2)
        s = sin.double().transpose(1, 2)
        a, b = n[..., : D // 2], n[..., D // 2 :]
        return torch.cat([a * c - b * s, b * c + a * s], -1)

    ok = True
    for f, r, x in ((fq, rq, q), (fk, rk, k)):
        o = oracle(x)
        err_f = (f.double() - o).abs().max().item()
        err_r = (r.double() - o).abs().max().item()
        ok &= bool(torch.isfinite(f).all().item()) and err_f <= err_r * 1.5 + 1e-3
    return ok


def verdict(q: Any, heads: int, table: Any, has_w: bool) -> Optional[bool]:
    """Probe cached per (device, C, heads, table dtype, weight); None while capturing a CUDA graph before it ran."""
    import torch

    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    key = (index, int(q.shape[-1]), int(heads), table.dtype, bool(has_w))
    found = _VERDICTS.get(key)
    if found is not None:
        return found
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            return None
    with _LOCK:
        if key not in _VERDICTS:
            try:
                with torch.cuda.device(q.device):
                    passed = _probe(
                        q.device, int(q.shape[-1]), int(heads), table.dtype, bool(has_w)
                    )
            except torch.OutOfMemoryError:
                raise
            except Exception:  # noqa: BLE001 -- a JIT / launch failure keeps the stock math
                passed = False
            if not passed:
                STATS["probe_failed"] += 1
            _VERDICTS[key] = passed
    return _VERDICTS[key]


def qk_norm_rope_impl(q, k, wq, wk, q_cos, q_sin, k_cos, k_sin, heads, eps) -> tuple:
    """The fused kernel when the device probe passed, the stock chain otherwise."""
    import torch

    out = None
    if (
        q.is_cuda
        and not is_rocm_torch(torch)
        and _triton_available()
        and verdict(q, heads, q_cos, wq is not None)
    ):
        try:
            out = _launch(q, k, wq, wk, q_cos, q_sin, k_cos, k_sin, heads, eps)
        except torch.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 -- a failed launch keeps the stock math
            out = None
    if out is not None:
        STATS["fused"] += 1
        return out
    STATS["stock"] += 1
    rq, rk = reference_qk_norm_rope(q, k, wq, wk, q_cos, q_sin, k_cos, k_sin, heads, eps)
    return rq.contiguous(), rk.contiguous()


@lru_cache(maxsize = None)
def qk_norm_rope_op() -> Any:
    """``torch.ops.unsloth_ltx2.qk_norm_rope``: opaque to dynamo / Inductor, registered once per process."""
    import torch

    @torch.library.custom_op(
        f"{_OP_NS}::{_OP_NAME}",
        mutates_args = (),
        schema = (
            "(Tensor q, Tensor k, Tensor? wq, Tensor? wk, Tensor q_cos, Tensor q_sin, Tensor? k_cos, "
            "Tensor? k_sin, int heads, float eps) -> (Tensor, Tensor)"
        ),
    )
    def qk_norm_rope(q, k, wq, wk, q_cos, q_sin, k_cos, k_sin, heads, eps):
        return qk_norm_rope_impl(q, k, wq, wk, q_cos, q_sin, k_cos, k_sin, heads, eps)

    @qk_norm_rope.register_fake
    def _(q, k, wq, wk, q_cos, q_sin, k_cos, k_sin, heads, eps):
        B, Sq, C = q.shape
        return q.new_empty((B, Sq, heads, C // heads)), k.new_empty(
            (k.shape[0], k.shape[1], heads, C // heads)
        )

    return qk_norm_rope


# ---------------------------------------------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------------------------------------------


def _norm_eps(norm: Any, dtype: Any) -> float:
    import torch
    eps = norm.eps
    return float(eps) if eps is not None else float(torch.finfo(dtype).eps)


def _plain_rms_norm(norm: Any) -> bool:
    import torch
    return type(norm) is torch.nn.RMSNorm and len(norm.normalized_shape) == 1


def norm_rope_unflatten(
    attn: Any, query: Any, key: Any, query_rotary_emb: Any, key_rotary_emb: Any
) -> tuple:
    """norm_q / norm_k, the RoPE and the unflatten to heads: the fused op where supported, else the stock chain.
    Every branch here reads module / tensor metadata only, so a fullgraph trace specialises it once per shape."""
    import torch

    if (
        query_rotary_emb is not None
        and attn.rope_type == "split"
        and _plain_rms_norm(attn.norm_q)
        and _plain_rms_norm(attn.norm_k)
        and supported(
            query,
            key,
            attn.norm_q.weight,
            attn.norm_k.weight,
            query_rotary_emb,
            key_rotary_emb,
            attn.heads,
        )
    ):
        eps = _norm_eps(attn.norm_q, query.dtype)
        if eps == _norm_eps(attn.norm_k, key.dtype):
            q_cos, q_sin = query_rotary_emb
            k_cos, k_sin = key_rotary_emb if key_rotary_emb is not None else (None, None)
            return torch.ops.unsloth_ltx2.qk_norm_rope(
                query,
                key,
                attn.norm_q.weight,
                attn.norm_k.weight,
                q_cos,
                q_sin,
                k_cos,
                k_sin,
                int(attn.heads),
                eps,
            )
    return _stock(attn, query, key, query_rotary_emb, key_rotary_emb)


def _stock(attn: Any, query: Any, key: Any, query_rotary_emb: Any, key_rotary_emb: Any) -> tuple:
    """Verbatim diffusers 0.42.1 ``LTX2AudioVideoAttnProcessor`` lines between to_q / to_k and the dispatch."""
    ltx2 = _LTX2
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if query_rotary_emb is not None:
        if attn.rope_type == "interleaved":
            query = ltx2.apply_interleaved_rotary_emb(query, query_rotary_emb)
            key = ltx2.apply_interleaved_rotary_emb(
                key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            )
        elif attn.rope_type == "split":
            query = ltx2.apply_split_rotary_emb(query, query_rotary_emb)
            key = ltx2.apply_split_rotary_emb(
                key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            )

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    return query, key


def _call_fingerprint(cls: Any) -> Optional[str]:
    try:
        src = textwrap.dedent(inspect.getsource(cls.__call__))
        fn = ast.parse(src).body[0]
        return hashlib.sha256(ast.dump(fn).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize = None)
def processor_classes() -> dict:
    """``{stock class name: fused subclass}`` for the stock processors whose ``__call__`` matches the audited body;
    empty when diffusers has no LTX-2 module."""
    global dispatch_attention_fn, _LTX2
    try:
        ltx2 = _ltx2_module()
        from diffusers.models.attention_dispatch import dispatch_attention_fn as stock_dispatch
    except Exception:  # noqa: BLE001
        return {}
    if dispatch_attention_fn is None:
        dispatch_attention_fn = stock_dispatch
    _LTX2 = ltx2
    import torch

    qk_norm_rope_op()  # register before any trace reaches torch.ops.unsloth_ltx2.qk_norm_rope

    out: dict = {}
    base = getattr(ltx2, "LTX2AudioVideoAttnProcessor", None)
    if (
        base is not None
        and _call_fingerprint(base) in KNOWN_CALL_FINGERPRINTS["LTX2AudioVideoAttnProcessor"]
    ):

        class FusedQKRopeLTX2AudioVideoAttnProcessor(base):
            """diffusers 0.42.1 ``LTX2AudioVideoAttnProcessor.__call__`` with the norm / RoPE / unflatten fused."""

            _unsloth_fused_qk_rope = True

            def __call__(
                self,
                attn,
                hidden_states,
                encoder_hidden_states = None,
                attention_mask = None,
                query_rotary_emb = None,
                key_rotary_emb = None,
            ):
                batch_size, sequence_length, _ = (
                    hidden_states.shape
                    if encoder_hidden_states is None
                    else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(
                        attention_mask, sequence_length, batch_size
                    )
                    attention_mask = attention_mask.view(
                        batch_size, attn.heads, -1, attention_mask.shape[-1]
                    )

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states

                if attn.to_gate_logits is not None:
                    gate_logits = attn.to_gate_logits(hidden_states)

                query = attn.to_q(hidden_states)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                query, key = norm_rope_unflatten(attn, query, key, query_rotary_emb, key_rotary_emb)
                value = value.unflatten(2, (attn.heads, -1))

                hidden_states = dispatch_attention_fn(
                    query,
                    key,
                    value,
                    attn_mask = attention_mask,
                    dropout_p = 0.0,
                    is_causal = False,
                    backend = self._attention_backend,
                    parallel_config = self._parallel_config,
                )
                hidden_states = hidden_states.flatten(2, 3)
                hidden_states = hidden_states.to(query.dtype)

                if attn.to_gate_logits is not None:
                    hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
                    gates = 2.0 * torch.sigmoid(gate_logits)
                    hidden_states = hidden_states * gates.unsqueeze(-1)
                    hidden_states = hidden_states.flatten(2, 3)

                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states

        out["LTX2AudioVideoAttnProcessor"] = FusedQKRopeLTX2AudioVideoAttnProcessor

    base = getattr(ltx2, "LTX2PerturbedAttnProcessor", None)
    if (
        base is not None
        and _call_fingerprint(base) in KNOWN_CALL_FINGERPRINTS["LTX2PerturbedAttnProcessor"]
    ):

        class FusedQKRopeLTX2PerturbedAttnProcessor(base):
            """diffusers 0.42.1 ``LTX2PerturbedAttnProcessor.__call__`` with the norm / RoPE / unflatten fused."""

            _unsloth_fused_qk_rope = True

            def __call__(
                self,
                attn,
                hidden_states,
                encoder_hidden_states = None,
                attention_mask = None,
                query_rotary_emb = None,
                key_rotary_emb = None,
                perturbation_mask = None,
                all_perturbed = None,
            ):
                batch_size, sequence_length, _ = (
                    hidden_states.shape
                    if encoder_hidden_states is None
                    else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(
                        attention_mask, sequence_length, batch_size
                    )
                    attention_mask = attention_mask.view(
                        batch_size, attn.heads, -1, attention_mask.shape[-1]
                    )

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states

                if attn.to_gate_logits is not None:
                    gate_logits = attn.to_gate_logits(hidden_states)

                value = attn.to_v(encoder_hidden_states)
                if all_perturbed is None:
                    all_perturbed = (
                        torch.all(perturbation_mask == 0)
                        if perturbation_mask is not None
                        else False
                    )

                if all_perturbed:
                    hidden_states = value
                else:
                    query = attn.to_q(hidden_states)
                    key = attn.to_k(encoder_hidden_states)

                    query, key = norm_rope_unflatten(
                        attn, query, key, query_rotary_emb, key_rotary_emb
                    )
                    value = value.unflatten(2, (attn.heads, -1))

                    hidden_states = dispatch_attention_fn(
                        query,
                        key,
                        value,
                        attn_mask = attention_mask,
                        dropout_p = 0.0,
                        is_causal = False,
                        backend = self._attention_backend,
                        parallel_config = self._parallel_config,
                    )
                    hidden_states = hidden_states.flatten(2, 3)
                    hidden_states = hidden_states.to(query.dtype)

                    if perturbation_mask is not None:
                        value = value.flatten(2, 3)
                        hidden_states = torch.lerp(value, hidden_states, perturbation_mask)

                if attn.to_gate_logits is not None:
                    hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
                    gates = 2.0 * torch.sigmoid(gate_logits)
                    hidden_states = hidden_states * gates.unsqueeze(-1)
                    hidden_states = hidden_states.flatten(2, 3)

                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states

        out["LTX2PerturbedAttnProcessor"] = FusedQKRopeLTX2PerturbedAttnProcessor
    return out


def install_ltx2_fused_qk_rope(
    pipe: Any,
    logger: Any = None,
    require_cuda: bool = True,
) -> int:
    """Swap the stock processor of every split-RoPE ``LTX2Attention`` in the pipe's LTX-2.x DiTs for its fused
    subclass (instance attributes such as ``_attention_backend`` carried over). Returns processors swapped; 0 when
    off, not LTX-2.x, off CUDA / on ROCm / without Triton, or on an unaudited diffusers processor."""
    if not qk_rope_enabled():
        return 0
    if require_cuda:
        try:
            import torch
            if not torch.cuda.is_available() or is_rocm_torch(torch) or not _triton_available():
                return 0
        except Exception:  # noqa: BLE001
            return 0
    try:
        from .diffusion_autoopt.targets import denoiser_dits
        dits = denoiser_dits(pipe)
    except Exception:  # noqa: BLE001
        dits = [getattr(pipe, "transformer", None)]
    classes = processor_classes()
    if not classes:
        return 0
    count = 0
    unaudited = set()
    for dit in dits:
        if dit is None or type(dit).__name__ != "LTX2VideoTransformer3DModel":
            continue
        for module in dit.modules():
            if (
                type(module).__name__ != "LTX2Attention"
                or getattr(module, "rope_type", None) != "split"
            ):
                continue
            proc = getattr(module, "processor", None)
            if proc is None:
                continue
            if getattr(type(proc), "_unsloth_fused_qk_rope", False):
                count += 1
                continue
            fused_cls = classes.get(type(proc).__name__)
            if fused_cls is None or type(proc) is not fused_cls.__mro__[1]:
                unaudited.add(type(proc).__name__)
                continue
            new = fused_cls.__new__(fused_cls)
            new.__dict__.update(proc.__dict__)
            try:
                module.set_processor(new)
            except Exception:  # noqa: BLE001
                module.processor = new
            count += 1
        setattr(dit, _INSTALLED_FLAG, True)
    if logger is not None:
        if count:
            logger.info(
                f"video.ltx2: fused q/k RMSNorm + split RoPE on {count} attention processors"
            )
        if unaudited:
            logger.info(
                f"video.ltx2: fused q/k RoPE skipped unaudited processors {sorted(unaudited)}"
            )
    return count


def engagement_stats() -> dict:
    """Copy of ``STATS``: ``fused`` / ``stock`` op calls since process start (compiled graphs included)."""
    return dict(STATS)
