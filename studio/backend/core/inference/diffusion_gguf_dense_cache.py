# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep each GGUF linear's dequantised weight after its first forward on Apple MPS, so later steps and images skip
the per-step dequant. The kept tensor is the one the layer's own forward handed to ``F.linear`` (captured, not
recomputed), so every later output is bit-identical.

Memory: on MPS the llama.cpp-exact dequant computes each weight in fp32 (2x the kept bf16) beside uint8 code
temporaries, and the fill interleaves one long-lived weight per layer with that short-lived scratch, so the MPS
caching allocator pooled about one more kept-size of freed scratch (M5 Max: footprint +2x kept on torch 2.10,
+1.2-1.35x on 2.14.1). Each kept layer therefore hands the pool back with ``torch.mps.empty_cache()``, and the guard
prices a fill at the measured footprint per kept byte (``footprint_multiplier``) against available memory, keeping
max(8 GiB, 25% of RAM) free. Auto turns the cache off with a status reason when the whole priced fill does not fit.
UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE: auto (MPS only) / 0 / 1 (any device, fills layer by layer while it fits);
UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE_GB caps the kept bytes; ..._FOOTPRINT overrides the multiplier;
..._TRIM=0 keeps the pool (no empty_cache)."""

from __future__ import annotations

import os
import threading
import types
from typing import Any, Callable, Optional

ENV = "UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE"
CAP_ENV = "UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE_GB"
FOOTPRINT_ENV = "UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE_FOOTPRINT"
TRIM_ENV = "UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE_TRIM"
MODE_OFF, MODE_ON, MODE_AUTO = "off", "on", "auto"
_GIB = 1024**3
# Unified memory left free after a fill: decode, activations and other apps live there.
_RESERVE_MIN_BYTES = 8 * _GIB
_RESERVE_FRACTION = 0.25
_ENTRY = "_unsloth_gguf_dense"
_STATE_ATTR = "_unsloth_gguf_dense_state"
_FORWARD_ATTR = "forward_native"
# Process footprint added per kept byte on MPS, worst measured arm (M5 Max, task 0042, before the pool trim):
# torch 2.10 klein +18.3 GB / 7.36 GB kept = 2.49, Z-Image 2.18; torch 2.14.1 klein 1.36, Z-Image 1.22.
_MPS_FOOTPRINT = ((2, 14), 1.4)
_MPS_FOOTPRINT_OLD = 2.5
_WATERMARK_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"

COUNTS = {"installs": 0, "layers": 0, "cached": 0, "hits": 0, "declined": 0, "bytes": 0, "trims": 0}
# The last install's decision: state on / off / partial, why, and the priced numbers.
STATUS: dict = {}


def mode() -> str:
    raw = (os.environ.get(ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return MODE_OFF
    if raw in ("1", "on", "true", "yes", "force"):
        return MODE_ON
    return MODE_AUTO


def _cap_bytes() -> Optional[int]:
    raw = (os.environ.get(CAP_ENV) or "").strip()
    if not raw:
        return None
    try:
        return max(0, int(float(raw) * _GIB))
    except ValueError:
        return None


def footprint_multiplier(device: Any = "mps", torch_version: Optional[str] = None) -> float:
    """Footprint per kept byte: measured on MPS per torch version, 1.0 elsewhere; the env overrides."""
    raw = (os.environ.get(FOOTPRINT_ENV) or "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    if str(getattr(device, "type", device) or "").split(":", 1)[0] != "mps":
        return 1.0
    if torch_version is None:
        import torch
        torch_version = torch.__version__
    import re

    parts = re.match(r"\s*(\d+)\.(\d+)", str(torch_version))
    if parts is None:
        return _MPS_FOOTPRINT_OLD
    version = (int(parts.group(1)), int(parts.group(2)))
    return _MPS_FOOTPRINT[1] if version >= _MPS_FOOTPRINT[0] else _MPS_FOOTPRINT_OLD


def trim_enabled() -> bool:
    return (os.environ.get(TRIM_ENV) or "").strip().lower() not in ("0", "off", "false", "no")


def mps_budget_bytes(recommended: int, ratio: Optional[str] = None) -> int:
    """The MPS allocator's limit: the recommended working set, lowered by a high-watermark ratio below 1
    (``set_per_process_memory_fraction`` / the env); 0 means no limit, above 1 the working set binds first."""
    raw = os.environ.get(_WATERMARK_ENV) if ratio is None else ratio
    try:
        r = float(raw) if raw not in (None, "") else None
    except ValueError:
        r = None
    if r is None or r <= 0 or r >= 1:
        return int(recommended)
    return int(recommended * r)


def _total_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        return 0


def free_bytes() -> Optional[int]:
    """Unified memory a fill may still use: available RAM, capped by the Metal working set left and a chat
    worker's reserved growth. None when nothing can be read (then nothing is kept)."""
    readings = []
    try:
        import psutil
        readings.append(int(psutil.virtual_memory().available))
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        mps = getattr(torch, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            budget = mps_budget_bytes(int(mps.recommended_max_memory()))
            if budget > 0:
                readings.append(budget - int(mps.driver_allocated_memory()))
    except Exception:  # noqa: BLE001
        pass
    try:
        from core.inference.gpu_residency import shared_metal_headroom_bytes
        shared = shared_metal_headroom_bytes()
        if shared is not None:
            readings.append(int(shared))
    except Exception:  # noqa: BLE001
        pass
    return min(readings) if readings else None


def reserve_bytes(total: Optional[int] = None) -> int:
    total = _total_ram_bytes() if total is None else int(total)
    return max(_RESERVE_MIN_BYTES, int(total * _RESERVE_FRACTION))


def plan_fill(
    dense_bytes: int,
    *,
    available: Optional[int],
    total: Optional[int] = None,
    multiplier: float = 1.0,
    cap: Optional[int] = None,
    reserve: Optional[int] = None,
) -> dict:
    """Whether keeping every weight (``dense_bytes``) fits: ``available - dense_bytes * multiplier`` must stay at or
    above the reserve (max(8 GiB, 25% of ``total`` RAM)). Pure, so a smaller Mac can be emulated."""
    reserve = reserve_bytes(total) if reserve is None else int(reserve)
    want = dense_bytes if cap is None else min(dense_bytes, cap)
    priced = int(want * multiplier)
    plan = {
        "dense_bytes": int(dense_bytes),
        "priced_bytes": priced,
        "multiplier": multiplier,
        "available_bytes": available,
        "reserve_bytes": reserve,
        "total_bytes": total,
    }
    if available is None:
        return dict(plan, on = False, reason = "auto: available memory unreadable")
    left = available - priced
    plan["left_bytes"] = left
    if left < reserve:
        return dict(
            plan,
            on = False,
            reason = (
                f"auto: priced footprint {priced / _GIB:.1f} GiB ({want / _GIB:.1f} GiB kept x {multiplier:g}) would "
                f"leave {left / _GIB:.1f} GiB of {available / _GIB:.1f} GiB available, under the {reserve / _GIB:.1f} GiB "
                "reserve"
            ),
        )
    return dict(plan, on = True, reason = None)


class _State:
    """Shared by one transformer's layers: the byte count kept and the gate a fill must pass."""

    def __init__(
        self,
        *,
        cap: Optional[int],
        free: Callable[[], Optional[int]],
        reserve: int,
        multiplier: float = 1.0,
        dense_bytes: Optional[int] = None,
        total: Optional[int] = None,
        logger: Any = None,
    ) -> None:
        self.cap = cap
        self.free = free
        self.reserve = reserve
        self.multiplier = multiplier
        # set in auto: the whole priced fill must fit at the first forward, else nothing is kept
        self.dense_bytes = dense_bytes
        self.total = total
        self.logger = logger
        self.kept = 0
        self.full = False
        self.lock = threading.Lock()

    def _plan_once(self) -> bool:
        if self.dense_bytes is None:
            return True
        plan = plan_fill(
            self.dense_bytes,
            available = self.free(),
            total = self.total,
            multiplier = self.multiplier,
            cap = self.cap,
            reserve = self.reserve,
        )
        self.dense_bytes = None
        STATUS.update(plan, state = "on" if plan["on"] else "off")
        if self.logger is not None:
            if plan["on"]:
                self.logger.info(
                    "diffusion.gguf_dense_cache: filling (priced %.1f GiB of %.1f GiB available)",
                    plan["priced_bytes"] / _GIB,
                    (plan["available_bytes"] or 0) / _GIB,
                )
            else:
                self.logger.info("diffusion.gguf_dense_cache: off (%s)", plan["reason"])
        return plan["on"]

    def admit(self, nbytes: int) -> bool:
        with self.lock:
            if self.full:
                return False
            if not self._plan_once():
                self.full = True
                return False
            if self.cap is not None and self.kept + nbytes > self.cap:
                return self._stop("byte cap")
            free = self.free()
            if free is None or free - int(nbytes * self.multiplier) < self.reserve:
                return self._stop("reserve")
            self.kept += nbytes
            if STATUS.get("state") == "pending":
                STATUS["state"] = "on"
            return True

    def _stop(self, why: str) -> bool:
        self.full = True
        if self.kept:
            STATUS.update(
                state = "partial",
                reason = f"fill stopped at the {why} after {self.kept / _GIB:.2f} GiB",
            )
        else:
            STATUS.update(state = "off", reason = f"fill stopped at the {why} before any layer")
        return False


def _release_pool(device: Any) -> None:
    """Hands the fill's freed dequant scratch back to Metal, so the next layer's weight and scratch reuse it instead
    of the pool growing by about one kept-size (MPS only; one sync per kept layer, first forward only)."""
    if getattr(device, "type", None) != "mps" or not trim_enabled():
        return
    import torch

    torch.mps.empty_cache()
    COUNTS["trims"] += 1


def _storage_bytes(t: Any) -> int:
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return int(t.numel() * t.element_size())


def _capture_linear(call: Callable[[], Any]) -> tuple[Any, list]:
    """Runs ``call`` and records the arguments of every ``F.linear`` it makes (thread-local mode)."""
    import torch
    from torch.overrides import TorchFunctionMode

    seen: list = []
    linear = torch.nn.functional.linear

    class _Capture(TorchFunctionMode):
        def __torch_function__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            kwargs = kwargs or {}
            if func is linear or func is torch._C._nn.linear:
                seen.append((args, kwargs))
            return func(*args, **kwargs)

    with _Capture():
        out = call()
    return out, seen


_TABLE: dict = {}


def _dequant_table() -> Any:
    table = _TABLE.get("t")
    if table is None:
        try:
            from diffusers.quantizers.gguf import utils as gguf_utils
            table = getattr(gguf_utils, "dequantize_functions", None)
        except Exception:  # noqa: BLE001
            table = None
        _TABLE["t"] = table if isinstance(table, dict) else {}
        table = _TABLE["t"]
    return table


def _key(module: Any) -> tuple:
    """What the kept weight depends on: the weight / bias objects and versions, the compute dtype, and which class
    forward and dequant function produced it (a later install or uninstall of another dequant path re-captures)."""
    weight, bias = module.weight, module.bias
    return (
        type(module).__dict__.get(_FORWARD_ATTR),
        _dequant_table().get(getattr(weight, "quant_type", None)),
        id(weight),
        getattr(weight, "_version", None),
        id(bias) if bias is not None else None,
        getattr(bias, "_version", None) if bias is not None else None,
        module.compute_dtype,
    )


def _forward_native(self: Any, inputs: Any) -> Any:
    import torch

    stock = type(self).forward_native
    entry = self.__dict__.get(_ENTRY)
    if torch.compiler.is_compiling() or inputs.requires_grad:
        # compiled: the graph fuses the dequant itself; autograd: a kept inference tensor cannot be saved for backward
        return stock(self, inputs)
    if entry is not None:
        key, dense, bias = entry
        if key == _key(self) and dense.device == inputs.device:
            COUNTS["hits"] += 1
            return torch.nn.functional.linear(inputs, dense, bias)
        self.__dict__.pop(_ENTRY, None)
    state = self.__dict__.get(_STATE_ATTR)
    if state is None or state.full or self.__dict__.get("_unsloth_gguf_dense_declined"):
        return stock(self, inputs)
    out, seen = _capture_linear(lambda: stock(self, inputs))
    kept = False
    if len(seen) == 1:
        args, kwargs = seen[0]
        args = tuple(args) + tuple(kwargs.get(k) for k in ("weight", "bias") if k in kwargs)
        x = args[0] if args else None
        dense = args[1] if len(args) > 1 else None
        bias = args[2] if len(args) > 2 else None
        if (
            x is inputs
            and isinstance(dense, torch.Tensor)
            and dense is not self.weight
            and not dense.requires_grad
            and dense.device == inputs.device
        ):
            nbytes = _storage_bytes(dense)
            if state.admit(nbytes):
                self.__dict__[_ENTRY] = (_key(self), dense, bias)
                COUNTS["cached"] += 1
                COUNTS["bytes"] += nbytes
                kept = True
                _release_pool(dense.device)
    if not kept and not state.full:
        # a forward that does not reduce to one F.linear on its input never will
        self.__dict__["_unsloth_gguf_dense_declined"] = True
        COUNTS["declined"] += 1
    return out


def _gguf_linears(model: Any) -> list:
    try:
        from diffusers.quantizers.gguf.utils import GGUFLinear
    except Exception:  # noqa: BLE001 - no GGUF support in this diffusers
        return []
    found = []
    for module in getattr(model, "modules", lambda: ())():
        if not isinstance(module, GGUFLinear):
            continue
        weight = getattr(module, "weight", None)
        if getattr(weight, "quant_type", None) is None or getattr(weight, "dim", lambda: 0)() != 2:
            continue
        if getattr(module, "compute_dtype", None) is None:
            continue
        found.append(module)
    return found


def decline_reason(device: Any, requested: Optional[str] = None) -> Optional[str]:
    m = requested or mode()
    if m == MODE_OFF:
        return "off"
    kind = str(getattr(device, "type", device) or "").split(":", 1)[0]
    if m == MODE_AUTO and kind != "mps":
        return f"auto: {kind or 'unknown'} device (MPS only)"
    return None


def install(
    model: Any,
    *,
    device: Any,
    logger: Any = None,
    free: Optional[Callable[[], Optional[int]]] = None,
    reserve: Optional[int] = None,
) -> int:
    """Marks ``model``'s GGUF linears to keep their dequantised weight; returns how many. Idempotent."""
    requested = mode()
    reason = decline_reason(device, requested)
    if reason is not None:
        STATUS.clear()
        STATUS.update(state = "off", reason = reason)
        if logger is not None:
            logger.info("diffusion.gguf_dense_cache: off (%s)", reason)
        return 0
    layers = _gguf_linears(model)
    if not layers:
        return 0
    total = _total_ram_bytes()
    multiplier = footprint_multiplier(device)
    state = _State(
        cap = _cap_bytes(),
        free = free or free_bytes,
        reserve = reserve_bytes(total) if reserve is None else int(reserve),
        multiplier = multiplier,
        # auto prices the whole fill at the first forward (the pipeline is placed by then)
        dense_bytes = dense_bytes(layers) if requested == MODE_AUTO else None,
        total = total,
        logger = logger,
    )
    STATUS.clear()
    STATUS.update(state = "pending", reason = None, multiplier = multiplier, reserve_bytes = state.reserve)
    for module in layers:
        if isinstance(module.__dict__.get(_FORWARD_ATTR), types.MethodType):
            continue
        module.__dict__[_STATE_ATTR] = state
        module.__dict__[_FORWARD_ATTR] = types.MethodType(_forward_native, module)
    COUNTS["installs"] += 1
    COUNTS["layers"] += len(layers)
    if logger is not None:
        logger.info(
            "diffusion.gguf_dense_cache: on (%d GGUF linears keep their dequantised weight while %.1f GiB stays "
            "free, priced at %gx the kept bytes)",
            len(layers),
            state.reserve / _GIB,
            multiplier,
        )
    return len(layers)


def dense_bytes(layers: list) -> int:
    """Bytes the kept weights and biases take once every layer is filled, in each layer's compute dtype."""
    import torch

    total = 0
    for module in layers:
        size = torch.empty((), dtype = module.compute_dtype).element_size()
        total += int(module.in_features) * int(module.out_features) * size
        if module.bias is not None:
            total += int(module.out_features) * size
    return total


def uninstall(model: Any) -> int:
    """Drops every kept weight and restores the class forward. Returns how many layers it touched."""
    touched = 0
    for module in getattr(model, "modules", lambda: ())():
        d = getattr(module, "__dict__", {})
        if _STATE_ATTR not in d:
            continue
        for attr in (_ENTRY, _STATE_ATTR, _FORWARD_ATTR, "_unsloth_gguf_dense_declined"):
            d.pop(attr, None)
        touched += 1
    return touched


def kept_bytes(model: Any) -> int:
    total = 0
    for module in getattr(model, "modules", lambda: ())():
        entry = getattr(module, "__dict__", {}).get(_ENTRY)
        if entry is not None:
            total += _storage_bytes(entry[1])
    return total
