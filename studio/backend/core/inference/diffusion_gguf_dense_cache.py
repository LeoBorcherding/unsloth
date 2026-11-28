# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep each GGUF linear's dequantised weight after its first forward on Apple MPS, so later steps and images skip
the per-step dequant. The kept tensor is the one the layer's own forward handed to ``F.linear`` (captured, not
recomputed), so every later output is bit-identical. Fills while unified memory keeps a reserve free; layers past it
stay per-step. UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE: auto (MPS only) / 0 / 1 (any device);
UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE_GB caps the kept bytes."""

from __future__ import annotations

import os
import threading
import types
from typing import Any, Callable, Optional

ENV = "UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE"
CAP_ENV = "UNSLOTH_DIFFUSION_GGUF_DENSE_CACHE_GB"
MODE_OFF, MODE_ON, MODE_AUTO = "off", "on", "auto"
_GIB = 1024**3
# Unified memory left free after a fill: decode, activations and other apps live there.
_RESERVE_MIN_BYTES = 8 * _GIB
_RESERVE_FRACTION = 0.25
_ENTRY = "_unsloth_gguf_dense"
_STATE_ATTR = "_unsloth_gguf_dense_state"
_FORWARD_ATTR = "forward_native"

COUNTS = {"installs": 0, "layers": 0, "cached": 0, "hits": 0, "declined": 0, "bytes": 0}


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
            budget = int(mps.recommended_max_memory())
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


class _State:
    """Shared by one transformer's layers: the byte count kept and the gate a fill must pass."""

    def __init__(
        self, *, cap: Optional[int], free: Callable[[], Optional[int]], reserve: int
    ) -> None:
        self.cap = cap
        self.free = free
        self.reserve = reserve
        self.kept = 0
        self.full = False
        self.lock = threading.Lock()

    def admit(self, nbytes: int) -> bool:
        with self.lock:
            if self.full:
                return False
            if self.cap is not None and self.kept + nbytes > self.cap:
                self.full = True
                return False
            free = self.free()
            if free is None or free - nbytes < self.reserve:
                self.full = True
                return False
            self.kept += nbytes
            return True


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
    reason = decline_reason(device)
    if reason is not None:
        if logger is not None:
            logger.info("diffusion.gguf_dense_cache: off (%s)", reason)
        return 0
    layers = _gguf_linears(model)
    if not layers:
        return 0
    state = _State(
        cap = _cap_bytes(),
        free = free or free_bytes,
        reserve = reserve_bytes() if reserve is None else int(reserve),
    )
    for module in layers:
        if isinstance(module.__dict__.get(_FORWARD_ATTR), types.MethodType):
            continue
        module.__dict__[_STATE_ATTR] = state
        module.__dict__[_FORWARD_ATTR] = types.MethodType(_forward_native, module)
    COUNTS["installs"] += 1
    COUNTS["layers"] += len(layers)
    if logger is not None:
        logger.info(
            "diffusion.gguf_dense_cache: on (%d GGUF linears keep their dequantised weight while %.1f GiB stays free)",
            len(layers),
            state.reserve / _GIB,
        )
    return len(layers)


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
