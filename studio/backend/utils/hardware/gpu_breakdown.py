# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which part of each GPU's used VRAM is Unsloth's, and what inside Unsloth holds it.

Unsloth's share is measured per process (the backend plus every child it spawned:
llama-server, inference workers, training), never estimated. A device with no
per-process reading gets ``unsloth_gb = None`` and the System page keeps its single bar.
"""

from __future__ import annotations

import platform
import re
import subprocess
from typing import Any, Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)

_GIB = 1024**3
_PID_RE = re.compile(r"pid_(\d+)_", re.IGNORECASE)


def unsloth_pids() -> set[int]:
    import os

    pids = {os.getpid()}
    try:
        import psutil

        pids.update(p.pid for p in psutil.Process().children(recursive = True))
    except Exception:
        pass
    return pids


def parse_process_counter(lines: Iterable[str]) -> dict[tuple[int, int], float]:
    """``<instance>|<bytes>`` lines from ``GPU Process Memory`` -> {(pid, luid): bytes}."""
    from utils.hardware.hardware import _engine_instance_luid

    out: dict[tuple[int, int], float] = {}
    for line in lines:
        instance, sep, raw = line.strip().rpartition("|")
        if not sep:
            continue
        m = _PID_RE.match(instance.strip())
        luid = _engine_instance_luid(instance)
        if m is None or luid is None:
            continue
        try:
            used = float(raw)
        except ValueError:
            continue
        if used > 0:
            key = (int(m.group(1)), luid)
            out[key] = out.get(key, 0.0) + used
    return out


def _windows_process_vram() -> Optional[dict[tuple[int, int], float]]:
    ps = (
        "$s=(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage'"
        " -ErrorAction SilentlyContinue).CounterSamples;"
        "if($s){$s|ForEach-Object{'{0}|{1}' -f $_.InstanceName,[int64]$_.CookedValue}}"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
    except Exception as e:
        logger.debug("GPU Process Memory counter failed: %s", e)
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return parse_process_counter(r.stdout.splitlines())


def _windows_rocm_device_luids(devices: list[dict[str, Any]]) -> dict[int, int]:
    """{device index: LUID} from HIP itself; empty when HIP can't be asked."""
    from utils.hardware.hardware import _rocm_windows_hip_adapter_ids

    ordered = sorted(
        (d for d in devices if isinstance(d.get("visible_ordinal"), int)),
        key = lambda d: d["visible_ordinal"],
    )
    if not ordered:
        return {}
    ids = _rocm_windows_hip_adapter_ids(
        [d["visible_ordinal"] for d in ordered], [str(d.get("name") or "") for d in ordered]
    )
    if not ids or len(ids) != len(ordered):
        return {}
    return {d["index"]: luid for d, (luid, _node) in zip(ordered, ids)}


def parse_nvidia_compute_apps(
    apps_csv: str, gpus_csv: str
) -> dict[tuple[int, int], float]:
    """nvidia-smi compute-apps + gpu index/uuid CSVs -> {(pid, device index): bytes}."""
    index_by_uuid: dict[str, int] = {}
    for line in gpus_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            index_by_uuid[parts[1]] = int(parts[0])
    out: dict[tuple[int, int], float] = {}
    for line in apps_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        index = index_by_uuid.get(parts[1])
        try:
            mib = float(parts[2])
        except ValueError:
            continue  # "[N/A]" under WDDM
        if index is not None and mib > 0:
            key = (int(parts[0]), index)
            out[key] = out.get(key, 0.0) + mib * 1024**2
    return out


def _nvidia_process_vram() -> Optional[dict[tuple[int, int], float]]:
    def run(args: list[str]) -> Optional[str]:
        try:
            r = subprocess.run(
                ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                capture_output = True,
                text = True,
                timeout = 5,
            )
        except Exception:
            return None
        return r.stdout if r.returncode == 0 else None

    apps = run(["--query-compute-apps=pid,gpu_uuid,used_memory"])
    gpus = run(["--query-gpu=index,uuid"])
    if apps is None or gpus is None:
        return None
    return parse_nvidia_compute_apps(apps, gpus)


def process_vram_by_device(
    devices: list[dict[str, Any]], backend: str
) -> Optional[dict[tuple[int, int], float]]:
    """{(pid, device index): bytes} for every process, or None where this host can't say."""
    system = platform.system()
    if system == "Windows" and backend.lower().startswith("rocm"):
        by_luid = _windows_process_vram()
        luids = _windows_rocm_device_luids(devices) if by_luid else {}
        if not luids:
            return None
        index_by_luid = {luid: index for index, luid in luids.items()}
        out: dict[tuple[int, int], float] = {}
        for (pid, luid), used in by_luid.items():
            index = index_by_luid.get(luid)
            if index is not None:
                out[(pid, index)] = out.get((pid, index), 0.0) + used
        return out
    if system == "Linux" and backend.lower().startswith("cuda"):
        return _nvidia_process_vram()
    return None


def build_breakdown(
    devices: list[dict[str, Any]],
    process_bytes: Optional[dict[tuple[int, int], float]],
    pids: set[int],
    owners: dict[int, tuple[str, str]],
    components: list[tuple[int, int, str, str, str, float]],
) -> list[dict[str, Any]]:
    """One record per device: Unsloth's measured share, what holds it, and the rest.

    ``owners`` names a child pid ({pid: (kind, model)}); ``components`` are tensors a
    process holds itself ((pid, device index, kind, model, part, bytes)), e.g. a diffusion
    pipeline's transformer and VAE. Whatever a pid holds beyond those is "runtime".
    """
    result = []
    for device in devices:
        index = device.get("index")
        used_gb = device.get("vram_used_gb")
        record: dict[str, Any] = {
            "index": index,
            "unsloth_gb": None,
            "other_gb": None,
            "items": [],
        }
        if process_bytes is None or index is None:
            result.append(record)
            continue
        items: list[dict[str, Any]] = []
        unsloth = 0.0
        for pid in sorted(pids):
            held = process_bytes.get((pid, index), 0.0)
            if held <= 0:
                continue
            unsloth += held
            parts = [c for c in components if c[0] == pid and c[1] == index]
            if pid in owners:
                kind, model = owners[pid]
                items.append({"kind": kind, "model": model, "part": None, "gb": held / _GIB})
                continue
            named = 0.0
            for _pid, _idx, kind, model, part, size in parts:
                size = min(size, held - named)
                if size <= 0:
                    break
                items.append({"kind": kind, "model": model, "part": part, "gb": size / _GIB})
                named += size
            if held - named > 0:
                items.append({"kind": "runtime", "model": None, "part": None, "gb": (held - named) / _GIB})
        merged: dict[tuple[str, Optional[str], Optional[str]], float] = {}
        for item in items:
            key = (item["kind"], item["model"], item["part"])
            merged[key] = merged.get(key, 0.0) + item["gb"]
        record["items"] = [
            {"kind": k, "model": model, "part": part, "gb": round(gb, 3)}
            for (k, model, part), gb in sorted(merged.items(), key = lambda kv: -kv[1])
        ]
        record["unsloth_gb"] = round(unsloth / _GIB, 3)
        if isinstance(used_gb, (int, float)):
            record["other_gb"] = round(max(0.0, used_gb - unsloth / _GIB), 3)
        result.append(record)
    return result


def diffusion_components(
    pid: int, devices: list[dict[str, Any]]
) -> list[tuple[int, int, str, str, str, float]]:
    """Bytes of each loaded diffusion component resident on each device right now."""
    try:
        from core.inference.diffusion import _diffusion_backend
    except Exception:
        return []
    backend = _diffusion_backend
    state = getattr(backend, "_state", None) if backend is not None else None
    pipe = getattr(state, "pipe", None)
    if pipe is None:
        return []
    model = getattr(state, "display_repo_id", None) or getattr(state, "repo_id", None) or "Image model"
    model = str(model).rsplit("/", 1)[-1]
    index_by_ordinal = {
        d["visible_ordinal"]: d["index"]
        for d in devices
        if isinstance(d.get("visible_ordinal"), int) and d.get("index") is not None
    }
    out = []
    for name, module in (getattr(pipe, "components", None) or {}).items():
        tensors = getattr(module, "parameters", None)
        if tensors is None:
            continue
        sizes: dict[int, float] = {}
        try:
            for t in [*module.parameters(), *module.buffers()]:
                dev = t.device
                if dev.type == "cuda" and dev.index in index_by_ordinal:
                    sizes[dev.index] = sizes.get(dev.index, 0.0) + t.numel() * t.element_size()
        except Exception:
            continue
        for ordinal, size in sizes.items():
            out.append((pid, index_by_ordinal[ordinal], "image", model, name, size))
    return sorted(out, key = lambda c: -c[5])
