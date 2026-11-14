# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Locating llama-server binaries, loader paths, ROCm libraries and runtime probes."""

import contextlib
import os
import re
import shutil
import sys
import threading

from pathlib import Path
from typing import Optional
from .log import (
    logger,
)


def _wsl_system_rocm_lib_dirs() -> "list[str]":
    """Bundled HIP segfaults on WSL /dev/dxg, so system ROCm libs load first. Mirrors install_llama_prebuilt; no-op off a ROCDXG WSL host."""
    try:
        if not os.path.exists("/dev/dxg"):
            return []
        with open("/proc/version", encoding = "utf-8", errors = "replace") as fh:
            if "microsoft" not in fh.read().lower():
                return []
    except (OSError, UnicodeDecodeError):
        return []
    out: "list[str]" = []
    for d in ("/opt/rocm/lib", "/opt/rocm/lib64"):
        if os.path.exists(os.path.join(d, "librocdxg.so")) or os.path.exists(
            os.path.join(d, "librocdxg.so.1")
        ):
            out.append(d)
    return out


def _bundled_hip_present(binary_dir: str) -> bool:
    """True when a prebuilt bundle ships its own HIP backend library."""
    if not binary_dir:
        return False
    try:
        # Glob libggml-hip.so* to match versioned suffixes, the same way the installer's health check does.
        return any(Path(str(binary_dir)).glob("libggml-hip.so*"))
    except OSError:
        return False


def _native_linux_system_rocm_lib_dirs(binary_dir: str = "") -> "list[str]":
    """Prepend whole system ROCm dir: bundled HIP can crash in hsa_init and mixing risks missing symbols. UNSLOTH_LLAMA_NO_SYSTEM_ROCM=1 skips it."""
    if os.environ.get("UNSLOTH_LLAMA_NO_SYSTEM_ROCM") == "1":
        return []
    if sys.platform != "linux" or os.path.exists("/dev/dxg"):
        return []
    if not os.path.exists("/dev/kfd"):
        return []
    if not _bundled_hip_present(binary_dir):
        return []
    # Env ROCm roots first so a stale /opt/rocm does not shadow them.
    candidates = []
    for var in ("HIP_PATH", "HIP_PATH_57", "ROCM_PATH"):
        val = os.environ.get(var)
        if val:
            candidates.append(val)
    candidates.append("/opt/rocm")
    out: "list[str]" = []
    seen: "set[str]" = set()
    for base in candidates:
        for lib_sub in ("lib", "lib64"):
            d = os.path.join(base, lib_sub)
            if d in seen:
                continue
            seen.add(d)
            if os.path.exists(os.path.join(d, "libhsa-runtime64.so")) or os.path.exists(
                os.path.join(d, "libhsa-runtime64.so.1")
            ):
                out.append(d)
                # ROCm keeps LLVM under root/lib/llvm even on lib64 hosts, so probe lib too, ahead of the bundle, else libamd_comgr binds the bundle libLLVM.
                for _sub in (lib_sub, "lib"):
                    llvm_lib = os.path.join(base, _sub, "llvm", "lib")
                    if llvm_lib not in seen and os.path.isdir(llvm_lib):
                        seen.add(llvm_lib)
                        out.append(llvm_lib)
    return out


# Matches llama.cpp's --fit margin; an iGPU's VRAM is shared system RAM.
_IGPU_HOST_RESERVE_MIB = 1024
_HOST_RAM_HEADROOM_MIB = 2048
# The host keeps the larger of a GB floor or a RAM share, else the desktop becomes unusable.
_CARVEOUT_ADVICE_MIN_HOST_GB = 8
# A fifth, not a quarter: 128 GB Strix Halo firmware offers and runs 96 GB.
_CARVEOUT_ADVICE_HOST_FRACTION = 0.20
# Drivers report the kept pool (95.83 for a 96 GB setting); slack avoids advising the current rung.
_CARVEOUT_NOMINAL_SLACK_GB = 0.5
# One string so the three call sites describe the override the same way.
_PAGEABLE_OVERRIDE_NOTE = (
    " This model was set to load without memory mapping, which reads every byte into "
    "RAM up front and cannot complete at this size, so Unsloth loaded it with memory "
    "mapping instead."
)
# The override is reported, not undone: the running server is already memory-mapped.
_PAGEABLE_OVERRIDE_TEXT_ONLY_NOTICE = (
    "This model was set to load without memory mapping, and with its vision projector "
    "in system RAM it was too large to load that way, so Unsloth loaded it with memory "
    "mapping instead. The projector then failed to start and this session is text-only, "
    "which the weights alone would have fit. The running server keeps the memory "
    "mapping it started with; load the model again to run it as you asked."
)
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"


def _apply_igpu_host_reserve_mib(free_mib: int, is_igpu: bool) -> int:
    """An iGPU's free VRAM is system RAM, so reserve the --fit margin; discrete cards are never touched."""
    if not is_igpu:
        return free_mib
    return max(0, free_mib - _IGPU_HOST_RESERVE_MIB)


def _resolve_llama_binary(binary: str, *, template_only: bool = False) -> Path:
    """Follows the link chain, bounded; template_only stops at the first link not from the installer."""
    resolved = Path(binary).resolve()
    seen = {resolved}
    for _ in range(8):
        try:
            with open(resolved, "rb") as _f:
                _head = _f.read(256)
        except OSError:
            break
        if not _head.startswith(b"#!"):
            break
        _m = re.search(r'exec "\$\(dirname "\$0"\)/([^"]+)"', _head.decode("utf-8", "ignore"))
        if not _m:
            break
        if template_only and not _is_installer_entrypoint(str(resolved)):
            break
        nxt = (resolved.parent / _m.group(1)).resolve()
        if nxt in seen:
            break
        seen.add(nxt)
        resolved = nxt
    return resolved


def _llama_lib_dir(binary: str) -> Path:
    return _resolve_llama_binary(binary).parent


# write_exec_wrapper emits exactly this; anything more is a user script.
_INSTALLER_WRAPPER_RE = re.compile(
    r"\A#!/bin/sh\n" r'exec "\$\(dirname "\$0"\)/[^"\n]+" "\$@"\n?\Z'
)


def _is_installer_entrypoint(binary: str) -> bool:
    """Only an exact installer template counts among scripts; the resolver must still follow ANY wrapper."""
    try:
        path = Path(binary)
        if path.is_symlink():
            return True
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return True
    if not head.startswith(b"#!"):
        return True
    return bool(_INSTALLER_WRAPPER_RE.match(head.decode("utf-8", "ignore")))


def _loader_path_var() -> str:
    """dyld ignores LD_LIBRARY_PATH, so macOS needs DYLD_LIBRARY_PATH; Windows uses PATH."""
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    if sys.platform == "win32":
        return "PATH"
    return "LD_LIBRARY_PATH"


def _prepend_loader_dir(existing: str, lib_dir: str) -> str:
    """Dedupes textually, not via realpath, so a dead network mount in the path cannot stall a model
    load."""
    out: list[str] = []
    seen: set[str] = set()
    for entry in (lib_dir, *existing.split(os.pathsep)):
        if not entry:
            continue
        key = os.path.normcase(os.path.normpath(entry))
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return os.pathsep.join(out)


_CPU_RUNTIME_OWNER_FILE = "UNSLOTH_OWNER_PID"

# Thread-local: polling wants live VRAM, not this load's memo; GPU backends must not carry over.
_LOAD_PROBE_STATE = threading.local()


def _load_probe_memo_armed() -> bool:
    return getattr(_LOAD_PROBE_STATE, "armed", False)


def _load_probe_memo_get(kind, binary):
    rows = getattr(_LOAD_PROBE_STATE, "rows", None)
    return rows.get((kind, binary)) if rows else None


def _load_probe_memo_put(kind, binary, rows) -> None:
    if not _load_probe_memo_armed():
        return
    store = getattr(_LOAD_PROBE_STATE, "rows", None)
    if store is None:
        store = _LOAD_PROBE_STATE.rows = {}
    store[(kind, binary)] = rows


def _arm_load_probe_memo() -> None:
    """Start memoising the launch probes for the placement decision, on this thread
    only. Keyed by kind: the placement asks two different subprocesses, the device
    enumeration and, for a Vulkan target, the discrete-versus-shared probe.

    Called at the placement work rather than at the load call, which also covers the
    Hub download: rows captured before a multi-minute download would price the fit
    against VRAM that has since been allocated. `_load_probe_memo_scope` on the load
    call owns the lifetime, so this never has to be unwound by hand.
    """
    # Already armed this load: keep the rows the launch reuses.
    if _load_probe_memo_armed():
        return
    _LOAD_PROBE_STATE.armed = True
    _LOAD_PROBE_STATE.rows = {}


@contextlib.contextmanager
def _load_probe_memo_scope():
    """Guarantee the memo cannot outlive a load however it exits. Arming is the
    narrower `_arm_load_probe_memo`, taken around the placement decision. The whole
    load also keeps a llama-fit-params failure that is not cached across loads (a
    timeout, a crash) and the Metal fit lines already logged, so a second budget pass
    neither waits on the probe again nor repeats its message."""
    prev = (
        getattr(_LOAD_PROBE_STATE, "metal_transient", None),
        getattr(_LOAD_PROBE_STATE, "logged", None),
    )
    _LOAD_PROBE_STATE.metal_transient = {}
    _LOAD_PROBE_STATE.logged = set()
    try:
        yield
    finally:
        _LOAD_PROBE_STATE.armed = False
        _LOAD_PROBE_STATE.rows = None
        _LOAD_PROBE_STATE.metal_transient, _LOAD_PROBE_STATE.logged = prev


def _info_once_per_load(msg: str, *args) -> None:
    """``logger.info`` that a load says once (_load_probe_memo_scope); always outside one."""
    seen = getattr(_LOAD_PROBE_STATE, "logged", None)
    if seen is not None:
        if (msg, args) in seen:
            return
        seen.add((msg, args))
    logger.info(msg, *args)


# "Probed, no usable answer", kept apart from None = "not probed yet".
_MISSING = object()


_vulkan_inventory_cache: dict = {}


_vulkan_inventory_lock = threading.Lock()


# Display reads only (/api/system would spawn the probe per poll); a GPU memory event or fresh_reads() skips it.
_VULKAN_INVENTORY_TTL_S = 30.0

# Discrete by construction: `_weights_in_host_memory` classifies them upstream. Vulkan
# gets the shared-memory probe; SYCL/OpenCL/MUSA/CANN may be integrated, so they decline.
_SELF_EVIDENTLY_DISCRETE = frozenset({"cuda"})

# Empty set may mean no ROCm torch (e.g. Windows), so AMD needs the classifier to have run.
_NEEDS_ROCM_CLASSIFICATION = frozenset({"rocm", "hip"})

# The rungs narrow these, so the memo must key on them too.
_DEVICE_VISIBILITY_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "GGML_BACKEND_PATH",
    "LLAMA_ARG_DEVICE",
    "LLAMA_ARG_MAIN_GPU",
)

# No internal spaces in the id excludes ggml_cuda_init's '  Device 0:' lines.
_LISTED_DEVICE_RE = re.compile(r"^\s+(\S+):\s")
_LIST_DEVICES_HEADER = "Available devices:"


def _parse_listed_devices(text: Optional[str]) -> Optional[list[str]]:
    """Only lines below the header count: backend init logging above it also has colons."""
    if not text:
        return None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == _LIST_DEVICES_HEADER:
            break
    else:
        # No header: an old build or unreadable output, neither of which is evidence.
        return None
    devices = []
    for line in lines[index + 1 :]:
        if not line.strip():
            continue
        if not line[:1].isspace():
            break  # dedent ends the block
        match = _LISTED_DEVICE_RE.match(line)
        if match:
            devices.append(match.group(1))
    return devices


# Exact or versioned soname: a prefix match also hit 'ggml-cuda.dll.disabled'.
_GGML_GPU_BACKEND_RE = re.compile(
    r"^(?:lib)?ggml-(?:cann|cuda|hip|metal|musa|opencl|sycl|virtgpu|vulkan)"
    r"(?:\.dll|\.so(?:\.\d+)*|(?:\.\d+)*\.dylib)$"
)


# libllama the same way: a `.bak` / `.disabled` copy is not the library the child loads.
_LLAMA_SHARED_LIB_RE = re.compile(
    r"^(?:lib)?llama(?:\.dll|\.so(?:\.\d+)*|(?:\.\d+)*\.dylib)$", re.IGNORECASE
)


def _cpu_runtime_owner_alive(staged_dir: Path) -> bool:
    """Whether a live process still owns this staged CPU-fallback runtime."""
    try:
        pid = int((staged_dir / _CPU_RUNTIME_OWNER_FILE).read_text(encoding = "utf-8").strip())
    except (OSError, ValueError):
        # No owner stamp: written by an older Unsloth, so leave it alone.
        return True
    if pid == os.getpid():
        return True
    # 0 and negatives address process groups, never an owner.
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        ctypes.set_last_error(0)
        handle = kernel32.OpenProcess(0x00100000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _sweep_abandoned_cpu_runtimes(runtime_root: Path) -> None:
    """Delete staged runtimes whose owning Unsloth is gone (kill -9, host crash)."""
    try:
        candidates = list(runtime_root.glob("llama-cpu-*"))
    except OSError:
        return
    for staged_dir in candidates:
        if not staged_dir.is_dir() or _cpu_runtime_owner_alive(staged_dir):
            continue
        with contextlib.suppress(OSError):
            shutil.rmtree(staged_dir, ignore_errors = True)


def _lib_dir_has_ggml_backend(lib_dir: Path, backend: str) -> bool:
    """Match an exact or versioned ggml backend library soname."""
    stem = f"ggml-{backend}.dll" if sys.platform == "win32" else f"libggml-{backend}.so"
    try:
        return any(
            f.name == stem or f.name.startswith(stem + ".")
            for f in lib_dir.iterdir()
            if f.is_file()
        )
    except OSError:
        return False


def _is_external_link(path: Path) -> bool:
    """True when ``path`` is a --with-llama-cpp-dir local link: a POSIX symlink
    or a Windows directory junction / reparse point. Such a link resolves into
    the user's own llama.cpp checkout, which Unsloth does not own."""
    try:
        if os.path.islink(path):
            return True
    except OSError:
        return False
    if os.name == "nt":
        try:
            import stat
            attrs = os.lstat(path).st_file_attributes  # type: ignore[attr-defined]
            return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except (OSError, AttributeError):
            return False
    return False
