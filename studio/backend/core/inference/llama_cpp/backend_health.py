# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, crash recovery and health wait."""

import httpx
import threading
import time

from dataclasses import replace
from typing import (
    Callable,
    Optional,
)
from core.inference.llama_server_args import strip_shadowing_flags

from .launch_flags import (
    _child_spec_env,
    _extra_args_requests_mtp,
)
from .log import logger
from .memory_policy import (
    _LLAMA_FIT_TARGET_DEFAULT_MIB,
    _VRAM_FLOOR_RESERVE_MIB,
    _extra_args_set_spec_type,
)
from .spec_args import _canonicalize_spec_mode
from core.inference.llama_server_args import fit_is_effectively_on
from core.inference.llama_server_args import fit_target_margin_in
from typing import Iterable
from typing import List
from typing import Mapping
import math
import os
import sys
from .core_types import (
    _AUTO_RECOVERY_LOAD,
    _automatic_recovery_load,
)
from .draft_args import (
    _COMPANION_DEVICE_FLAG_GROUPS,
    _extra_args_device,
    _extra_args_mtp_draft_path,
)
from .launch_support import (
    _local_ssl_context,
)
from .server_log import (
    _PORT_OWNER_REFUSED_MARKER,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _HealthMixin:
    def _maybe_recover_from_mtp_crash(
        self,
        exc: Optional[BaseException] = None,
        *,
        sidecar_oom: bool = False,
    ) -> bool:
        """Schedule one background reload without MTP after a mid-generation death.

        An MTP launch can pass /health and the load-time fallback, then abort on
        a later decode (MTP+tensor flash-attn, Gemma 4 on Apple Silicon #13253).
        Not a persistent ban: a fresh load re-tries MTP. Returns True if scheduled.
        ``sidecar_oom`` is the Metal-retirement caller: an Auto DSpark / DFlash launch
        whose server was just killed after running out of Metal memory.
        """
        # Cheap async-safe gate: only our live MTP launch, not cancelled, with a
        # snapshot to replay.
        if self._cancel_event.is_set():
            return False
        if not (self._mtp_runtime_fallback_active or sidecar_oom):
            return False
        # Read before claiming: a raise after the claim strands the flag, and nothing
        # else clears it, blocking every later respawn.
        snapshot = self._last_load_intent
        proc = self._process
        if snapshot is None or proc is None:
            return False
        if self._metal_oom_is_terminal(snapshot):
            return False
        # Stopped on purpose (Metal OOM, lost device): the next request's respawn replays it.
        if not sidecar_oom and proc is getattr(self, "_retired_process", None):
            return False
        # Still serving: a transient error, so leave the caller its normal retry.
        if proc.poll() is None and self._server_socket_is_open():
            return False
        # Single-flight: the first failure claims the reload.
        with self._mtp_runtime_fallback_lock:
            if self._mtp_runtime_fallback_in_progress:
                return False
            self._mtp_runtime_fallback_in_progress = True

        def _recover():
            try:
                # Confirm exit so a transient stream error can't disable MTP.
                deadline = time.monotonic() + 5.0
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if proc.poll() is None:
                    logger.debug("Generation error but llama-server is alive; keeping MTP.")
                    return
                if not sidecar_oom:
                    logger.warning(
                        "llama-server exited mid-generation with MTP speculative "
                        "decoding (%s); reloading without speculative decoding.",
                        type(exc).__name__ if exc is not None else "server exited",
                    )
                # Re-check under the load lock so a newer load isn't clobbered.
                requested_mode = snapshot.speculative_type
                with self._serial_load_lock:
                    with self._lock:
                        if self._cancel_event.is_set():
                            logger.info("MTP-crash reload skipped: load was cancelled/unloaded.")
                            return
                        if self._process is not proc:
                            logger.info("MTP-crash reload skipped: a newer load is already active.")
                            return
                        if self._last_load_intent != snapshot:
                            logger.info("MTP-crash reload skipped: load settings changed.")
                            return
                        if self._metal_oom_is_terminal(snapshot):
                            logger.info("MTP-crash reload skipped: out of Metal memory twice.")
                            return
                        if not sidecar_oom and proc is getattr(self, "_retired_process", None):
                            logger.info("MTP-crash reload skipped: the server was retired.")
                            return
                        unload_epoch = self._unload_epoch
                        dropped_draft = self._mtp_draft_path or self._mtp_draft_suppressed_path
                    # llama.cpp appends spec types, so strip them from extras.
                    _ea = list(snapshot.extra_args or ())
                    fallback_extra_args = snapshot.extra_args
                    if _extra_args_requests_mtp(_ea, env = _child_spec_env(_ea)):
                        fallback_extra_args = tuple(
                            strip_shadowing_flags(
                                _ea,
                                strip_context = False,
                                strip_cache = False,
                                strip_spec = True,
                                strip_template = False,
                                strip_split_mode = False,
                            )
                        )
                    fallback = replace(
                        snapshot,
                        speculative_type = "off",
                        extra_args = fallback_extra_args,
                    )
                    with _automatic_recovery_load():
                        started = bool(self.load_model(fallback))
                    if not started:
                        logger.info("MTP-crash reload stopped before the replacement was ready.")
                        return
                    undo_reload = False
                    with self._lock:
                        if self._cancel_event.is_set() or self._unload_epoch != unload_epoch:
                            undo_reload = True
                        else:
                            # Keep the original extras: a comparator seeing only the stripped list reloads the MTP
                            # config.
                            if fallback_extra_args is not snapshot.extra_args:
                                self._requested_extra_args = (
                                    self._strip_device_extra_args(_ea)
                                    if self._gpu_ids_own_placement(snapshot.gpu_ids)
                                    else list(_ea)
                                )
                            # Restore the requested mode + reason load_model("off") cleared,
                            # so /status shows the user's mode + note (like the startup fallback).
                            # None when the extras own --spec-type, as the duplicate-load match expects.
                            self._requested_spec_mode = (
                                None
                                if _extra_args_set_spec_type(_ea)
                                else _canonicalize_spec_mode(requested_mode)
                            )
                            self._spec_fallback_reason = "runtime_error"
                            # Keep the Auto identity so the same request matches instead of
                            # relaunching the sidecar.
                            if dropped_draft:
                                self._mtp_draft_path = None
                                self._mtp_draft_suppressed_path = dropped_draft
                    if undo_reload:
                        logger.info(
                            "MTP-crash reload undone: the model was unloaded during reload."
                        )
                        self.unload_model()
                        return
                logger.info("Reloaded without speculative decoding after the mid-generation crash.")
            except Exception as e:
                logger.error(f"Reload without MTP failed: {e}")
            finally:
                with self._mtp_runtime_fallback_lock:
                    self._mtp_runtime_fallback_in_progress = False

        try:
            threading.Thread(target = _recover, daemon = True, name = "mtp-crash-reload").start()
        except RuntimeError as exc:
            # Release the claim: a reload that never started would block respawn forever.
            with self._mtp_runtime_fallback_lock:
                self._mtp_runtime_fallback_in_progress = False
            logger.error(f"Could not start the MTP-crash reload: {exc}")
            return False
        return True

    def _start_mtp_crash_watchdog(self) -> None:
        """Recover a llama-server that dies with no request watching. Never armed by a non-MTP
        recovery reload, so a server that keeps dying can't loop."""
        if not self._mtp_runtime_fallback_active and getattr(_AUTO_RECOVERY_LOAD, "active", False):
            return
        proc = self._process
        if proc is None:
            return
        # Replace any prior watchdog (loads are serialised, so at most one).
        self._stop_mtp_crash_watchdog()
        stop = threading.Event()
        self._mtp_watchdog_stop = stop

        def _watch():
            # Exit on stop or process death. _kill_process sets stop before
            # terminating, so re-check it: only a real crash (stop unset) recovers.
            while not stop.wait(1.0):
                if proc.poll() is not None:
                    if not stop.is_set():
                        self._maybe_recover_from_mtp_crash() or self._respawn_if_dead()
                    return

        t = threading.Thread(target = _watch, daemon = True, name = "mtp-crash-watchdog")
        self._mtp_watchdog_thread = t
        t.start()

    def _stop_mtp_crash_watchdog(self) -> None:
        """Signal the crash watchdog to exit; called before any deliberate kill."""
        stop = getattr(self, "_mtp_watchdog_stop", None)
        if stop is not None:
            stop.set()
        self._mtp_watchdog_thread = None

    # Extend the health deadline while startup work continues.
    _HEALTH_STALL_TIMEOUT_S = 600.0

    # The ceiling's multiple when the probe listed a device (that stall ends with advice only).
    _LISTED_VULKAN_CEILING_FACTOR = 3.0

    # Unrenewable auto-Vulkan ceiling per GiB of model (never below the stall window).
    _AUTO_VULKAN_START_S_PER_GIB = 60.0

    # First auto-Vulkan launch after the probe timed out or faulted.
    _SUSPECT_VULKAN_STALL_TIMEOUT_S = 300.0
    _STARTUP_PROGRESS_SAMPLE_S = 5.0
    # Ignore activity from the health probes themselves.
    _STARTUP_PROGRESS_MIN_BYTES = 1 << 20
    _STARTUP_PROGRESS_MIN_PAGE_FAULTS = 64
    _STARTUP_PROGRESS_MIN_CPU_FRACTION = 0.05

    @staticmethod
    def _page_faults(member, memory) -> int:
        # mmap loading can progress through page faults while RSS stays flat.
        for field in ("pfaults", "num_page_faults"):  # macOS, Windows
            if hasattr(memory, field):
                return getattr(memory, field)
        try:
            with open(f"/proc/{member.pid}/stat", "rb") as f:
                fields = f.read().rsplit(b")", 1)[1].split()
            return int(fields[7]) + int(fields[9])  # minflt + majflt
        except (OSError, IndexError, ValueError):
            return 0

    @classmethod
    def _startup_work_sample(cls, process) -> Optional[tuple[float, int, int, int]]:
        """Return cumulative (CPU, RSS, bytes read, page faults) for the process tree."""
        pid = getattr(process, "pid", None)
        if not isinstance(pid, int):
            return None
        try:
            import psutil
            root = psutil.Process(pid)
            members = [root, *root.children(recursive = True)]
        except Exception:
            return None
        cpu = 0.0
        rss = read = faults = 0
        for member in members:
            try:
                with member.oneshot():
                    times = member.cpu_times()
                    memory = member.memory_info()
                    try:
                        io = member.io_counters()
                    except (AttributeError, psutil.AccessDenied):
                        io = None  # macOS has no per-process I/O counters
                member_faults = cls._page_faults(member, memory)
            except Exception:
                if member is root:
                    return None
                continue
            cpu += times.user + times.system
            rss += memory.rss
            faults += member_faults
            if io is not None:
                read += getattr(io, "read_chars", io.read_bytes)  # Windows has no read_chars
        return (cpu, rss, read, faults)

    @classmethod
    def _made_startup_progress(
        cls,
        before: tuple[float, int, int, int],
        after: tuple[float, int, int, int],
        elapsed: float,
        cpu_counts: bool = True,
    ) -> bool:
        # RSS is a high-water mark so evicted pages are not counted twice.
        return (
            after[1] - before[1] >= cls._STARTUP_PROGRESS_MIN_BYTES
            or after[2] - before[2] >= cls._STARTUP_PROGRESS_MIN_BYTES
            or after[3] - before[3] >= cls._STARTUP_PROGRESS_MIN_PAGE_FAULTS
            or (
                cpu_counts
                and after[0] - before[0] >= cls._STARTUP_PROGRESS_MIN_CPU_FRACTION * elapsed
            )
        )

    def _wait_for_health(
        self,
        timeout: float = 120.0,
        interval: float = 0.5,
        cancelled: Optional[Callable[[], bool]] = None,
        cpu_counts_as_progress: bool = True,
        max_total_s: Optional[float] = None,
    ) -> bool:
        """Poll /health until success, exit, cancellation, or a startup stall.

        ``cpu_counts_as_progress`` False renews only on loading work (RSS, reads, faults):
        a spinning GPU driver burns CPU forever (#142). ``max_total_s`` caps the whole wait.
        """
        deadline = time.monotonic() + timeout
        ceiling = deadline - timeout + max_total_s if max_total_s is not None else None
        self._health_wait_stalled = False
        # "no_progress" (the stall window passed with no loading work) or "ceiling".
        self._health_wait_stall_kind = None
        clipped_by_ceiling = False
        work_sample = None
        work_sampled_at = None
        work_checked_at = None
        url = f"{self.base_url}/health"
        health_probe_event = getattr(self, "_health_probe_event", None)
        if health_probe_event is None:  # Backward-compatible with lightweight test doubles.
            health_probe_event = self._health_probe_event = threading.Event()

        if cancelled is None:
            # A scoped /load carries its own cancel event, so load_model passes its own
            # predicate; fall back to the unload flag for callers that have none.
            cancel_event = getattr(self, "_cancel_event", None)
            cancelled = cancel_event.is_set if cancel_event is not None else None

        # Why this wait ended, for callers that must tell a cancel apart from a crash:
        # a cancel landing during CPU-fallback staging is not a cancelled wait.
        self._health_wait_cancelled = False
        # No teardown reset here: _torn_down_process is matched by identity.
        process = None  # the child this wait last looked at, read again after the loop

        while (now := time.monotonic()) < deadline:
            # Durable, unlike the per-process marker below: once teardown begins,
            # every later iteration sees it.
            if getattr(self, "_shutting_down", False):
                logger.info("llama-server was torn down while waiting for it to become healthy")
                self._health_wait_cancelled = True
                return False
            # unload_model() blocks on self._lock, which the load holds across this wait.
            if cancelled is not None and cancelled():
                logger.info("llama-server startup cancelled before it became healthy")
                self._health_wait_cancelled = True
                return False
            # Cleared before probing so output during the request stays latched for
            # the fallback wait below.
            health_probe_event.clear()
            # Read once: shutdown clears the reference from another thread (#10353).
            process = self._process
            if process is None:
                logger.info("llama-server was torn down while waiting for it to become healthy")
                # Legacy path stays silent at llama.cpp's default, so an untouched slider adds no flag.
                self._health_wait_cancelled = True
                return False
            # Process crashed?
            if process.poll() is not None:
                # A teardown holds the reference, so this child exiting is deliberate.
                if getattr(self, "_torn_down_process", None) is process:
                    logger.info("llama-server was torn down while waiting for it to become healthy")
                    self._health_wait_cancelled = True
                    return False
                # Let the drain thread collect final output.
                if self._stdout_thread is not None:
                    self._stdout_thread.join(timeout = 2)
                output = "\n".join(self._stdout_lines[-50:])
                # Keep the tail: crash details print after the long startup banner.
                _log_hint = (
                    f" Full log: {self._llama_log_path}"
                    if getattr(self, "_llama_log_path", None)
                    else ""
                )
                logger.error(
                    f"llama-server exited with code {process.returncode}. "
                    f"Output (tail): {output[-2000:]}{_log_hint}"
                )
                return False

            if work_checked_at is None or now - work_checked_at >= self._STARTUP_PROGRESS_SAMPLE_S:
                work_checked_at = now
                sample = self._startup_work_sample(process)
                if sample is not None:
                    if work_sample is not None and self._made_startup_progress(
                        work_sample,
                        sample,
                        now - work_sampled_at,
                        cpu_counts = cpu_counts_as_progress,
                    ):
                        deadline = now + timeout
                        if ceiling is not None and deadline > ceiling:
                            deadline = ceiling
                            clipped_by_ceiling = True
                    peak = sample[1] if work_sample is None else max(work_sample[1], sample[1])
                    work_sample = (sample[0], peak, sample[2], sample[3])
                    work_sampled_at = now

            try:
                # trust_env=False: an HTTP proxy can 503 for 127.0.0.1 and hang the load.
                # Shared context: httpx otherwise loads the CA bundle on every probe.
                resp = httpx.get(url, timeout = 2.0, trust_env = False, verify = _local_ssl_context())
                if resp.status_code == 200:
                    # A cancel during the 2s probe must not publish the dropped model.
                    if cancelled is not None and cancelled():
                        logger.info("llama-server became healthy after the load was cancelled")
                        self._health_wait_cancelled = True
                        return False
                    # A 200 arriving as shutdown began must not publish _healthy.
                    if getattr(self, "_shutting_down", False):
                        logger.info("llama-server became healthy while the app was shutting down")
                        self._health_wait_cancelled = True
                        return False
                    if not self._health_listener_is_ours(process, getattr(self, "_port", None)):
                        self._stdout_lines.append(_PORT_OWNER_REFUSED_MARKER)
                        return False
                    return True
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                # TCP RST while binding (Windows WinError 10054).
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.WriteError,
            ):
                pass

            health_probe_event.wait(interval)

        if cancelled is not None and cancelled():
            logger.info("llama-server startup cancelled at the health-check deadline")
            self._health_wait_cancelled = True
            return False

        # Check both: _shutting_down is set on entry, _torn_down_process later.
        if getattr(self, "_shutting_down", False) or (
            process is not None and getattr(self, "_torn_down_process", None) is process
        ):
            logger.info("llama-server was torn down while waiting for it to become healthy")
            self._health_wait_cancelled = True
            return False

        # Exited during the last probe or interval: a crash for the crash recovery, not a stall.
        if process is not None and process.poll() is not None:
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout = 2)
            logger.error(
                f"llama-server exited with code {process.returncode} at the health-check deadline. "
                f"Output (tail): {chr(10).join(self._stdout_lines[-50:])[-2000:]}"
            )
            return False

        # Leave a marker so _classify_llama_start_failure tells a live but
        # never-healthy load (too large, or a proxy hijacking the loopback
        # probe) apart from a bad GGUF (#5740).
        marker = f"llama-server health check timed out: no startup progress for {timeout:g}s"
        stall_kind = "no_progress"
        if clipped_by_ceiling:
            marker = f"llama-server health check timed out: not healthy after {max_total_s:g}s"
            stall_kind = "ceiling"
        self._stdout_lines.append(marker)
        logger.error(marker)
        self._health_wait_stalled = True
        self._health_wait_stall_kind = stall_kind
        return False

    @staticmethod
    def _fit_target_margin_mib(
        *,
        auto_fit: bool,
        fit_target_delta_mib: float = 0.0,
        supports_fit_target: bool = True,
        floor_mib: float = 0.0,
    ) -> float:
        """Free VRAM per device llama.cpp's fitter keeps for itself under ``--fit on``.

        One reader for the flag and for the load-mode fit, so the margin the child is
        told to keep and the margin the fit refuses to spend cannot drift. A server
        too old for ``--fit-target`` still keeps llama.cpp's own 1024 MiB default
        (common/common.h, ``fit_params_target``), which is also what the legacy path
        deliberately stays silent about. ``floor_mib`` is a margin the host needs kept
        whatever the budget says (the Metal wired headroom).
        """
        if not supports_fit_target:
            return _LLAMA_FIT_TARGET_DEFAULT_MIB
        base = _VRAM_FLOOR_RESERVE_MIB if auto_fit else _LLAMA_FIT_TARGET_DEFAULT_MIB
        return max(_VRAM_FLOOR_RESERVE_MIB, base + fit_target_delta_mib, floor_mib)

    @staticmethod
    def _per_device_fit_target_argv(
        cmd: list[str], env: Mapping[str, str], margins_by_idx: Optional[Mapping[int, float]]
    ) -> list[str]:
        """``cmd`` with Studio's ``--fit-target`` spelled per device for the child's mask.

        llama.cpp broadcasts a single value, so on mixed cards the margin sized for
        the largest card can exceed a smaller card's whole memory and leave it
        unused. Built from the FINAL child env at spawn time, since a retry can
        re-pin the mask. Returns ``cmd`` unchanged when the order cannot be proven
        (a device without a margin, a non-numeric mask, another device order) or
        when every card would get the same value, which a single value already says.
        """
        if not margins_by_idx:
            return cmd
        at = None
        for k in range(len(cmd) - 2, -1, -1):
            if cmd[k] == "--fit-target":
                at = k
                break
        if at is None:
            return cmd
        if (str(env.get("CUDA_DEVICE_ORDER") or "")).strip().upper() != "PCI_BUS_ID":
            return cmd
        # --device narrows the model's devices and --rpc puts remote ones first, both
        # re-indexing the list; a companion widen can add --device after planning.
        if (_extra_args_device(cmd, {"--device", "-dev", "--rpc"}) or "").strip() or any(
            str(env.get(k, "")).strip() for k in ("LLAMA_ARG_DEVICE", "LLAMA_ARG_RPC")
        ):
            return cmd
        mask = env.get("CUDA_VISIBLE_DEVICES")
        if mask is None:
            order = sorted(int(i) for i in margins_by_idx)
        else:
            try:
                order = [int(tok) for tok in str(mask).split(",") if tok.strip()]
            except ValueError:
                return cmd
        if not order or any(i < 0 or i not in margins_by_idx for i in order):
            return cmd
        values = [str(int(margins_by_idx[i])) for i in order]
        spelled = values[0] if len(set(values)) == 1 else ",".join(values)
        if cmd[at + 1] == spelled:
            return cmd
        out = list(cmd)
        out[at + 1] = spelled
        logger.info(
            "Per-device --fit-target %s for GPUs %s: each card keeps its own budget margin.",
            spelled,
            order,
        )
        return out

    def _positional_child_order_unproven(self, child_env: Mapping[str, str]) -> bool:
        """Whether a per-card list built over the parent's GPU ids may not line up with the
        child's CUDA enumeration (planning and spawn share it).

        nvidia-smi ids follow PCI order, so the child must run under PCI_BUS_ID
        (FASTEST_FIRST is CUDA's default) with a mask a resolver maps, unless the uuid
        mask order is known. Torch-fallback ids are CUDA ordinals in the parent's own
        order, so a child switched to another order (a GPU pick pins PCI_BUS_ID) counts
        them differently. HIP ignores CUDA_DEVICE_ORDER.
        """
        child_order = str(child_env.get("CUDA_DEVICE_ORDER", "")).strip().upper()
        if self._GPU_IDS_ARE_PCI_INDICES is True:
            return (
                child_order != "PCI_BUS_ID" or self._visibility_mask_is_unmappable()
            ) and self._smi_uuid_mask_order() is None
        if self._GPU_IDS_ARE_PCI_INDICES is False:
            parent_order = (os.environ.get("CUDA_DEVICE_ORDER") or "").strip().upper()
            return (parent_order or "FASTEST_FIRST") != (
                child_order or "FASTEST_FIRST"
            ) and not self._host_torch_is_rocm()
        return False

    def _log_sidecar_fit_reserve(
        self,
        run_cmd: List[str],
        env: Mapping[str, str],
        kind: str,
        sidecar_path: Optional[str],
        reserved_mib: int,
        logged: list,
    ) -> None:
        """Report whether the spawned argv reserves the DSpark / DFlash sidecar that
        llama.cpp's --fit cannot size. Once per outcome per load (``logged`` holds the
        last one), so a retry that changes it says so and a repeat stays quiet."""
        if not fit_is_effectively_on(run_cmd, env) or not _extra_args_mtp_draft_path(
            run_cmd, env = env
        ):
            return
        outcome = (kind, bool(reserved_mib))
        if logged and logged[-1] == outcome:
            return
        logged.append(outcome)
        label = "DSpark" if kind == "dspark" else "DFlash"
        if reserved_mib:
            logger.info(
                "%s under --fit on: llama.cpp cannot size the sidecar, so "
                "--fit-target keeps %.2f GB free for it.",
                label,
                reserved_mib / 1024,
            )
            return
        # Not a blocker: the fit step cannot measure a borrowed-tensor draft
        # (llama-context.cpp:153-160).
        logger.info(
            "%s under --fit on: llama.cpp cannot size the sidecar during "
            "fitting, so its ~%.1f GB is not reserved. Pin GPU Layers in "
            "Manual mode if the load runs out of VRAM.",
            label,
            (self._get_gguf_size_bytes(sidecar_path) or 0) / 1e9,
        )

    @staticmethod
    def _fit_target_per_device_ok(
        *,
        is_vulkan_backend: bool,
        extra_args: Optional[Iterable[str]],
        env: Optional[Mapping[str, str]],
    ) -> bool:
        """Whether a per-card ``--fit-target`` list can be lined up with llama.cpp's devices.

        The list is positional over the child's device enumeration. CUDA under
        PCI_BUS_ID enumerates the visible physical ids in mask order (ascending when
        unmasked), which ties a position to an nvidia-smi index. Vulkan and HIP order
        devices their own way, a UUID or MIG mask cannot be mapped back to indices,
        and --device / --rpc / a pass-through --fit-target change or own the list, so
        each of those keeps the single broadcast margin.
        """
        if is_vulkan_backend or sys.platform == "darwin":
            return False
        if LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is not True:
            return False
        if (os.environ.get("CUDA_DEVICE_ORDER") or "").strip().upper() != "PCI_BUS_ID":
            return False
        if LlamaCppBackend._visibility_mask_is_unmappable():
            return False
        # A resolvable uuid mask still reaches the child as uuids, which the spawn-time
        # spelling cannot order, so the pricing keeps the broadcast margin with it.
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mask is not None and not all(t.strip().isdigit() for t in mask.split(",") if t.strip()):
            return False
        if LlamaCppBackend._SMI_MIG_INDICES:
            return False
        if LlamaCppBackend._host_torch_is_rocm():
            return False
        env = env or {}
        if fit_target_margin_in(extra_args or (), env) is not None:
            return False
        if (_extra_args_device(extra_args, {"--device", "-dev", "--rpc"}) or "").strip():
            return False
        # A companion device flag can make the pin add --device after planning, which
        # the spawn-time spelling then declines; decline here so pricing agrees.
        if any(_extra_args_device(extra_args, flags) for flags in _COMPANION_DEVICE_FLAG_GROUPS):
            return False
        if any(str(env.get(k, "")).strip() for k in ("LLAMA_ARG_DEVICE", "LLAMA_ARG_RPC")):
            return False
        return True

    @staticmethod
    def _fit_target_margins_mib(
        *,
        auto_fit: bool,
        fit_target_delta_by_idx: Optional[Mapping[int, float]],
        supports_fit_target: bool = True,
    ) -> Optional[dict[int, float]]:
        """``_fit_target_margin_mib`` for each card, keyed by physical GPU id.

        None when there is no per-card delta, so callers keep the single margin. One
        formula for the priced margins and the emitted list, like the scalar.
        """
        if not fit_target_delta_by_idx or not supports_fit_target:
            return None
        return {
            int(idx): LlamaCppBackend._fit_target_margin_mib(
                auto_fit = auto_fit, fit_target_delta_mib = delta
            )
            for idx, delta in fit_target_delta_by_idx.items()
        }

    def _sidecar_fit_reserve_mib(
        self,
        reserve_bytes: int,
        sidecar_path: Optional[str],
        *,
        flat_mib: int = 0,
    ) -> int:
        """MiB a DSpark/DFlash sidecar needs kept free by the child's ``--fit-target``.

        The planner's byte-accurate reserve (draft weights + draft KV + draft compute +
        target verification rows), never less than the sidecar file itself, which is
        all that is known when the reserve fell back to the flat fraction (0 bytes).
        ``flat_mib``: that fraction of the target's pool, passed only when the planner held
        it back, so it is a floor: the child keeps at least what the plan kept.
        Rounded up: a short MiB is a short allocation.
        """
        try:
            file_bytes = int(self._get_gguf_size_bytes(sidecar_path) or 0) if sidecar_path else 0
        except Exception:
            file_bytes = 0
        need = max(int(reserve_bytes or 0), file_bytes)
        return max(math.ceil(need / (1024 * 1024)), max(0, int(flat_mib or 0)))

    @staticmethod
    def _ctx_integrity_flags(
        n_parallel: int,
        use_fit: bool,
        auto_fit: bool,
        requested_ctx: int,
        effective_ctx: int,
        caps: dict,
        *,
        fit_target_delta_mib: float = 0.0,
        fit_target_floor_mib: float = 0.0,
        fit_target_reserve_mib: int = 0,
    ) -> list[str]:
        """Explicit --parallel drops the --kv-unified default and splits -c per slot; restore the
        shared pool. ``fit_target_reserve_mib`` is added on top for a sidecar the fitter cannot
        measure."""
        flags: list[str] = []
        if n_parallel > 1 and caps.get("supports_kv_unified"):
            flags.append("--kv-unified")
        if use_fit and caps.get("supports_fit_ctx"):
            if requested_ctx > 0 and effective_ctx > 0:
                # Floor the fit step at the explicitly requested ctx.
                flags.extend(["--fit-ctx", str(effective_ctx)])
            elif auto_fit:
                # Manual + Auto omits -c, so floor at 8192 so --fit keeps a usable window.
                flags.extend(["--fit-ctx", "8192"])
        if use_fit and caps.get("supports_fit_target"):
            # 512 margin under Manual + Auto, llama.cpp's 1024 default on the legacy path.
            _target = LlamaCppBackend._fit_target_margin_mib(
                auto_fit = auto_fit,
                fit_target_delta_mib = fit_target_delta_mib,
                floor_mib = fit_target_floor_mib,
            )
            # On top of every margin, not max(): the sidecar's bytes must stay free.
            _target += max(0, fit_target_reserve_mib)
            # The legacy path stays silent when it would only restate llama.cpp's
            # own default, so an untouched slider leaves the command unchanged.
            if auto_fit or _target != _LLAMA_FIT_TARGET_DEFAULT_MIB:
                flags.extend(["--fit-target", str(int(_target))])
        return flags

    def _query_server_props(self) -> Optional[dict]:
        """llama-server's ``/props``, or None when it cannot be read."""
        url = f"{self.base_url}/props"
        try:
            # /props is not one of llama-server's public endpoints, so with the
            # child's --api-key an unauthenticated read 401s: the context readback silently keeps
            # the requested -c, and video reads as unsupported on a model that
            # supports it. None when there is no child key, which is httpx's
            # default and what every other call site here relies on.
            resp = httpx.get(
                url,
                headers = self._auth_headers,
                timeout = 5.0,
                trust_env = False,
                verify = _local_ssl_context(),
            )
            if resp.status_code != 200:
                return None
            props = resp.json()
            return props if isinstance(props, dict) else None
        except Exception:
            return None

    def _query_server_n_ctx(self) -> Optional[int]:
        """Adopts real n_ctx up to the explicit request so padding is not advertised; max_context_length stays put, since /props confirms allocation, not VRAM fit"""
        props = self._query_server_props()
        if props is None:
            return None
        modalities = props.get("modalities")
        if isinstance(modalities, dict):
            self._has_video_input = bool(modalities.get("video"))
        settings = props.get("default_generation_settings") or {}
        n_ctx = settings.get("n_ctx")
        return int(n_ctx) if n_ctx else None

    def _reconcile_effective_ctx_with_server(self, requested_n_ctx: int = 0) -> None:
        """Opens a streaming POST, retrying through prefill; yields (response, first_token_deadline) on 200 and owns the httpx client; RuntimeError on any other status"""
        actual_n_ctx = self._query_server_n_ctx()
        if not actual_n_ctx or actual_n_ctx <= 0:
            return
        slots = 1 if self._kv_cache_unified else self.effective_parallel_slots
        self._kv_cache_context_total = actual_n_ctx * slots
        effective_n_ctx = self._effective_context_length
        if not effective_n_ctx:
            # Same ceiling as the raise below, or a missing seed would advertise
            # past the explicit request.
            self._effective_context_length = (
                min(actual_n_ctx, requested_n_ctx) if requested_n_ctx > 0 else actual_n_ctx
            )
            return
        if actual_n_ctx < effective_n_ctx:
            logger.warning(
                "llama-server allocated a smaller per-request context than "
                f"requested ({effective_n_ctx} -> {actual_n_ctx}; "
                "memory fit or --parallel slot split); clients must treat "
                f"{actual_n_ctx} as the real context window."
            )
            self._effective_context_length = actual_n_ctx
            return
        if requested_n_ctx > effective_n_ctx:
            confirmed_n_ctx = min(actual_n_ctx, requested_n_ctx)
            if confirmed_n_ctx > effective_n_ctx:
                logger.info(
                    "llama-server confirmed a larger per-request context from "
                    f"the explicit override ({effective_n_ctx} -> {confirmed_n_ctx})."
                )
                self._effective_context_length = confirmed_n_ctx
