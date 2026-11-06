# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, unload, process teardown and orphan cleanup."""

import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time

from pathlib import Path
from typing import Optional
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory
from utils.process_lifetime import is_signalable_pid as _is_signalable_pid

from .binary_runtime import _is_external_link
from .core_types import _serving_backends_snapshot
from .launch_support import (
    _PROC_ROOT,
    _QUIT_SIGNAL_EXIT_GRACE_S,
    _SIGKILL_REAP_TIMEOUT_S,
    _expanded_user_path,
)
from .log import logger

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _ShutdownMixin:
    def load_cancelled(self) -> bool:
        """True if a load was cancelled (e.g. via unload/_cancel_event) and not
        yet consumed by the next load_model. Lets the tensor->layer fallback
        avoid restarting a load the user just cancelled."""
        return self._cancel_event.is_set()

    @_invalidates_gpu_memory("llama.cpp unload")
    def unload_model(self) -> bool:
        """Terminate the subprocess and cancel any in-flight download."""
        self._cancel_event.set()
        with self._lock:
            self._unload_epoch += 1
            # Read before the kill clears it: callers unload from a finally either way.
            _was_resident = self._process is not None
            self._kill_process()
            self._cleanup_cpu_fallback_runtime()
            if self._api_key_file is not None:
                with contextlib.suppress(OSError):
                    os.unlink(self._api_key_file)
                self._api_key_file = None
            # The one unload line: routes/inference.py logged a second, differently named.
            if _was_resident:
                logger.info(f"Unloaded GGUF model: {self._model_identifier}")
            self._model_identifier = None
            self._gguf_path = None
            self._gguf_load_identity = None
            self._hf_repo = None
            self._mtp_draft_path = None
            self._mtp_draft_suppressed_path = None
            self._mtp_draft_suppressed_reason = None
            self._spec_fallback_reason = None

            self._mmproj_fallback_reason = None
            self._capability_probe_inconclusive = False
            self._spec_drafter_kind = None
            self._dspark_sidecar_absent = False
            self._dflash_sidecar_absent = False
            self._dflash_retry_needed = False
            self._launch_binary_revision = ()
            self._cpu_fallback_reason = None
            self._last_load_intent = None
            self._mtp_runtime_fallback_active = False
            self._metal_oom_strikes = None
            self._metal_oom_terminal_key = None
            self._gpu_oom_strikes = None
            self._hf_variant = None
            self._is_vision = False
            self._disable_vision = False
            self._vision_disabled_by_user = False
            self._is_audio = False
            self._audio_type = None
            self._audio_probed = False
            self._has_audio_input = False
            self._has_video_input = False
            self._video_needs_ffmpeg = False
            self._mmproj_has_audio = False
            self._mmproj_accepts_image = True
            self._port = None
            self._healthy = False
            self._context_length = None
            self._effective_context_length = None
            self._max_context_length = None
            self._vram_fit_context_length = None
            self._reset_effective_parallel_slots()
            self._slot_save_dir = None
            self._slot_save_binary = None
            self._slot_loaded_identity = None
            self._prompt_cache_disabled = False
            self._swa_full = False
            self._kv_cache_unified = False
            self._memory_state = None
            self._memory_direct_io = None
            self._memory_dio_applicable = False
            self._memory_dio_flags = []
            self._memory_dio_user_tokens = []
            self._memory_policy_active = False
            self._memory_policy_extras_touched = False
            self._memory_mlock_applicable = True
            self._memory_pending_launch = None
            self._vram_fraction_pending = None
            self._loaded_by_user_action = False
            self._n_ubatch = self._DEFAULT_N_UBATCH
            self._flash_attn_enabled = True
            self._effective_cache_types = ("f16", "f16")
            self._requested_cache_types = ("f16", "f16")
            self._kv_cache_context_total = None
            self._chat_template = None
            self._markup_tokens = []
            self._markup_profile = None
            self._chat_template_override = None
            self._empty_chat_render_refused = False
            self._supports_reasoning = False
            self._reasoning_always_on = False
            self._reasoning_budget = -1
            self._reasoning_budget_message = ""
            self._requested_reasoning_budget = -1
            self._requested_reasoning_budget_message = ""
            self._reasoning_style = "enable_thinking"
            self._reasoning_effort_levels = []
            self._reasoning_default = True
            self._supports_preserve_thinking = False
            self._preserve_thinking_default = False
            self._supports_tools = False
            self._cache_type_kv = None
            # Not a bare pid_is_running read: kills are async, so asking right after one answers its own latency.
            self._gpu_ids = None
            self._requested_gpu_ids = None
            self._tensor_parallel = False
            self._gpu_memory_mode = "auto"
            self._gpu_layers = -1
            self._n_cpu_moe = 0
            self._tensor_split = None
            self._auto_tensor_split = None
            self._auto_tensor_split_emitted = None
            self._arch_gate_forced_cpu = False
            self._layer_preserves_tensor_intent = False
            self._speculative_type = None
            self._requested_spec_mode = None
            self._spec_draft_n_max = None
            self._n_layers = None
            self._n_experts = None
            self._leading_dense_block_count = None
            self._n_kv_heads = None
            self._n_kv_heads_by_layer = None
            self._n_heads = None
            self._embedding_length = None
            self._pooling_type = None
            self._kv_key_length = None
            self._kv_value_length = None
            self._sliding_window = None
            self._sliding_window_pattern = None
            self._full_attention_interval = None
            self._kv_lora_rank = None
            self._key_length_mla = None
            self._kv_key_length_swa = None
            self._kv_value_length_swa = None
            self._ssm_inner_size = None
            self._ssm_state_size = None
            self._ssm_group_count = None
            self._ssm_conv_kernel = None
            self._kda_head_dim = None
            self._shared_kv_layers = None
            self._nextn_predict_layers = None
            # Clean up temp chat template file.
            if hasattr(self, "_chat_template_file") and self._chat_template_file:
                try:
                    os.unlink(self._chat_template_file.name)
                except Exception:
                    pass
                self._chat_template_file = None
            self._unload_audio_codec()
            return True

    @staticmethod
    def _leading_process_group(pid):
        """Returns the pid's own process group when it leads one; pid 1 must be rejected, since getpgid(1) == 1 makes killpg SIGKILL every process the user owns."""
        if not _is_signalable_pid(pid):
            return None
        if os.name != "posix" or not hasattr(os, "getpgid"):
            return None
        try:
            pgid = os.getpgid(pid)
        except OSError:
            return None
        return pgid if pgid == pid else None

    @staticmethod
    def _quit_signal_grace_left(pid) -> Optional[float]:
        if not _is_signalable_pid(pid) or not hasattr(os, "getpgid"):
            return None
        from utils.process_lifetime import seconds_since_quit_signal

        elapsed = seconds_since_quit_signal()
        try:
            if elapsed is None or os.getpgid(pid) != os.getpgrp():
                return None
        except OSError:
            return None
        return max(0.0, _QUIT_SIGNAL_EXIT_GRACE_S - elapsed)

    def _still_listening(self) -> bool:
        port = getattr(self, "_port", None)
        if not port:
            return True
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = 0.5):
                return True
        except ConnectionRefusedError:
            return False
        except OSError:
            return True

    @staticmethod
    def _kill_process_group(pgid):
        """Take down what the leader left behind, if anything is still there."""
        if not _is_signalable_pid(pgid):
            return  # killpg(1) is kill(-1); killpg(0) is our own group
        if not hasattr(os, "killpg"):
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    @staticmethod
    def _collect_descendants(pid):
        """Returns (children, known); empty with known False is not enumerable, so keep the record; reading it as none would delete the only handles on survivors."""
        try:
            from utils.process_lifetime import collect_descendants_known
            return collect_descendants_known(pid)
        except Exception:
            # The import or the walk itself failed, which says nothing about the tree.
            return [], False

    @staticmethod
    def _tree_kill_surviving_process(pid) -> bool:
        """True when gone; a last resort after terminate, where the held Popen settles ownership."""
        if not _is_signalable_pid(pid):
            return False
        try:
            from utils.process_lifetime import confirm_pid_exited, terminate_pid
        except Exception:
            return False
        try:
            terminate_pid(pid, timeout = 5.0, owner_verified = True)
        except Exception as e:
            logger.warning(f"Could not terminate surviving llama-server {pid}: {e}")
        try:
            # Not a bare `pid_is_running` read: every kill terminate_pid just made is
            # asynchronous, and asking on the line after one answers its own latency. The
            # two arms below it settle internally now, so this is belt and braces, but a
            # read that decides whether to delete the last handle on a live server is worth
            # spelling the same way everywhere.
            gone = confirm_pid_exited(pid)
        except Exception:
            return False
        if not gone:
            logger.warning(
                f"llama-server {pid} is still running after a tree kill; "
                "keeping its record so the next launch can reap it"
            )
        return gone

    @staticmethod
    def _confirm_group_kill_landed(pid) -> bool:
        """Bounded poll, since kills land asynchronously; False keeps the record and pidfile for reaping."""
        if not _is_signalable_pid(pid):
            return False
        try:
            from utils.process_lifetime import confirm_pid_exited
        except Exception:
            return False
        try:
            gone = confirm_pid_exited(pid)
        except Exception:
            return False
        if not gone:
            logger.warning(
                f"llama-server {pid} is still running after its process group was killed; "
                "keeping its record so the next launch can reap it"
            )
        return gone

    @staticmethod
    def _terminate_descendants(collected):
        """Returns survivors as (pid, identity), adopted with that identity since a bare pid may be a stranger; deleting their record would orphan a live GPU worker."""
        if not collected:
            return []
        try:
            from utils.process_lifetime import adopt_pid, terminate_descendants
            survivors = terminate_descendants(collected, timeout = 5.0)
        except Exception as e:
            logger.debug(f"Could not terminate server descendants: {e}")
            # from_snapshot: a None identity means unreadable, not new; capturing now records a recycled pid.
            return list(collected)
        for pid, identity in survivors:
            try:
                # Process-wide, not per-instance: atexit teardown skips run.py, and helper-load backends rely on it.
                adopt_pid(pid, identity, from_snapshot = True)
            except Exception:
                pass
        if survivors:
            logger.warning(
                f"llama-server descendants still running after the sweep: "
                f"{[pid for pid, _ in survivors]}; "
                "recorded so the next launch can reap them"
            )
        return survivors

    def _publish_healthy(self) -> bool:
        """Publishes _healthy under the spawn lock, so a stale load cannot publish after teardown."""
        with self._spawn_lock:
            if self._spawn_is_stale():
                logger.info("load no longer belongs to this lifecycle; not publishing it healthy")
                return False
            self._healthy = True
            self._planned_vram_mib = dict(getattr(self, "_pending_plan_mib", {}))
            return True

    def _close_attempt_log(
        self,
        *,
        reason: str = "closed",
        exit_code: "Optional[int]" = None,
    ) -> None:
        """Clear shutdown state per lifecycle from run_server before anything can spawn; take _teardown_lock first so an in-progress kill finishes first"""
        self._llama_log_tee_failed = False
        fh = getattr(self, "_llama_log_fh", None)
        if fh is None:
            return
        if exit_code is None:
            try:
                exit_code = self._process.poll()
            except Exception:
                pass
        try:
            fh.write(f"[studio] attempt end reason={reason} exit_code={exit_code}\n")
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
        self._llama_log_fh = None
        try:
            logger.info(
                f"attempt log closed reason={reason} exit_code={exit_code} "
                f"path={getattr(self, '_llama_log_path', None)}"
            )
        except Exception:
            # The logger raises once stdout is closed; teardown must not.
            pass

    def _begin_server_lifecycle(self) -> None:
        """Helper backends are not marked per-instance, so also check the process-wide latch or they spawn after the sweep; read it second, the attribute is cheaper"""
        with self._teardown_lock:
            with self._spawn_lock:
                self._shutting_down = False
                self._torn_down_process = None

    def _spawn_is_stale(self) -> bool:
        """Whether this load may no longer spawn. Caller holds _spawn_lock."""
        if getattr(self, "_shutting_down", False):
            return True
        # Helper backends are not marked per-instance, so also check the process-wide latch.
        from utils.process_lifetime import is_process_shutting_down

        return is_process_shutting_down()

    @_invalidates_gpu_memory("llama.cpp stop")
    def _kill_process(self, *, teardown: bool = False):
        """Terminate the subprocess if running.

        ``teardown`` marks an app-level stop (shutdown, atexit) rather than the
        retry ladder reaping a child it is about to replace: only the former may
        end an in-flight health wait.

        A teardown holds _teardown_lock for the WHOLE kill, because the terminate and
        wait below keep reading self._process and finally clear it: a lifecycle
        reopened mid-kill would have its new child dropped or terminated here, so
        _begin_server_lifecycle takes the same lock and waits.

        _spawn_lock is taken only long enough to set the flag, NOT across the kill.
        A spawn arriving mid-teardown then reads the flag and refuses in microseconds
        instead of queuing behind a SIGTERM/SIGKILL escalation that can run for
        seconds; the load thread it belongs to is one shutdown is already waiting on.
        Marked above the early return, since a quit during a download still has to be
        recorded.
        """
        if teardown:
            # Process-wide as well as per-instance: the atexit teardown reaches here
            # without going through run.py, and the backends a helper load builds for
            # itself are only ever covered by the shared latch.
            from utils.process_lifetime import mark_process_shutting_down

            mark_process_shutting_down()
            with self._teardown_lock:
                with self._spawn_lock:
                    self._shutting_down = True
                self._kill_process_body(teardown = True)
            return
        self._kill_process_body(teardown = False)

    def _kill_process_body(self, *, teardown: bool):
        """Read both before terminate: getpgid fails once the leader is reaped and children reparent on exit; the visual server must be named while the link exists"""
        self._stop_mtp_crash_watchdog()
        self._reset_effective_parallel_slots()
        # _process may be a test stand-in lacking terminate(); skip signalling, fall through to finally.
        self._diffusion_requested_ngl = None
        self._child_gpu_physical_ids = None
        if self._process is None:
            self._planned_vram_mib = {}
            return
        # Teardown records the process by identity: the reference stays set across the waits below.
        terminable = hasattr(self._process, "terminate")
        if not terminable:
            logger.debug("no terminable llama-server process to kill; clearing state")
        # Read both before terminate: getpgid fails once the leader is reaped.
        _pid = getattr(self._process, "pid", None)
        _pgid = self._leading_process_group(_pid)
        _descendants, _descendants_known = self._collect_descendants(_pid)
        _exited_first = getattr(self._process, "poll", lambda: None)() is not None
        if teardown:
            # Log between kill() and wait(): a TimeoutExpired from wait() escapes this handler unreported.
            self._torn_down_process = self._process
        try:
            if terminable:
                _grace = self._quit_signal_grace_left(_pid)
                if _grace is not None:
                    # The quit signal already reached it; a second one aborts llama-server on Metal.
                    try:
                        self._process.wait(timeout = _grace)
                    except subprocess.TimeoutExpired:
                        # Still listening: our signal never reached it (a stop sent to Studio alone).
                        if not self._still_listening():
                            self._process.kill()
                            self._process.wait(timeout = _SIGKILL_REAP_TIMEOUT_S)
                self._process.terminate()
                self._process.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            # Signal before logging: a closed stdout makes the logger raise and skip SIGKILL.
            self._process.kill()
            # Waiver is for real children only: a stand-in pid with no poll() could signal a stranger.
            logger.warning("llama-server did not exit on SIGTERM, sent SIGKILL")
            self._process.wait(timeout = _SIGKILL_REAP_TIMEOUT_S)
        except Exception as e:
            logger.warning(f"Error killing llama-server process: {e}")
        finally:
            self._kill_process_group(_pgid)
            _surviving_descendants = self._terminate_descendants(_descendants)
            self._stop_stats_logger()
            # Forget the pid only once confirmed gone, so a startup sweep can still reap survivors.
            _killed_pid = getattr(self._process, "pid", None)
            _exit_code = getattr(self._process, "poll", lambda: None)()
            _exited = _exit_code is not None
            # Only the read-back is kept: the tree kill already signalled this group; killpg is async.
            _owns_child = terminable and callable(getattr(self._process, "poll", None))
            if _owns_child and _killed_pid is not None and not _exited:
                if _pgid is not None and hasattr(os, "killpg"):
                    # Drives _wait_for_vram_settle in next load_model; set in finally so every kill path records it.
                    _exited = self._confirm_group_kill_landed(_killed_pid)
                else:
                    # On Windows terminate only hits the leader; taskkill /T /F is the last handle.
                    _exited = self._tree_kill_surviving_process(_killed_pid)
            # Keep the record while any descendant survives: pidfile and next reap key on it.
            if _surviving_descendants:
                _exited = False
            # Remove the pidfile only once exit is confirmed, since it is the next launch's only handle; with no child handle, drop it as before
            if not _descendants_known and _pid is not None:
                logger.warning(
                    "llama-server descendants could not be enumerated before the kill; "
                    "keeping its record so the next launch can reap anything left"
                )
                _exited = False
            if _killed_pid is not None and _exited:
                try:
                    from utils.process_lifetime import forget_pid
                    forget_pid(_killed_pid)
                except Exception:
                    pass
            self._process = None
            self._planned_vram_mib = {}
            # Remove the pidfile only once exit is confirmed; with no child handle drop it as before.
            if getattr(self, "_owns_pidfile", True) and (
                _killed_pid is None or _exited or not _owns_child
            ):
                self._clear_server_pid()
            # Clear healthy so a /load during the replacement's warm-up can't
            # short-circuit against the previous server's health (#5401).
            self._healthy = False
            # Unknown, so the training guard treats the loading server as VRAM-resident.
            self._gpu_offload_active = None
            self._gpu_offload_layers = None
            self._weight_placement = None
            self._offload_overridden = False
            self._gpu_backend_unavailable = False
            # Candidates are (pid, resolved binary, kill callable); shared handling below keeps paths agreeing.
            self._last_kill_monotonic = time.monotonic()
            stdout_thread = getattr(self, "_stdout_thread", None)
            if stdout_thread is not None:
                stdout_thread.join(timeout = 2)
                self._stdout_thread = None
            self._close_attempt_log(
                reason = "exited" if _exited_first else "killed", exit_code = _exit_code
            )

    def _other_planned_vram_mib(self) -> dict[int, int]:
        """MiB per GPU the other serving backends' running servers and the resident image/video models hold."""
        held: dict[int, int] = {}
        serving = _serving_backends_snapshot()
        if self not in serving:
            return held
        for backend in serving:
            if backend is self or not getattr(backend, "is_active", False):
                continue
            # A dead server keeps _process until respawned, but its VRAM is free.
            poll = getattr(getattr(backend, "_process", None), "poll", None)
            if callable(poll) and poll() is not None:
                continue
            for idx, mib in getattr(backend, "_planned_vram_mib", {}).items():
                held[idx] = held.get(idx, 0) + mib
        # Weights plus generation headroom, the full figure _net_of_held_vram needs (empty under Off).
        from core.inference.gpu_residency import media_held_mib

        for idx, mib in media_held_mib().items():
            held[idx] = held.get(idx, 0) + mib
        return held

    def _metal_media_mib(self) -> tuple[int, int]:
        """(price, unallocated) MiB of media resident on Apple's GPU, plus MLX chat growth beside them."""
        if self not in _serving_backends_snapshot():
            return 0, 0
        from core.inference.gpu_residency import (
            _mlx_chat_unallocated_mib,
            media_held_mib,
            media_unallocated_mib,
        )

        held = media_held_mib().get(0, 0)
        if not held:
            return 0, 0
        return held, media_unallocated_mib().get(0, 0) + (_mlx_chat_unallocated_mib() or 0)

    def _media_unallocated_vram_mib(self) -> dict[int, int]:
        """MiB per GPU of the resident image/video models' price not yet allocated (empty under Off)."""
        if self not in _serving_backends_snapshot():
            return {}
        from core.inference.gpu_residency import media_unallocated_mib

        return media_unallocated_mib()

    def _other_servers_running(self) -> bool:
        serving = _serving_backends_snapshot()
        if self not in serving:
            return False
        for backend in serving:
            process = getattr(backend, "_process", None) if backend is not self else None
            if process is not None and (not hasattr(process, "poll") or process.poll() is None):
                return True
        from core.inference.gpu_residency import media_held_mib

        return bool(media_held_mib())

    def _note_server_pid(self, pid: int) -> None:
        if getattr(self, "_owns_pidfile", True):
            self._record_server_pid(pid)
            return
        try:
            from utils.process_lifetime import adopt_pid
            adopt_pid(pid)
        except Exception as e:
            logger.debug(f"Could not track llama-server for lifetime sweep: {e}")

    @staticmethod
    def _server_pidfile_path() -> Optional[Path]:
        """Pidfile recording the live llama-server PID, under the active studio root
        (per-root, so concurrent Unsloth instances with distinct UNSLOTH_STUDIO_HOME stay
        isolated, mirroring the reaper's custom-root isolation)."""
        try:
            from utils.paths.storage_roots import studio_root  # noqa: WPS433
            return studio_root() / "llama-server.pid"
        except Exception:
            return None

    @classmethod
    def _record_server_pid(cls, pid: int) -> None:
        """Stores pid:starttime so a later startup can reject a recycled PID; a bare pid is still read."""
        # The lifetime record covers every child; the startup sweep reads it on macOS.
        try:
            from utils.process_lifetime import adopt_pid
            adopt_pid(pid)
        except Exception as e:
            logger.debug(f"Could not track llama-server for lifetime sweep: {e}")
        path = cls._server_pidfile_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_text(f"{pid}:{cls._pid_start_identity(pid)}", encoding = "utf-8")
        except Exception as e:
            logger.debug(f"Could not write llama-server pidfile: {e}")

    @classmethod
    def _clear_server_pid(cls) -> None:
        """Best-effort removal of the llama-server pidfile."""
        path = cls._server_pidfile_path()
        if path is None:
            return
        try:
            path.unlink(missing_ok = True)
        except Exception as e:
            logger.debug(f"Could not remove llama-server pidfile: {e}")

    @staticmethod
    def _pid_is_llama_server(pid: int) -> bool:
        """True only if pid is a live process whose binary is a llama-server. Guards
        against PID reuse before killing a recorded orphan; returns False on any
        uncertainty so an unrelated process is never killed."""
        try:
            import psutil
            try:
                proc = psutil.Process(pid)
                if (proc.name() or "").lower().startswith("llama-server"):
                    return True
                return Path(proc.exe() or "").name.lower().startswith("llama-server")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return False
        except ImportError:
            pass
        if sys.platform != "linux":
            return False
        try:
            if Path(os.readlink(f"/proc/{pid}/exe")).name.lower().startswith("llama-server"):
                return True
        except OSError:
            pass
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                tokens = fh.read().split(b"\x00")
            first = tokens[0].decode("utf-8", "replace") if tokens else ""
            return Path(first).name.lower().startswith("llama-server")
        except OSError:
            return False

    @staticmethod
    def _pid_start_identity(pid: int) -> str:
        """Process start time as a string ('' when unknown); a recycled PID will not match it."""
        try:
            import psutil
            try:
                return str(psutil.Process(pid).create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return ""
        except ImportError:
            pass
        if sys.platform == "linux":
            try:
                with open(f"/proc/{pid}/stat", "rb") as fh:
                    data = fh.read()
                # field 22 (starttime), counted from after the ")" that closes comm.
                return data[data.rfind(b")") + 2 :].split()[19].decode()
            except (OSError, IndexError):
                return ""
        return ""

    @staticmethod
    def _pid_parent_is_alive(pid: int) -> bool:
        """True when the parent still runs, so the server is not orphaned; uncertainty reads as alive."""
        try:
            import psutil

            try:
                ppid = psutil.Process(pid).ppid()
            except psutil.NoSuchProcess:
                return False  # the recorded server itself is gone
            except psutil.Error:
                return True  # cannot tell -- never risk killing a live server
            if ppid <= 1:
                return False  # reparented to init -> orphan
            return psutil.pid_exists(ppid)
        except ImportError:
            pass
        if sys.platform == "linux":
            try:
                with open(f"/proc/{pid}/stat", "rb") as fh:
                    data = fh.read()
                ppid = int(data[data.rfind(b")") + 2 :].split()[1])
            except (OSError, IndexError, ValueError):
                return False
            if ppid <= 1:
                return False
            return Path(f"/proc/{ppid}").exists()
        return True  # no psutil and no /proc: unknown

    @staticmethod
    def _unlink_pidfile(path: Path) -> None:
        """Best-effort removal of a resolved pidfile path."""
        try:
            path.unlink(missing_ok = True)
        except Exception:
            pass

    @classmethod
    def _reap_recorded_pid(cls) -> int:
        """Reaps only a true orphan: a live server whose parent still runs is never killed."""
        path = cls._server_pidfile_path()
        if path is None or not path.exists():
            return 0

        pid = -1
        identity = ""
        try:
            pid_str, _, identity = path.read_text(encoding = "utf-8").strip().partition(":")
            pid = int(pid_str)
        except Exception:
            pid = -1

        # Bound check before the cmdline check so pid 1 can never get through.
        if not _is_signalable_pid(pid):
            cls._unlink_pidfile(path)  # garbage record
            return 0
        if pid == os.getpid():
            return 0  # never our own pid; leave the record alone

        if cls._pid_parent_is_alive(pid):
            return 0

        # Parent is gone: candidate orphan. Reject a PID recycled to something else.
        if identity and cls._pid_start_identity(pid) != identity:
            cls._unlink_pidfile(path)
            return 0

        killed = 0
        if cls._pid_is_llama_server(pid):
            try:
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                killed = 1
                logger.info(f"Killed orphaned llama-server from pidfile (pid={pid})")
            except (ProcessLookupError, PermissionError):
                pass
            except Exception as e:
                logger.debug(f"Could not kill recorded llama-server pid {pid}: {e}")
        cls._unlink_pidfile(path)
        return killed

    @staticmethod
    def _kill_orphaned_servers() -> int:
        """Kills only servers under Unsloth roots or env overrides; candidates are gathered before
        any kill."""
        # Reap the recorded PID only if a true orphan, so a helper backend never kills a live server.
        killed = LlamaCppBackend._reap_recorded_pid()
        try:
            # -- Build the ownership allowlist --------------------------------
            # exact_binaries = env overrides; install_roots = Unsloth-owned trees.
            install_roots: list[Path] = []

            # Env-mode custom root (mirrors _find_llama_server_binary).
            _resolved_sr, _is_legacy = LlamaCppBackend._resolved_studio_root_and_is_legacy()
            _is_custom_root = not _is_legacy
            if _is_custom_root:
                install_roots.append(_resolved_sr / "llama.cpp")

            # Env mode skips this so a custom-root Unsloth can't kill a default install's server.
            if not _is_custom_root:
                install_roots.append(Path.home() / ".unsloth" / "llama.cpp")

            # Legacy in-tree build dirs (older setup.sh)
            project_root = Path(__file__).resolve().parents[5]
            install_roots.append(project_root / "llama.cpp")

            # Legacy: extracted binary
            install_roots.append(project_root / "bin")

            # UNSLOTH_LLAMA_CPP_PATH env var (custom install dir)
            custom_dir = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
            if custom_dir:
                # expanduser to match _find_llama_server_binary, or we cannot clean up our own server.
                install_roots.append(_expanded_user_path(custom_dir))

            # LLAMA_SERVER_PATH env var (exact binary path)
            exact_binaries: list[Path] = []
            env_binary = os.environ.get("LLAMA_SERVER_PATH")
            if env_binary:
                try:
                    exact_binaries.append(Path(env_binary).resolve())
                except OSError:
                    pass

            # Resolve all roots so is_relative_to works reliably.
            resolved_roots: list[Path] = []
            for root in install_roots:
                try:
                    # Local-link roots resolve into the user's checkout; skip so we never kill their server.
                    if _is_external_link(root):
                        continue
                    resolved_roots.append(root.resolve())
                except OSError:
                    pass

            my_pid = os.getpid()

            # -- Enumerate candidate processes ---------------------------------
            # (pid, resolved binary, kill callable) per llama-server-looking process;
            # ownership, orphan and kill handling is shared below so the paths agree.
            candidates = []

            def _looks_like_server(name):
                return bool(name) and name.lower().startswith("llama-server")

            def _stat_start_time(line):
                right = line.rfind(b")")
                fields = line[right + 2 :].split()
                return fields[19] if len(fields) > 19 else None

            def _proc_start_time(proc_dir):
                """Field 22 of /proc/<pid>/stat, in jiffies since boot."""
                try:
                    with open(f"{proc_dir}/stat", "rb") as fh:
                        line = fh.read()
                except OSError:
                    return None
                right = line.rfind(b")")
                if right < 0:
                    return None
                # Fields after comm start at field 3, so starttime is index 19.
                fields = line[right + 2 :].split()
                return fields[19] if len(fields) > 19 else None

            def _make_signal_killer(proc_dir, pid, start_time):
                def _kill():
                    # Re-check starttime: a reused PID gets a later one, so a mismatch is not our process.
                    if start_time is None or _proc_start_time(proc_dir) != start_time:
                        return False
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        return False
                    return True

                return _kill

            # /proc is Linux only; fall back to psutil. Tests repoint _PROC_ROOT.
            proc_root = _PROC_ROOT
            if sys.platform == "linux" and os.path.isdir(proc_root):
                # Faster than psutil.process_iter: read comm once, resolve exe only on a name match.
                def _proc_exe(proc_dir):
                    try:
                        target = os.readlink(f"{proc_dir}/exe")
                    except OSError:
                        return None
                    if not target:
                        return None
                    # Strip the kernel's ' (deleted)' suffix like psutil does, or upgrade orphans are missed.
                    if target.endswith(" (deleted)") and not os.path.exists(target):
                        target = target[: -len(" (deleted)")]
                    try:
                        return Path(target).resolve()
                    except OSError:
                        return None

                with os.scandir(proc_root) as entries:
                    for entry in entries:
                        if not entry.name.isdigit():
                            continue
                        try:
                            pid = int(entry.name)
                        except ValueError:
                            continue
                        if pid == my_pid:
                            continue
                        try:
                            with open(f"{entry.path}/stat", "rb") as fh:
                                stat_line = fh.read()
                        except OSError:
                            continue  # the process exited mid-scan
                        left = stat_line.find(b"(")
                        right = stat_line.rfind(b")")
                        if left < 0 or right <= left:
                            continue
                        # comm is truncated to 15 bytes, which keeps the 'llama-server' prefix.
                        name = stat_line[left + 1 : right].decode("utf-8", "replace")
                        if not _looks_like_server(name):
                            continue
                        exe_path = _proc_exe(entry.path)
                        if exe_path is None:
                            continue
                        candidates.append(
                            (
                                pid,
                                exe_path,
                                _make_signal_killer(entry.path, pid, _stat_start_time(stat_line)),
                            )
                        )
            else:
                try:
                    import psutil
                except ImportError:
                    psutil = None

                if psutil is not None:

                    def _make_psutil_killer(psutil_mod, proc):
                        def _kill():
                            try:
                                proc.kill()
                            except (
                                psutil_mod.NoSuchProcess,
                                psutil_mod.AccessDenied,
                                psutil_mod.ZombieProcess,
                            ):
                                return False
                            return True

                        return _kill

                    for proc in psutil.process_iter(["pid", "name", "exe"]):
                        try:
                            pid = proc.info["pid"]
                            if pid == my_pid:
                                continue
                            if not _looks_like_server(proc.info.get("name") or ""):
                                continue
                            exe = proc.info.get("exe")
                            if not exe:
                                continue
                            candidates.append(
                                (pid, Path(exe).resolve(), _make_psutil_killer(psutil, proc))
                            )
                        except (
                            psutil.NoSuchProcess,
                            psutil.AccessDenied,
                            psutil.ZombieProcess,
                        ):
                            pass
                else:
                    # Skip before writing if the estimate alone exceeds the cap, not after a full write and discard.
                    return killed

            # -- Ownership check, orphan check, kill ---------------------------
            for pid, binary, kill in candidates:
                if not _is_signalable_pid(pid):
                    continue
                # Ownership: exact match OR binary under a known root.
                is_ours = binary in exact_binaries or any(
                    binary.is_relative_to(root) for root in resolved_roots
                )
                if not is_ours:
                    continue

                # A live parent means a running Unsloth (or the user's shell) still owns it.
                if LlamaCppBackend._pid_parent_is_alive(pid):
                    continue

                if kill():
                    killed += 1
                    logger.info(f"Killed orphaned llama-server process (pid={pid})")
        except Exception:
            logger.warning("Error during orphan server cleanup", exc_info = True)
        return killed

    def _cleanup(self):
        """At atexit log streams may be closed, so it must not log errors; stdlib needs
        raiseExceptions off."""
        raise_exceptions = logging.raiseExceptions
        logging.raiseExceptions = False
        try:
            self._kill_process(teardown = True)
            # Returns a -ot spill plan or None to abstain on any uncertain input (argv stays byte-identical to --fit on); must NOT run when anything else owns placement
            self._cleanup_cpu_fallback_runtime()
        except Exception:
            # atexit swallows this anyway, and there is nowhere left to report it.
            pass
        finally:
            logging.raiseExceptions = raise_exceptions
