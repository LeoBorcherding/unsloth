# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: port allocation and stdout drain."""

import os
import socket
import struct
import tempfile
import time

from pathlib import Path
from typing import Optional

from .core_types import _GPU_DEVICE_LOST_MARKERS
from .gguf_cache import (
    _cached_candidate_matches_revision_size,
    _cached_complete_candidate,
    _cached_variant_sizes,
    _resolve_repo_id_casing,
    _resolve_variant_gguf_files,
    cached_gguf_for_load,
)
from .hf_offline import (
    _GGUF_UINT_WIDTHS,
    _MISS,
    _hf_offline_if_unreachable,
    _hf_repo_from_url,
)
from .log import logger
from .reasoning import (
    _note_gguf_metadata_read,
    detect_reasoning_flags,
)
from .server_log import (
    _LLAMA_LOG_FULL_BYTES,
    _LOG_LEVEL_TOKEN_RE,
    _STDOUT_HEAD_MAX,
    _STDOUT_TAIL_KEEP,
    _STDOUT_TRIM_AT,
    _VULKAN_DIFFUSION_GPU_IDS_ERROR,
    _llama_line_is_warning_or_error,
)
from .swa_cache import _resolve_swa_pattern
from .binary_runtime import (
    _HOST_RAM_HEADROOM_MIB,
)
from .core_types import (
    _DRAFT_CONTEXT_FAILURE_MARKERS,
    _METAL_BACKEND_POISONED_MARKERS,
)
from .draft_args import (
    _extra_args_device,
)
from .hf_offline import (
    _GGUF_MARKUP_FIRST_BYTES,
)
from .server_log import (
    _LOOPBACK_REACHABLE_BIND_IPS,
    _socket_owner_is_always_named,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _ProcessIoMixin:
    # ── Port allocation ───────────────────────────────────────────

    @staticmethod
    def _find_free_port() -> int:
        """Find an available TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @staticmethod
    def _health_listener_is_ours(
        process,
        port,
        bind_host = None,
    ) -> bool:
        """False when the port answering /health is demonstrably not held by our child's tree.

        _find_free_port releases the port before llama-server binds it, so another local user could
        answer /health and receive every prompt plus the per-launch key. Their socket shows no pid
        on Linux/macOS, so our own (readable) tree not listening counts as foreign. Unreadable
        ownership keeps today's behaviour. On Windows psutil can report our live listener as NONE
        and/or pid 0 (#15022), so only a named foreign pid refuses there."""
        pid = getattr(process, "pid", None)
        if not isinstance(pid, int) or not isinstance(port, int):
            return True
        # A child that already exited cannot be the server that answered.
        poll = getattr(process, "poll", None)
        if callable(poll) and poll() is not None:
            return False
        try:
            import psutil
            from core.inference.sd_cpp_server import _has_ancestor
        except Exception:  # noqa: BLE001 -- optional dependency
            return True
        try:
            root = psutil.Process(pid)
            tree = [root, *root.children(recursive = True)]
        except psutil.NoSuchProcess:
            return False
        except Exception:  # noqa: BLE001 -- access denied / platform quirk
            return True

        windows = _socket_owner_is_always_named()
        bind_ips = set(_LOOPBACK_REACHABLE_BIND_IPS)
        if bind_host:
            try:
                bind_ips.update(a[4][0] for a in socket.getaddrinfo(bind_host, port))
            except OSError:
                pass

        def _listens(conn) -> bool:
            laddr = getattr(conn, "laddr", None)
            if not (
                bool(laddr)
                and getattr(laddr, "port", None) == port
                and getattr(laddr, "ip", None) in bind_ips
            ):
                return False
            if conn.status == psutil.CONN_LISTEN:
                return True
            # Windows can report the live listener as NONE: a TCP socket with no peer still is one.
            return (
                windows
                and conn.status == psutil.CONN_NONE
                and not getattr(conn, "raddr", None)
                and getattr(conn, "type", socket.SOCK_STREAM) == socket.SOCK_STREAM
            )

        tree_readable = True
        for member in tree:
            try:
                read = getattr(member, "net_connections", None) or member.connections
                if any(_listens(c) for c in read(kind = "inet")):
                    return True
            except psutil.NoSuchProcess:
                continue
            except Exception:  # noqa: BLE001 -- access denied / platform quirk
                tree_readable = False
        tree_pids = {m.pid for m in tree}
        foreign = None
        try:
            # Scan every listener: a foreign one listed before ours must not refuse our start.
            for conn in psutil.net_connections(kind = "inet"):
                # pid 0 is the System Idle Process on Windows: psutil's other "unknown owner".
                if not _listens(conn) or conn.pid is None or (windows and conn.pid == 0):
                    continue
                if conn.pid in tree_pids or _has_ancestor(conn.pid, pid):
                    return True
                foreign = conn.pid
        except Exception as exc:  # noqa: BLE001 -- permissions / platform quirks
            logger.debug("llama-server port ownership check unavailable: %s", exc)
        if foreign is not None:
            logger.error(
                "llama-server port %s is held by pid %s, not our child %s; refusing to use it",
                port,
                foreign,
                pid,
            )
            return False
        if tree_readable and not windows:
            logger.error(
                "llama-server port %s answered /health but our child %s is not listening on it; "
                "refusing to use it",
                port,
                pid,
            )
            return False
        return True

    # ── Stdout drain (prevents pipe deadlock on Windows) ─────────

    def _drain_stdout(self):
        """Drains llama-server stdout in a background thread so a full ~4 KB Windows pipe cannot block it; tees each line to _llama_log_fh when set."""
        startup_len: Optional[int] = None
        log_bytes = 0
        levelled = to_info = False
        proc = self._process
        device_lost_seen = False
        metal_poison = {"draft": False}
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    lines = self._stdout_lines
                    lines.append(line)
                    if len(lines) > _STDOUT_TRIM_AT:
                        # Before readiness the whole log so far is startup: keep its head too.
                        head = _STDOUT_HEAD_MAX if startup_len is None else startup_len
                        head = min(head, _STDOUT_HEAD_MAX, len(lines))
                        del lines[head : len(lines) - _STDOUT_TAIL_KEEP]
                    # Only readiness lines wake the probe; waking per tensor-load line spins startup.
                    line_lower = line.lower()
                    ready = "server is listening" in line_lower or "model loaded" in line_lower
                    if ready:
                        if startup_len is None:
                            startup_len = len(self._stdout_lines)
                        health_probe_event = getattr(self, "_health_probe_event", None)
                        if health_probe_event is not None:
                            health_probe_event.set()
                    # Unprefixed lines continue the previous record (e.g. a request dump).
                    level = _LOG_LEVEL_TOKEN_RE.match(line)
                    if level is not None:
                        levelled = True
                        to_info = level.group(1) in "WE" or (ready and level.group(1) == "I")
                        # E lines only: unprefixed lines can be a request dump with user text.
                        if level.group(1) == "E":
                            if any(m in line for m in _DRAFT_CONTEXT_FAILURE_MARKERS):
                                metal_poison["draft"] = True
                            # Startup OOM has its own retries; this one is a served child's.
                            if startup_len is not None and self._is_gpu_memory_start_failure(line):
                                self._gpu_oom_process = proc
                            if not device_lost_seen and any(
                                marker in line for marker in _GPU_DEVICE_LOST_MARKERS
                            ):
                                device_lost_seen = True
                                self._retire_device_lost_server_soon(proc)
                            elif (
                                not device_lost_seen
                                and not metal_poison.get("pending")
                                and any(
                                    marker in line for marker in _METAL_BACKEND_POISONED_MARKERS
                                )
                            ):
                                # Not latched: a request can echo this text, so retirement confirms
                                # a compute failure first.
                                metal_poison["pending"] = True
                                self._retire_device_lost_server_soon(
                                    proc, metal_poison = metal_poison
                                )
                    elif not levelled:
                        to_info = ready or _llama_line_is_warning_or_error(line, line_lower)
                    try:
                        if to_info:
                            logger.info(f"[llama-server] {line}")
                        else:
                            logger.debug(f"[llama-server] {line}")
                    except Exception:
                        # A closed stdout must not stop the drain and stall the server.
                        pass
                    fh = getattr(self, "_llama_log_fh", None)
                    if fh is not None:
                        try:
                            if log_bytes >= _LLAMA_LOG_FULL_BYTES:
                                if not _llama_line_is_warning_or_error(line, line_lower):
                                    continue
                            elif log_bytes + len(line) + 1 >= _LLAMA_LOG_FULL_BYTES:
                                fh.write(
                                    "[studio] log past 64 MiB: keeping warnings and errors only\n"
                                )
                            log_bytes += len(line) + 1
                            fh.write(line + "\n")
                            fh.flush()
                        except ValueError:
                            # Closed under us by the kill path.
                            pass
                        except OSError as e:
                            if not getattr(self, "_llama_log_tee_failed", False):
                                self._llama_log_tee_failed = True
                                try:
                                    logger.warning(
                                        f"Could not write llama-server log "
                                        f"{getattr(self, '_llama_log_path', None)}: {e}"
                                    )
                                except Exception:
                                    pass
        except Exception:
            # Never let the drain die: a full stdout pipe can deadlock llama-server on Windows.
            logger.debug("llama-server stdout drain stopped", exc_info = True)
        finally:
            # EOF means the process exited; wake the health loop immediately.
            health_probe_event = getattr(self, "_health_probe_event", None)
            if health_probe_event is not None:
                health_probe_event.set()

    # GGUF KV type sizes for fast skipping
    _GGUF_TYPE_SIZE = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }

    @staticmethod
    def _gguf_skip_value(f, vtype: int) -> None:
        """Skip a GGUF KV value without reading it."""
        sz = LlamaCppBackend._GGUF_TYPE_SIZE.get(vtype)
        if sz is not None:
            f.seek(sz, 1)
        elif vtype == 8:  # STRING
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
        elif vtype == 9:  # ARRAY
            atype = struct.unpack("<I", f.read(4))[0]
            alen = struct.unpack("<Q", f.read(8))[0]
            elem_sz = LlamaCppBackend._GGUF_TYPE_SIZE.get(atype)
            if elem_sz is not None:
                f.seek(elem_sz * alen, 1)
            elif atype == 8:
                from utils.models.gguf_metadata import skip_gguf_strings
                for _ in range(alen - skip_gguf_strings(f, alen)):
                    slen = struct.unpack("<Q", f.read(8))[0]
                    f.seek(slen, 1)
            else:
                for _ in range(alen):
                    LlamaCppBackend._gguf_skip_value(f, atype)
        else:
            # Its size is unknown, so every later read would be misaligned.
            raise ValueError(f"unknown GGUF value type {vtype}")

    @staticmethod
    def _gguf_read_array_value(f, atype: int, alen: int) -> Optional[list]:
        if atype == 4:  # UINT32
            return [struct.unpack("<I", f.read(4))[0] for _ in range(alen)]
        if atype == 5:  # INT32
            return [struct.unpack("<i", f.read(4))[0] for _ in range(alen)]
        if atype == 7:  # BOOL
            return [struct.unpack("<?", f.read(1))[0] for _ in range(alen)]

        if atype == 8:  # STRING: kept only for the delimiter-shaped entries (#7066).
            from core.inference.chat_template_helpers import delimiter_shaped_tokens
            from utils.models.gguf_metadata import take_gguf_strings_by_first_byte

            done, raws = take_gguf_strings_by_first_byte(f, alen, _GGUF_MARKUP_FIRST_BYTES)
            kept: list = []
            for raw in raws:
                try:
                    kept.append(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    continue
            for _ in range(alen - done):
                n = struct.unpack("<Q", f.read(8))[0]
                raw = f.read(n)
                # Only <...> / [...] entries are kept, so skip decoding the rest; UTF-8 makes the first byte
                # decisive. Still read, not seeked, so short reads and non-seekable streams behave the same.
                if raw[:1] != b"<" and raw[:1] != b"[":
                    continue
                # Undecodable entries are not markers.
                try:
                    kept.append(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    continue
            return delimiter_shaped_tokens(kept)

        for _ in range(alen):
            LlamaCppBackend._gguf_skip_value(f, atype)
        return None

    @classmethod
    def _gguf_path_is_diffusion(cls, gguf_path: str, model_identifier: str) -> bool:
        """Classify a downloaded GGUF without mutating the active backend."""
        probe = object.__new__(cls)
        probe._model_identifier = model_identifier
        probe._read_gguf_metadata(gguf_path)
        return probe._is_diffusion

    @classmethod
    def _gguf_path_wants_nested_mtp(cls, gguf_path: str) -> bool:
        """May this target take an MTP/ drafter? Only qwen4exp with no head of its own; arch-gated since MTP/ may cover some quants only; fails closed."""
        try:
            probe = object.__new__(cls)
            probe._model_identifier = "mtp-head-probe"
            probe._read_gguf_metadata(gguf_path)
            return probe._architecture == "qwen4exp" and not probe._nextn_predict_layers
        except Exception as e:
            logger.debug("Nested MTP eligibility probe failed for %s: %s", gguf_path, e)
            return False

    def _reject_vulkan_diffusion_gpu_ids_before_teardown(
        self, gguf_path: str, model_identifier: str
    ) -> None:
        """Reject Vulkan + gpu_ids for diffusion GGUFs before Phase 1 teardown."""
        if self._gguf_path_is_diffusion(gguf_path, model_identifier):
            raise ValueError(_VULKAN_DIFFUSION_GPU_IDS_ERROR)

    @classmethod
    def _non_chat_gguf_refusal_for_path(
        cls, gguf_path: str, model_identifier: Optional[str]
    ) -> Optional[str]:
        """Header check on a probe object, not self (which would overwrite live metadata), run before Phase 1 teardown so a refusal costs no resident model."""
        probe = object.__new__(cls)
        probe._model_identifier = model_identifier
        probe._read_gguf_metadata(gguf_path)
        return probe._non_chat_gguf_refusal(gguf_path)

    @classmethod
    def _verified_cached_gguf(cls, intent, hf_repo, hf_variant) -> Optional[str]:
        """Return a carried file only if it still matches the load and active cache."""
        verified = getattr(intent, "verified_gguf", None)
        if not verified:
            return None
        try:
            repo, variant, path, sizes = verified
        except (TypeError, ValueError):
            return None
        if (repo or "").lower() != (hf_repo or "").lower():
            return None
        if (variant or "").lower() != (hf_variant or "").lower():
            return None
        if not path:
            return None
        try:
            if not Path(path).is_file():
                return None
            from utils.hf_cache_settings import get_hf_cache_paths
            hub_cache = Path(os.path.abspath(get_hf_cache_paths().hub_cache))
        except Exception:  # noqa: BLE001 -- an unreadable cache setting is not a reusable path
            return None
        # Cache settings can change while a load is in flight.
        try:
            if not Path(os.path.abspath(path)).is_relative_to(hub_cache):
                return None
        except (OSError, ValueError):
            return None
        # Recheck every shard's recorded size against this snapshot; both local, no Hub calls.
        try:
            if _cached_variant_sizes(_resolve_repo_id_casing(repo), variant, path) != sizes:
                return None
        except OSError:
            return None
        return path

    @classmethod
    def non_chat_gguf_refusal_for_intent(cls, intent) -> Optional[str]:
        """Asks before the route evicts a resident Images/Video pipeline; fails open like the in-
        load checks."""
        try:
            gguf_path = getattr(intent, "gguf_path", None)
            hf_repo = getattr(intent, "hf_repo", None)
            identifier = getattr(intent, "model_identifier", None)
            if gguf_path and not hf_repo and Path(gguf_path).is_file():
                return cls._non_chat_gguf_refusal_for_path(gguf_path, identifier)
            if hf_repo:
                hf_variant = getattr(intent, "hf_variant", None)
                verified = cls._verified_cached_gguf(intent, hf_repo, hf_variant)
                if verified:
                    verdict = cls._non_chat_gguf_refusal_for_path(verified, identifier)
                    cls._hand_over_route_verdict(hf_repo, hf_variant, verdict)
                    return verdict
                # Same offline window as the download, so Hub calls do not wait out retry backoff.
                with _hf_offline_if_unreachable():
                    verdict = cls._remote_non_chat_gguf_verdict(
                        hf_repo = hf_repo,
                        hf_variant = hf_variant,
                        hf_token = getattr(intent, "hf_token", None),
                        model_identifier = identifier,
                    )
                cls._hand_over_route_verdict(hf_repo, hf_variant, verdict)
                return verdict
        except Exception as e:  # noqa: BLE001 -- a probe that failed is not a verdict
            logger.debug("Non-chat GGUF preflight failed for the route: %s", e)
        return None

    def host_offload_warning_for_intent(self, intent) -> Optional[str]:
        """Advisory only, never a refusal: prices physical RAM and VRAM totals, not free memory; local paths only, never a download."""
        try:
            gguf_path = getattr(intent, "gguf_path", None)
            if not gguf_path or getattr(intent, "hf_repo", None):
                return None
            if not Path(gguf_path).is_file():
                return None
            binary = self._find_llama_server_binary()
            if not binary:
                return None
            # extras are appended after -m at launch and llama.cpp is last-wins, so read them
            argv = [
                binary,
                "-m",
                str(gguf_path),
                *(str(arg) for arg in getattr(intent, "extra_args", None) or ()),
            ]
            total_ram_mib = self._total_system_memory_mib()
            if total_ram_mib is None:
                return None
            if self._host_offload_fits_without_gpus(argv, total_ram_mib):
                return None
            probed = self._get_gpu_memory(binary)
            # an igpu and a MIG/vGPU line report total 0, leaving the ceiling unknown
            if any(total <= 0 for _idx, _free, total in probed):
                return None
            return self._launch_host_shortfall_message(
                argv,
                [(idx, total) for idx, _free, total in probed],
                avail_mib = total_ram_mib,
            )
        except Exception as e:  # noqa: BLE001 -- a probe that failed is not a verdict
            logger.debug("Host-RAM preflight failed for the route: %s", e)
        return None

    def _host_offload_fits_without_gpus(self, argv: list[str], total_ram_mib: int) -> bool:
        """Whole GGUF fits RAM less headroom: every credit (VRAM, lazy read) only lowers the
        priced bytes, so no GPU probe can warn. False when unsure or opted out (that path logs)."""
        if self._host_offload_warning_opted_out():
            return False
        model_path = _extra_args_device(argv, self._ARGV_MODEL)
        if not model_path:
            return False
        try:
            model_bytes = self._get_gguf_size_bytes(model_path)
        except Exception:  # noqa: BLE001
            return False
        return 0 < model_bytes and model_bytes / (1024 * 1024) <= (
            total_ram_mib - _HOST_RAM_HEADROOM_MIB
        )

    # Each ask may take several 15s Hub calls, so the route hands its verdict to the load.
    # Taken once by its own load; the TTL only covers a route ask no load came for.
    _HANDOFF_TTL_S = 120.0
    _route_verdict_handoff: Optional[
        tuple[tuple[Optional[str], Optional[str]], Optional[str], float]
    ] = None

    @classmethod
    def _take_route_verdict(cls, hf_repo, hf_variant):
        """The verdict the route just took for this key, or ``_MISS``. Consumed on read."""
        handoff = cls._route_verdict_handoff
        if handoff is None:
            return _MISS
        key, verdict, at = handoff
        if key != (hf_repo, hf_variant) or (time.monotonic() - at) >= cls._HANDOFF_TTL_S:
            return _MISS
        cls._route_verdict_handoff = None
        return verdict

    @classmethod
    def _hand_over_route_verdict(cls, hf_repo, hf_variant, verdict) -> None:
        cls._route_verdict_handoff = ((hf_repo, hf_variant), verdict, time.monotonic())

    @classmethod
    def _remote_non_chat_gguf_refusal(
        cls,
        *,
        hf_repo: str,
        hf_variant: Optional[str],
        hf_token: Optional[str],
        model_identifier: Optional[str],
    ) -> Optional[str]:
        """Decides a repo load from a 256 KiB header prefix before Phase 1 teardown, without a download; fails open, never stricter than the full-file check."""
        handed_over = cls._take_route_verdict(hf_repo, hf_variant)
        if handed_over is not _MISS:
            logger.debug("Reusing the header verdict the route already took for %s", hf_repo)
            return handed_over
        return cls._remote_non_chat_gguf_verdict(
            hf_repo = hf_repo,
            hf_variant = hf_variant,
            hf_token = hf_token,
            model_identifier = model_identifier,
        )

    @classmethod
    def _remote_non_chat_gguf_verdict(
        cls,
        *,
        hf_repo: str,
        hf_variant: Optional[str],
        hf_token: Optional[str],
        model_identifier: Optional[str],
    ) -> Optional[str]:
        """The uncached probe: one listing, one cache resolution, one range read."""
        try:
            from core.inference.diffusion_compat import (
                _read_gguf_header,
                _read_local_header,
            )
        except Exception as e:  # noqa: BLE001 -- an unavailable probe is not a verdict
            logger.debug("GGUF header probe unavailable: %s", e)
            return None
        try:
            gguf_filename, shards = _resolve_variant_gguf_files(hf_repo, hf_variant, hf_token)
            if not gguf_filename:
                return None
            # Resolve with the same precedence and verification as _download_gguf so the probe judges
            # the file the load will open. verify_sizes fails open offline.
            local = cached_gguf_for_load(hf_repo, hf_variant, verify_sizes = True, hf_token = hf_token)
            if local is None and not hf_variant:
                # No variant: mirror _download_gguf's other half. Never the bare local path, which would
                # bring back a candidate the verified lookup rejected.
                candidate = _cached_complete_candidate(hf_repo, gguf_filename, list(shards))
                if candidate is not None and _cached_candidate_matches_revision_size(
                    hf_repo, candidate, hf_token
                ):
                    local = candidate[0]
            prefix = (
                _read_local_header(local)
                if local
                else _read_gguf_header(hf_repo, gguf_filename, hf_token)
            )
            # Magic, version and the two counts: anything shorter is not a GGUF at all.
            if len(prefix) < 24:
                return None
            with tempfile.TemporaryDirectory(prefix = "unsloth-gguf-probe-") as probe_dir:
                # Named after the real file: a GGUF declaring no architecture is judged by its name, and a temp name would lose the page pointer.
                probe_path = os.path.join(probe_dir, os.path.basename(gguf_filename))
                with open(probe_path, "wb") as handle:
                    handle.write(prefix)
                return cls._non_chat_gguf_refusal_for_path(probe_path, model_identifier)
        except Exception as e:  # noqa: BLE001 -- a probe that failed is not a verdict
            logger.debug("Non-chat GGUF header probe failed for %s: %s", hf_repo, e)
            return None

    def _read_gguf_metadata(self, gguf_path: str) -> None:
        """Metadata lives in shard 1 of split GGUFs; per-model flags are reset first so none carry over."""
        # Reset so stale flags do not carry over when switching models.
        self._context_length = None
        self._chat_template = None
        self._empty_chat_render_refused = False
        self._markup_tokens = []
        self._markup_profile = None
        self._supports_reasoning = False
        self._reasoning_always_on = False
        self._reasoning_style = "enable_thinking"
        self._reasoning_effort_levels = []
        self._reasoning_default = True
        self._supports_preserve_thinking = False
        self._preserve_thinking_default = False
        self._supports_tools = False
        self._n_layers = None
        self._n_experts = None
        self._leading_dense_block_count = None
        self._n_kv_heads = None
        self._n_kv_heads_by_layer = None
        self._n_heads = None
        self._embedding_length = None
        self._pooling_type = None
        self._gguf_path = gguf_path
        self._feed_forward_length = None
        self._feed_forward_length_by_layer = None
        self._expert_used_count = None
        self._expert_feed_forward_length = None
        self._expert_shared_feed_forward_length = None
        self._expert_shared_count = None
        self._moe_latent_size = None
        self._embedding_length_per_layer_input = None
        self._vocab_size = None
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
        self._architecture = None
        self._gguf_type = None
        self._is_diffusion = False
        self._gguf_header_parsed = False
        self._gguf_split_index = None

        try:
            canvas_seen = False
            WANTED = {
                "general.architecture",
                "general.type",
                "tokenizer.chat_template",
                # Vocab size = tokens array length (no vocab_size key in many GGUFs).
                "tokenizer.ggml.tokens",
                # Block-diffusion marker (DiffusionGemma); routes to the diffusion runner.
                "diffusion.canvas_length",
                # Source-repo hints for the SWA resolver's HF fallback.
                "general.source.huggingface.repository",
                "general.source.url",
                "general.source.repo_url",
                "general.base_model.0.repo_url",
                "general.base_model.0.organization",
                "general.base_model.0.name",
                "general.basename",
                "general.organization",
                "general.size_label",
                "general.finetune",
            }
            # Arch-specific keys added dynamically once we know the arch.
            arch_keys: dict[str, str] = {}  # gguf_key -> attribute name
            arch = None
            pooling_by_arch: dict[str, int] = {}

            nextn_by_arch: dict[str, int] = {}
            sliding_window_pattern_period: Optional[int] = None
            general: dict[str, str] = {}

            with open(gguf_path, "rb") as f:
                magic = struct.unpack("<I", f.read(4))[0]
                if magic != 0x46554747:  # b"GGUF" as little-endian u32
                    return
                _version = struct.unpack("<I", f.read(4))[0]
                _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))

                # Set by the loop's `else` below: True only if no truncation `break` fired.
                kv_complete = False
                for _ in range(kv_count):
                    # Tolerate truncated input (partial HTTP range header): stop and let the resolver fallback run on what was parsed.
                    try:
                        key_len_bytes = f.read(8)
                        if len(key_len_bytes) < 8:
                            break
                        key_len = struct.unpack("<Q", key_len_bytes)[0]
                        key_bytes = f.read(key_len)
                        if len(key_bytes) < key_len:
                            break
                        key = key_bytes.decode("utf-8")
                        vtype_bytes = f.read(4)
                        if len(vtype_bytes) < 4:
                            break
                        vtype = struct.unpack("<I", vtype_bytes)[0]
                        # gguf-split puts all metadata in shard 1; later shards carry only split.*
                        # keys. Read the index,
                        # not presence, since shard 1 has these keys too.
                        if key == "split.no":
                            width = _GGUF_UINT_WIDTHS.get(vtype)
                            if width is not None:
                                raw = f.read(width)
                                if len(raw) < width:
                                    break
                                self._gguf_split_index = int.from_bytes(raw, "little")
                                continue
                    except (struct.error, UnicodeDecodeError):
                        break

                    try:
                        if (
                            key in WANTED
                            or key in arch_keys
                            or key.endswith(".pooling_type")
                            or key.endswith(".nextn_predict_layers")
                        ):
                            if vtype == 8:  # STRING
                                slen = struct.unpack("<Q", f.read(8))[0]
                                val_s = f.read(slen).decode("utf-8")
                                if key.startswith("general.") and key != "general.architecture":
                                    general[key] = val_s
                                if key == "general.type":
                                    self._gguf_type = val_s.strip().lower()
                                if key == "general.architecture":
                                    arch = val_s
                                    self._architecture = val_s
                                    arch_keys = {
                                        f"{arch}.context_length": "context_length",
                                        f"{arch}.block_count": "n_layers",
                                        f"{arch}.expert_count": "n_experts",
                                        f"{arch}.leading_dense_block_count": "leading_dense_block_count",
                                        f"{arch}.attention.head_count_kv": "n_kv_heads",
                                        f"{arch}.attention.head_count": "n_heads",
                                        f"{arch}.embedding_length": "embedding_length",
                                        f"{arch}.pooling_type": "pooling_type",
                                        f"{arch}.feed_forward_length": "feed_forward_length",
                                        f"{arch}.expert_used_count": "expert_used_count",
                                        f"{arch}.expert_feed_forward_length": "expert_feed_forward_length",
                                        f"{arch}.expert_shared_feed_forward_length": "expert_shared_feed_forward_length",
                                        f"{arch}.expert_shared_count": "expert_shared_count",
                                        f"{arch}.moe_latent_size": "moe_latent_size",
                                        f"{arch}.embedding_length_per_layer_input": "embedding_length_per_layer_input",
                                        f"{arch}.attention.key_length": "kv_key_length",
                                        f"{arch}.attention.value_length": "kv_value_length",
                                        f"{arch}.attention.sliding_window": "sliding_window",
                                        f"{arch}.attention.sliding_window_pattern": "sliding_window_pattern",
                                        f"{arch}.full_attention_interval": "full_attention_interval",
                                        f"{arch}.attention.kv_lora_rank": "kv_lora_rank",
                                        f"{arch}.attention.key_length_mla": "key_length_mla",
                                        f"{arch}.attention.key_length_swa": "kv_key_length_swa",
                                        f"{arch}.attention.value_length_swa": "kv_value_length_swa",
                                        f"{arch}.attention.shared_kv_layers": "shared_kv_layers",
                                        f"{arch}.ssm.inner_size": "ssm_inner_size",
                                        f"{arch}.ssm.state_size": "ssm_state_size",
                                        f"{arch}.ssm.group_count": "ssm_group_count",
                                        f"{arch}.ssm.conv_kernel": "ssm_conv_kernel",
                                        f"{arch}.kda.head_dim": "kda_head_dim",
                                        f"{arch}.nextn_predict_layers": "nextn_predict_layers",
                                    }
                                elif key == "tokenizer.chat_template":
                                    self._chat_template = val_s
                            elif vtype in (4, 10):  # UINT32 or UINT64
                                val_i = (
                                    struct.unpack("<I", f.read(4))[0]
                                    if vtype == 4
                                    else struct.unpack("<Q", f.read(8))[0]
                                )
                                if key == "diffusion.canvas_length":
                                    canvas_seen = True
                                if key.endswith(".pooling_type"):
                                    pooling_by_arch[key] = val_i

                                if key.endswith(".nextn_predict_layers"):
                                    nextn_by_arch[key] = val_i
                                attr = arch_keys.get(key)
                                if attr:
                                    if attr == "sliding_window_pattern":
                                        sliding_window_pattern_period = val_i
                                    else:
                                        setattr(self, f"_{attr}", val_i)
                            elif vtype == 9:  # ARRAY
                                atype = struct.unpack("<I", f.read(4))[0]
                                alen = struct.unpack("<Q", f.read(8))[0]
                                # Vocab size = token count; keep the length, not the strings.
                                if key == "tokenizer.ggml.tokens":
                                    self._vocab_size = int(alen)
                                val_a = self._gguf_read_array_value(f, atype, alen)
                                if key == "tokenizer.ggml.tokens" and val_a is not None:
                                    # Keep only delimiter-shaped entries, a few hundred of a six-figure vocab, not the whole vocabulary, enough to separate sentinels across model families.
                                    from core.inference.chat_template_helpers import (
                                        delimiter_shaped_tokens,
                                    )
                                    self._markup_tokens = delimiter_shaped_tokens(val_a)
                                attr = arch_keys.get(key)
                                if attr == "n_kv_heads" and val_a is not None:
                                    self._n_kv_heads_by_layer = [int(x) for x in val_a]
                                    if self._n_kv_heads is None and self._n_kv_heads_by_layer:
                                        self._n_kv_heads = max(self._n_kv_heads_by_layer)
                                elif attr == "sliding_window_pattern" and val_a is not None:
                                    self._sliding_window_pattern = [bool(x) for x in val_a]
                                    sliding_window_pattern_period = None
                                elif attr == "feed_forward_length" and val_a:
                                    self._feed_forward_length_by_layer = [int(x) for x in val_a]
                                    self._feed_forward_length = max(
                                        self._feed_forward_length_by_layer
                                    )
                                elif (
                                    attr in ("expert_used_count", "expert_feed_forward_length")
                                    and val_a
                                ):
                                    # Per-layer on Nemotron 3.5 Bolt; the widest layer sizes the graph.
                                    setattr(self, f"_{attr}", max(int(x) for x in val_a))
                            else:
                                self._gguf_skip_value(f, vtype)
                        else:
                            self._gguf_skip_value(f, vtype)
                    except (struct.error, UnicodeDecodeError, ValueError):
                        # Truncated input (e.g. HTTP byte-range header
                        # fetch), or a value type with no known size; break
                        # so the resolver fallback runs on what we have.
                        break
                else:
                    kv_complete = True

            # Bind buffered metadata after discovering the architecture namespace.
            if arch is not None:
                self._pooling_type = pooling_by_arch.get(f"{arch}.pooling_type", self._pooling_type)
                self._nextn_predict_layers = nextn_by_arch.get(
                    f"{arch}.nextn_predict_layers",
                    self._nextn_predict_layers,
                )

            # Decide diffusion routing before the SWA resolver: it can raise on an unknown arch and drop DiffusionGemma to plain llama-server.
            self._is_diffusion = bool(
                (arch and arch.lower().startswith("diffusion")) or canvas_seen
            )
            if self._is_diffusion:
                logger.info(
                    f"GGUF metadata: diffusion model detected (architecture={arch}); "
                    "will serve via the diffusion runner"
                )

            # Expand a scalar period straight from the GGUF first.
            if (
                self._sliding_window_pattern is None
                and sliding_window_pattern_period
                and self._n_layers
            ):
                self._sliding_window_pattern = [
                    (i + 1) % sliding_window_pattern_period != 0 for i in range(self._n_layers)
                ]

            # Diffusion models skip the resolver: no SWA pattern, and it can raise for them.
            if (
                self._sliding_window_pattern is None
                and self._sliding_window
                and self._n_layers
                and not self._is_diffusion
            ):
                hf_repo_candidates = (
                    general.get("general.source.huggingface.repository"),
                    _hf_repo_from_url(general.get("general.source.url")),
                    _hf_repo_from_url(general.get("general.source.repo_url")),
                    _hf_repo_from_url(general.get("general.base_model.0.repo_url")),
                    (
                        f"{general['general.base_model.0.organization']}/"
                        f"{general['general.base_model.0.name']}".replace(" ", "-")
                        if general.get("general.base_model.0.organization")
                        and general.get("general.base_model.0.name")
                        else None
                    ),
                    (
                        f"{general['general.organization']}/{general['general.basename']}".replace(
                            " ", "-"
                        )
                        if general.get("general.organization") and general.get("general.basename")
                        else None
                    ),
                )
                self._sliding_window_pattern = _resolve_swa_pattern(
                    arch,
                    self._n_layers,
                    hf_repo_candidates,
                )

            # Only a complete walk means _architecture None is 'declares none', not 'read stopped early'.
            self._gguf_header_parsed = kv_complete

            # Many call sites re-read the header per request; only log the first read of a file.
            first_read = _note_gguf_metadata_read(gguf_path)
            level = "info" if first_read else "debug"
            log_at = logger.info if first_read else logger.debug
            if self._context_length:
                log_at(f"GGUF metadata: context_length={self._context_length}")
            if self._chat_template:
                log_at(f"GGUF metadata: chat_template={len(self._chat_template)} chars")
                # Detect thinking/reasoning support from chat template.
                flags = detect_reasoning_flags(
                    self._chat_template,
                    self._model_identifier,
                    log_source = "GGUF metadata",
                    log_level = level,
                )
                self._supports_reasoning = flags["supports_reasoning"]
                self._reasoning_style = flags["reasoning_style"]
                self._reasoning_effort_levels = flags.get("reasoning_effort_levels", [])
                self._reasoning_always_on = flags["reasoning_always_on"]
                self._supports_preserve_thinking = flags["supports_preserve_thinking"]
                self._preserve_thinking_default = flags["preserve_thinking_default"]
                self._supports_tools = flags["supports_tools"]
        except Exception as e:
            logger.warning(f"Failed to read GGUF metadata: {e}")
