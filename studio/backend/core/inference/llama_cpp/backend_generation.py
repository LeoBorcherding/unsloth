# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: message building and generation."""

import contextlib
import httpx
import json
import re
import socket
import subprocess
import sys
import threading
import time

from typing import (
    Callable,
    Generator,
    Optional,
    Union,
)
from core.inference.context_window import (
    estimate_message_tokens_without_unpriced_media,
    messages_without_unpriced_media,
)
from core.inference.stream_errors import stream_error_from_chunk
from core.inference.tool_call_parser import (
    parse_tool_calls_from_text as _shared_parse_tool_calls_from_text,
)
from utils.first_token_timeout import (
    finish_first_token_wait,
    socket_timeout,
    start_first_token_wait,
)

from .compaction import (
    _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
    _DEFAULT_MAX_TOKENS_FLOOR,
    _archive_and_recall,
    _backend_supports_tools,
    _boundary_metadata,
    _can_reset_epoch,
    _compaction_fit_kwargs,
    _conversation_recall_reserve,
    _fit_with_instruction_pins,
    _keeps_compaction_boundary,
    _records_boundary,
    _retrieval_budget,
    _sticky_compaction_details,
)
from .core_types import (
    _DEVICE_LOST_LOG_GRACE_S,
    _LlamaStreamCancelled,
    _apply_seeded_llama_request,
    _is_gpu_device_lost,
)
from .launch_support import _local_ssl_context
from .log import logger
from .timings import (
    _backfill_usage_from_timings,
    _llama_chunk_has_generated_output,
    _perf_callback_wants_timings,
    _report_live_llama_timings,
)
from .tool_loop_text import (
    _DEFAULT_STREAM_STALL_TIMEOUT_S,
    _RESPAWN_REAP_GRACE_S,
    _finalize_reasoning_only_cumulative,
    _quote_cut_event,
)
from core.inference.llama_server_args import _flag_name
from core.inference.stream_errors import is_tool_parse_rejection
from dataclasses import replace
from .compaction import (
    _compaction_reporter,
)
from .core_types import (
    _TERMINAL_RETIREMENT_HOOKS,
    _automatic_recovery_load,
)
from .draft_args import (
    _extra_args_mtp_draft_path,
)
from .launch_flags import (
    _child_spec_env,
)
from .memory_policy import (
    _extra_args_set_spec_type,
)
from .spec_args import (
    _canonicalize_spec_mode,
)
from .tool_loop_text import (
    _GPU_OOM_HEADROOM_MAX_STRIKES,
    _GPU_OOM_HEADROOM_STEP_MIB,
    _with_template_extras,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _GenerationMixin:
    # ── Message building (OpenAI format) ──────────────────────────

    @staticmethod
    def _parse_tool_calls_from_text(
        content: str,
        *,
        allow_incomplete: bool = True,
        enabled_tool_names: Optional[set] = None,
    ) -> list[dict]:
        """Wrapper around the shared parser; ``enabled_tool_names`` gates the markerless bare-JSON form."""
        return _shared_parse_tool_calls_from_text(
            content,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )

    @staticmethod
    def _build_openai_messages(messages: list[dict], image_b64: Optional[str] = None) -> list[dict]:
        """Build OpenAI-format messages, optionally injecting an image_url part
        into the last user message for vision models. As-is if no image."""
        if not image_b64:
            return messages

        # Convert the last user message to multimodal content parts
        result = [msg.copy() for msg in messages]
        last_user_idx = None
        for i, msg in enumerate(result):
            if msg["role"] == "user":
                last_user_idx = i

        if last_user_idx is not None:
            text_content = result[last_user_idx].get("content", "")
            result[last_user_idx]["content"] = [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]

        return result

    # ── Generation (proxy to llama-server) ────────────────────────

    @contextlib.contextmanager
    def _open_stream(self, url: str, payload: dict, cancel_event):
        """Polls cancel_event between short reads (Windows will not wake a blocked recv); re-reads the live read timeout per call; httpcore snapshots it at body start"""
        from core.inference.mcp_images import prepare_image_turn_boundaries

        if "messages" in payload:
            payload = {
                **payload,
                "messages": prepare_image_turn_boundaries(
                    payload["messages"],
                    getattr(self, "_chat_template_override", None)
                    or getattr(self, "_chat_template", None),
                ),
            }
        from core.rag.conversation_archive import inference_running

        stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
        with (
            inference_running(),
            httpx.Client(
                timeout = stream_timeout,
                limits = httpx.Limits(max_keepalive_connections = 0),
                trust_env = False,
                verify = _local_ssl_context(),
            ) as client,
        ):
            first_token_deadline = start_first_token_wait()
            with self._stream_with_retry(
                client,
                url,
                payload,
                cancel_event,
                headers = self._auth_headers,
                first_token_deadline = first_token_deadline,
            ) as response:
                if response.status_code != 200:
                    error_body = response.read().decode()
                    raise RuntimeError(
                        f"llama-server returned {response.status_code}: {error_body}"
                    )
                yield response, first_token_deadline

    @staticmethod
    def _sse_event_payload(event: str) -> Optional[dict]:
        """Decode one complete SSE event's ``data:`` payload, or None."""
        payload_lines = [
            line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")
        ]
        if not payload_lines:
            return None
        try:
            data = json.loads("\n".join(payload_lines))
        except (TypeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _sse_event_prefill_progress(event: str) -> Optional[float]:
        """Return the reported processed prompt tokens, if present."""
        data = LlamaCppBackend._sse_event_payload(event)
        if data is None:
            return None
        progress = data.get("prompt_progress")
        if not isinstance(progress, dict):
            return None
        try:
            return float(progress.get("processed"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sse_event_has_generated_output(event: str) -> bool:
        """Return true when a complete SSE event carries model-generated output."""
        data = LlamaCppBackend._sse_event_payload(event)
        if data is None:
            return False
        return _llama_chunk_has_generated_output(data)

    def _iter_tool_round_text(
        self,
        response: "httpx.Response",
        payload: dict,
        cancel_event: Optional[threading.Event],
        first_token_deadline: Optional[float],
        on_respawn: Optional[Callable[[], None]] = None,
    ) -> Generator[str, None, None]:
        """Raw SSE text of one tool round. A tool-call parse rejection before any output (Llama 3.x
        ``{"type": "function", ...}``, ``;``-joined calls) is retried once with tool_choice "none",
        which turns llama-server's parser off so Studio's parser reads the raw reply."""
        stream = self._iter_text_cancellable(
            response, cancel_event, first_token_deadline = first_token_deadline
        )
        if not payload.get("tools") or payload.get("tool_choice") == "none":
            yield from stream
            return
        pending = ""
        for chunk in stream:
            pending += chunk
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                data = None
                stripped = line.strip()
                if stripped.startswith("data: ") and stripped != "data: [DONE]":
                    try:
                        data = json.loads(stripped[6:])
                    except json.JSONDecodeError:
                        pass
                if is_tool_parse_rejection(data):
                    logger.info(
                        "llama-server could not parse the reply as tool calls; "
                        "asking again with its tool-call parser off"
                    )
                    response.close()
                    payload["tool_choice"] = "none"
                    with self._open_chat_stream_with_respawn_retry(
                        payload, cancel_event, on_respawn = on_respawn
                    ) as (retry_response, retry_deadline):
                        yield from self._iter_text_cancellable(
                            retry_response, cancel_event, first_token_deadline = retry_deadline
                        )
                    return
                yield line + "\n"
                # Output already reached the caller: a second request could not be joined to it.
                if isinstance(data, dict) and _llama_chunk_has_generated_output(data):
                    yield pending
                    yield from stream
                    return
        yield pending

    @staticmethod
    def _iter_text_cancellable(
        response: "httpx.Response",
        cancel_event: Optional[threading.Event] = None,
        stall_timeout_s: float = _DEFAULT_STREAM_STALL_TIMEOUT_S,
        first_token_deadline: Optional[float] = None,
        post_first_chunk_read_timeout_s: Optional[float] = _DEFAULT_STREAM_STALL_TIMEOUT_S,
    ) -> Generator[str, None, None]:
        """Iterate a stream while polling cancel and stall timeouts."""
        text_iter = response.iter_text()
        if first_token_deadline is None:
            first_token_deadline = start_first_token_wait()
        last_chunk_at: Optional[float] = None
        last_prefill_progress: Optional[float] = None
        prefill_sse_buffer = ""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                response.close()
                return
            try:
                if last_chunk_at is None:
                    remaining_s = first_token_deadline - time.monotonic()
                    if remaining_s <= 0:
                        raise httpx.ReadTimeout("The model did not produce a first token in time.")
                    LlamaCppBackend._set_stream_read_timeout(response, socket_timeout(remaining_s))
                chunk = next(text_iter)
                starts_output = last_chunk_at is not None
                if chunk and last_chunk_at is None:
                    prefill_sse_buffer += chunk
                    while match := re.search(r"(?:\r\n|\r|\n){2}", prefill_sse_buffer):
                        event = prefill_sse_buffer[: match.start()]
                        prefill_sse_buffer = prefill_sse_buffer[match.end() :]
                        if LlamaCppBackend._sse_event_has_generated_output(event):
                            starts_output = True
                            prefill_sse_buffer = ""
                            break
                        # Renew only on increasing progress so repeated events still time out.
                        processed = LlamaCppBackend._sse_event_prefill_progress(event)
                        if processed is not None and (
                            last_prefill_progress is None or processed > last_prefill_progress
                        ):
                            last_prefill_progress = processed
                            first_token_deadline = start_first_token_wait()
                if chunk and starts_output:
                    finish_first_token_wait()
                    if last_chunk_at is None and post_first_chunk_read_timeout_s is not None:
                        LlamaCppBackend._set_stream_read_timeout(
                            response,
                            post_first_chunk_read_timeout_s,
                        )
                    last_chunk_at = time.monotonic()
                yield chunk
            except StopIteration:
                return
            except httpx.ReadTimeout:
                now = time.monotonic()
                if last_chunk_at is None:
                    if now >= first_token_deadline:
                        raise
                elif now - last_chunk_at >= stall_timeout_s:
                    raise httpx.ReadTimeout("The model stopped producing tokens mid-response.")
                continue

    @staticmethod
    def _set_stream_read_timeout(
        response: "httpx.Response", read_timeout_s: Optional[float]
    ) -> None:
        """Lower only post-header stream reads; keep prefill timeout long."""
        try:
            timeout_ext = response.request.extensions.get("timeout")
            if isinstance(timeout_ext, dict):
                timeout_ext["read"] = read_timeout_s
        except Exception:
            logger.debug("Could not lower response read timeout", exc_info = True)

    @staticmethod
    def _shutdown_active_httpx_sockets(client: "httpx.Client") -> None:
        """Best-effort interrupt for a sync httpx read blocked in recv(), whether
        parked before headers (prefill) or mid-stream."""
        try:
            pool = getattr(getattr(client, "_transport", None), "_pool", None)
            connections = list(getattr(pool, "_connections", []) or [])
            for connection in connections:
                inner = getattr(connection, "_connection", None)
                stream = getattr(inner, "_network_stream", None)
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        except Exception:
            logger.debug("Could not shutdown active httpx socket", exc_info = True)
        try:
            client.close()
        except Exception:
            logger.debug("Could not close httpx client", exc_info = True)

    @staticmethod
    def _install_cancel_aware_read(
        client: "httpx.Client",
        cancel_event: threading.Event,
        response: Optional["httpx.Response"] = None,
        poll_s: float = 0.2,
    ) -> None:
        """Polls cancel_event between short reads, since Windows does not wake a blocked recv() on
        shutdown."""
        import httpcore

        def _live_read_timeout(snapshot: Optional[float]) -> Optional[float]:
            if response is None:
                return snapshot
            try:
                ext = response.request.extensions.get("timeout")
                if isinstance(ext, dict) and "read" in ext:
                    value = ext["read"]
                    # None is a live value too: an unlimited first-token wait has no read bound.
                    if value is None or isinstance(value, (int, float)):
                        return value
            except Exception:
                pass
            return snapshot

        try:
            pool = getattr(getattr(client, "_transport", None), "_pool", None)
            for connection in list(getattr(pool, "_connections", []) or []):
                inner = getattr(connection, "_connection", None)
                stream = getattr(inner, "_network_stream", None)
                if stream is None or getattr(stream, "_unsloth_cancel_wrapped", False):
                    continue
                orig_read = stream.read

                def read(
                    max_bytes,
                    timeout = None,
                    _orig = orig_read,
                ):
                    effective = _live_read_timeout(timeout)
                    deadline = None if effective is None else time.monotonic() + effective
                    while True:
                        if cancel_event.is_set():
                            raise httpcore.ReadError("stream cancelled by user")
                        if deadline is None:
                            step = poll_s
                        else:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise httpcore.ReadTimeout("read operation timed out")
                            step = min(poll_s, remaining)
                        try:
                            return _orig(max_bytes, timeout = step)
                        except httpcore.ReadTimeout:
                            if deadline is not None and time.monotonic() >= deadline:
                                raise
                            continue  # slow but alive: keep reading

                stream.read = read
                stream._unsloth_cancel_wrapped = True
        except Exception:
            logger.debug("Could not install cancel-aware read", exc_info = True)

    @staticmethod
    @contextlib.contextmanager
    def _stream_with_retry(
        client: "httpx.Client",
        url: str,
        payload: dict,
        cancel_event: Optional[threading.Event] = None,
        headers: Optional[dict] = None,
        first_token_deadline: Optional[float] = None,
    ):
        """Open one streaming POST and let cancel interrupt prefill or reads."""
        if cancel_event is not None and cancel_event.is_set():
            raise _LlamaStreamCancelled

        _cancel_closed = threading.Event()
        _response_ref: list = [None]

        def _cancel_watcher():
            while not _cancel_closed.is_set():
                if cancel_event.wait(timeout = 0.3):
                    while not _cancel_closed.is_set():
                        r = _response_ref[0]
                        try:
                            # close() can't wake a blocked recv(); shut the socket down first.
                            LlamaCppBackend._shutdown_active_httpx_sockets(client)
                            if r is not None:
                                r.close()
                            return
                        except Exception as e:
                            logger.debug(f"Error closing request in cancel watcher: {e}")
                        _cancel_closed.wait(timeout = 0.1)
                    return

        watcher = None
        if cancel_event is not None:
            watcher = threading.Thread(target = _cancel_watcher, daemon = True, name = "prefill-cancel")
            watcher.start()

        try:
            if first_token_deadline is None:
                first_token_deadline = start_first_token_wait()
            prefill_read_timeout = max(0.1, first_token_deadline - time.monotonic())
            if sys.platform == "win32":
                # Stop cannot wake a Winsock recv parked before headers, so that wait keeps the default bound.
                prefill_read_timeout = min(prefill_read_timeout, _DEFAULT_FIRST_TOKEN_TIMEOUT_S)
            prefill_timeout = httpx.Timeout(
                connect = 30,
                read = socket_timeout(prefill_read_timeout),
                write = 10,
                pool = 10,
            )
            with client.stream(
                "POST",
                url,
                json = payload,
                timeout = prefill_timeout,
                headers = headers,
            ) as response:
                _response_ref[0] = response
                # Portable mid-stream cancel: the reader polls cancel itself, so Stop
                # interrupts a stalled read where the watcher's Windows socket shutdown
                # does not. Installed without a cancel event too: only this wrapper
                # honors the live stall timeout over the one latched for the header wait.
                LlamaCppBackend._install_cancel_aware_read(
                    client, cancel_event or threading.Event(), response
                )
                if cancel_event is not None and cancel_event.is_set():
                    raise _LlamaStreamCancelled
                yield response
                return
        except (httpx.RequestError, RuntimeError):
            # Response was closed by the cancel watcher
            if cancel_event is not None and cancel_event.is_set():
                raise _LlamaStreamCancelled
            raise
        finally:
            _cancel_closed.set()

    def _server_socket_is_open(self, timeout_s: float = 0.15) -> bool:
        """True while the port still accepts; the socket dies with the process, so no wait for reaping."""
        port = self._port
        if not port:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = timeout_s):
                return True
        except OSError:
            return False

    def _respawn_if_dead(self) -> bool:
        """Replays load_model when the process died, since is_loaded still reads True; one reload at
        a time."""
        # Read outside the lock so queued callers skip the grace wait for a replaced child.
        served_by = self._process
        with self._respawn_lock:
            proc = self._process
            if proc is None:
                return False
            if self._cancel_event.is_set():
                # unload_model sets this before it kills; don't aim the retry at it.
                return False
            if proc is not served_by:
                # Replaced while we queued: this child never served our request.
                return self._healthy
            if proc.poll() is None:
                # Still serving: transient error, skip the grace wait.
                if self._server_socket_is_open():
                    return self._healthy
                # A closing server can beat its own exit status; wait briefly for it.
                deadline = time.monotonic() + _RESPAWN_REAP_GRACE_S
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
            if proc.poll() is None:
                # Alive: either a concurrent caller already respawned it (healthy), or
                # this connection error wasn't a dead server.
                return self._healthy
            with self._mtp_runtime_fallback_lock:
                if self._mtp_runtime_fallback_in_progress:
                    # An MTP-free reload owns this; replaying old kwargs would abort it.
                    logger.info("Respawn skipped: an MTP-free reload is already recovering.")
                    return False
            # The RLock lets the load_model below re-enter it.
            with self._serial_load_lock:
                if self._process is not proc:
                    logger.info("Respawn skipped: a newer load is already active.")
                    return self._healthy
                # Snapshot under _lock: unload holds it, never _serial_load_lock.
                with self._lock:
                    if self._cancel_event.is_set():
                        logger.info("Respawn skipped: the model was unloaded.")
                        return False
                    intent = self._last_load_intent
                    if intent is None:
                        return False
                    if self._metal_oom_is_terminal(intent):
                        logger.error(
                            f"Not restarting llama-server for '{self._model_identifier}': "
                            "it ran out of Metal memory twice with these settings."
                        )
                        return False
                    epoch = self._unload_epoch
                    self._healthy = False
                logger.warning(
                    f"llama-server for '{self._model_identifier}' exited "
                    f"(code {proc.returncode}); respawning to recover the session"
                )
                if self._died_of_gpu_oom(proc):
                    self._note_gpu_oom_strike(intent)
                try:
                    with _automatic_recovery_load():
                        started = bool(self.load_model(intent))
                except Exception as exc:
                    logger.error(f"Failed to respawn llama-server: {exc}")
                    return False
                if started and self._unload_epoch != epoch:
                    # load_model cleared _cancel_event, so the epoch is the only unload evidence.
                    logger.info("Respawn undone: the model was unloaded during the reload.")
                    self.unload_model()
                    return False
                return started

    def _stop_stats_logger(self) -> None:
        """Stop the engine-stats poller of the child going away. getattr: teardown must
        tolerate a partially-built backend (failed __init__ or a __new__-built instance)."""
        if getattr(self, "_stats_logger", None) is not None:
            self._stats_logger.stop()
            self._stats_logger = None

    def _retire_device_lost_server(
        self,
        served_by,
        metal_poison: Optional[dict] = None,
    ) -> None:
        """Kill the llama-server that hit a lost GPU device, if it is still current (#11453).

        It survives VK_ERROR_DEVICE_LOST but fails every later request; a dead child is
        what _respawn_if_dead already recovers. ``metal_poison`` marks the Metal variant:
        a failed command buffer (usually out of memory) latches the backend the same way.
        """
        if metal_poison is not None:
            # Outside the locks: the probe waits on the child up to its timeout.
            if (
                served_by is None
                or served_by is not self._process
                or served_by.poll() is not None
                or self._cancel_event.is_set()
                or not self._metal_compute_failure_confirmed()
            ):
                metal_poison["pending"] = False
                return
        # _serial_load_lock too: no load may publish another model between the kill and the strike.
        with self._respawn_lock, self._serial_load_lock:
            if (
                served_by is None
                or served_by is not self._process
                or served_by.poll() is not None
                or self._cancel_event.is_set()
            ):
                return
            # First, or the MTP watchdog reads this kill as an MTP crash.
            self._stop_mtp_crash_watchdog()
            self._retired_process = served_by
            try:
                served_by.kill()
                served_by.wait(timeout = 10)
            except (OSError, subprocess.TimeoutExpired):
                pass
            # Else it polls the dead child's port, which another server may reuse.
            self._stop_stats_logger()
            if metal_poison is not None:
                # Under _respawn_lock, or a request could respawn the speculative config.
                terminal = False
                with self._lock:
                    if self._process is served_by and not self._cancel_event.is_set():
                        terminal = self._after_metal_poisoned_server(
                            bool(metal_poison.get("draft"))
                        )
                if not terminal:
                    return
                # Nothing will respawn it: unload it like any other model, still under
                # _serial_load_lock.
                self.unload_model()
                # A reload queued on _serial_load_lock may land before the hooks run.
                retired_epoch = self._unload_epoch
        if metal_poison is not None:
            # Outside every backend lock (gpu_arbiter._lock comes first).
            for hook in list(_TERMINAL_RETIREMENT_HOOKS):
                try:
                    hook(self, retired_epoch)
                except Exception as exc:  # noqa: BLE001 -- best effort, as a reap
                    logger.debug("Terminal retirement cleanup failed: %s", exc)
            return
        try:
            logger.warning(
                f"llama-server for '{self._model_identifier}' lost its GPU device "
                "(the graphics driver reset it); stopped it so the next request starts "
                "a fresh one. On Intel Arc with Vulkan, GGML_VK_DISABLE_COOPMAT=1 may "
                "avoid the reset."
            )
        except Exception:
            # The logger raises once stdout is closed; recovery must not.
            pass

    def _after_metal_poisoned_server(self, draft_failed: bool) -> bool:
        """Choose the recovery after a Metal-poisoned server was retired; True on the terminal
        second strike."""
        intent = self._last_load_intent
        key = self._metal_oom_key(intent) if intent is not None else None
        prior = getattr(self, "_metal_oom_strikes", None)
        strikes = prior[1] + 1 if prior is not None and key is not None and prior[0] == key else 1
        self._metal_oom_strikes = (key, strikes)
        where = "the speculative draft context" if draft_failed else "a decode"
        try:
            name = self._model_identifier
            if strikes >= 2:
                # Not replaced: an identical explicit load must spawn a new child.
                self._healthy = False
                self._metal_oom_terminal_key = key
                logger.error(
                    f"llama-server for '{name}' ran out of Metal memory again in {where} "
                    "after a restart; not restarting it. Lower the context length or the "
                    "GPU layers, turn off speculative decoding, or choose a smaller quant, "
                    "then load the model again."
                )
                return True
            requested_mode = getattr(intent, "speculative_type", None)
            recovery_extras = list(getattr(intent, "extra_args", None) or ())
            # A bare --model-draft in the extras survives an Off reload (has_dft()).
            auto_sidecar = (
                self._speculative_type in ("draft-dspark", "draft-dflash")
                and _canonicalize_spec_mode(requested_mode) in (None, "auto")
                and not _extra_args_set_spec_type(recovery_extras)
                and not _extra_args_mtp_draft_path(
                    recovery_extras, env = _child_spec_env(recovery_extras)
                )
            )
            if auto_sidecar and self._maybe_recover_from_mtp_crash(sidecar_oom = True):
                logger.warning(
                    f"llama-server for '{name}' ran out of Metal memory in {where} with the "
                    f"{self._speculative_type.split('-', 1)[1]} drafter and its Metal backend "
                    "cannot recover; reloading once without speculative decoding."
                )
                return False
            logger.warning(
                f"llama-server for '{name}' ran out of Metal memory in {where} and its Metal "
                "backend cannot recover; stopped it so the next request starts a fresh one."
            )
        except Exception:
            pass
        return strikes >= 2

    @contextlib.contextmanager
    def _while_still_retired(self, epoch):
        """Hold _serial_load_lock, which every load takes, and yield whether this backend is
        still the one a terminal Metal strike unloaded at ``epoch``: nothing spawned since and
        no other unload. A cleanup acting on a replacement would kill it."""
        with self._serial_load_lock:
            yield self._process is None and getattr(self, "_unload_epoch", None) == epoch

    def _metal_oom_is_terminal(self, intent) -> bool:
        """Whether ``intent`` already ran out of Metal memory twice (no more replays)."""
        terminal = getattr(self, "_metal_oom_terminal_key", None)
        return (
            terminal is not None and intent is not None and self._metal_oom_key(intent) == terminal
        )

    def _gpu_oom_headroom_mib(self, intent) -> float:
        """Extra --fit-target for a reload of the load that ran out of GPU memory; 0 otherwise."""
        prior = getattr(self, "_gpu_oom_strikes", None)
        if prior is None or intent is None or prior[0] != self._metal_oom_key(intent):
            return 0.0
        return prior[1] * _GPU_OOM_HEADROOM_STEP_MIB

    def _note_gpu_oom_strike(self, intent) -> None:
        key = self._metal_oom_key(intent)
        prior = getattr(self, "_gpu_oom_strikes", None)
        strikes = prior[1] + 1 if prior is not None and prior[0] == key else 1
        strikes = min(strikes, _GPU_OOM_HEADROOM_MAX_STRIKES)
        self._gpu_oom_strikes = (key, strikes)
        try:
            logger.warning(
                f"llama-server for '{self._model_identifier}' ran out of GPU memory; "
                f"respawning it with {int(strikes * _GPU_OOM_HEADROOM_STEP_MIB)} MiB more "
                "free VRAM kept (--fit-target)."
            )
        except Exception:
            pass

    def _died_of_gpu_oom(self, proc) -> bool:
        """Whether the dead served child logged a GPU out-of-memory error before exiting."""
        drain = getattr(self, "_stdout_thread", None)
        if drain is not None and drain is not threading.current_thread():
            # Its last lines can still be in the pipe when the exit status lands.
            drain.join(timeout = 2)
        return proc is not None and getattr(self, "_gpu_oom_process", None) is proc

    @staticmethod
    def _metal_oom_key(intent):
        """The load an OOM strike is charged to: the speculative mode is what the fallback
        changes, so it is left out and the no-drafter reload keeps the strike count. The
        verified-cache hint is per request (the committed snapshot drops it), so it is left
        out too, or a hinted explicit reload and the snapshot it commits disagree."""
        try:
            return replace(intent, speculative_type = None, verified_gguf = None)
        except TypeError:
            return intent

    def _metal_compute_failure_confirmed(self) -> bool:
        """Whether the child really fails compute: a poisoned Metal backend rejects every
        graph, so a one-token request comes back 500 "Compute error.". The log marker
        alone is not proof, since request text can reach the server's error output. An
        --embedding server is restricted to embeddings, so it is probed on the native
        /embedding route, which computes under every pooling type but RANK (/v1/embeddings
        400s on NONE pooling before compute). RANK pools n_cls_out floats while
        send_embedding reads n_embd_out of them, so a reranker (--rerank or --pooling rank)
        is probed on /rerank, and one ranking only by the GGUF's default (/rerank refuses
        it) is not probed at all."""
        try:
            embedding = bool(self.is_embedding_gguf)
        except Exception:
            embedding = False
        args = [str(a) for a in (self._extra_args or ())]
        # Argv only (load_model strips inherited LLAMA_ARG_POOLING); last-wins, as common/arg.cpp
        # applies them.
        pooling = None
        for i, raw in enumerate(args):
            flag = _flag_name(raw)
            if flag in ("--embedding", "--embeddings"):
                embedding = True
            elif flag in ("--rerank", "--reranking"):
                embedding = True
                pooling = "rank"
            elif flag == "--pooling":
                _, eq, inline = raw.partition("=")
                value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
                pooling = value.strip().lower() or pooling
        if embedding and pooling == "rank":
            route, body = "/rerank", {"query": " ", "documents": [" "]}
        elif embedding and pooling is None and getattr(self, "_pooling_type", None) == 4:
            return False
        elif embedding:
            route, body = "/embedding", {"content": " "}
        else:
            route, body = "/completion", {"prompt": " ", "n_predict": 1, "stream": False}
        try:
            resp = httpx.post(
                f"{self.base_url}{route}",
                json = body,
                headers = self._auth_headers,
                timeout = 10.0,
                trust_env = False,
                verify = _local_ssl_context(),
            )
            if resp.status_code < 500:
                return False
            error = resp.json().get("error", {})
            message = error.get("message", "") if isinstance(error, dict) else error
            return (
                str(message)
                .rstrip()
                .endswith(("Compute error.", "failed to process speculative batch"))
            )
        except Exception:
            return False

    def _retire_device_lost_server_soon(
        self,
        served_by,
        metal_poison: Optional[dict] = None,
    ) -> None:
        """Retire after a device loss in the server log, for the passthrough routes.

        The grace lets a chat request that hit the same error retire and retry first.
        """

        def _retire():
            time.sleep(_DEVICE_LOST_LOG_GRACE_S)
            self._retire_device_lost_server(served_by, metal_poison = metal_poison)

        try:
            threading.Thread(target = _retire, daemon = True, name = "llama-device-lost").start()
        except RuntimeError:
            if metal_poison is not None:
                # No retirement will run: let a later poison line try again.
                metal_poison["pending"] = False

    @contextlib.contextmanager
    def _open_chat_stream_with_respawn_retry(
        self,
        payload: dict,
        cancel_event,
        on_respawn: Optional[Callable[[], None]] = None,
    ):
        """Respawns once; retries only before the response opens (replay could duplicate output); timeouts never retried; base_url resolved per attempt."""
        for attempt in range(2):
            response_opened = False
            served_by = getattr(self, "_process", None)
            try:
                url = f"{self.base_url}/v1/chat/completions"
                with self._open_stream(url, payload, cancel_event) as opened:
                    response_opened = True
                    yield opened
                    return
            except RuntimeError as exc:
                if not _is_gpu_device_lost(exc):
                    raise
                # Retry only before the 200; once opened the consumer may have emitted output.
                self._retire_device_lost_server(served_by)
                if response_opened or attempt > 0 or not self._respawn_if_dead():
                    raise
                logger.warning(
                    "llama-server lost its GPU device; restarted it and retrying the generation"
                )
                if on_respawn is not None:
                    on_respawn()
                continue
            except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if response_opened:
                    raise
                if self._maybe_recover_from_mtp_crash(exc):
                    raise RuntimeError("Lost connection to llama-server") from exc
                if attempt == 0 and self._respawn_if_dead():
                    logger.warning(
                        "llama-server was unreachable; respawned it and retrying the generation"
                    )
                    if on_respawn is not None:
                        on_respawn()
                    continue
                raise

    @staticmethod
    def decode_slot_from_chunk(chunk) -> Optional[int]:
        """Slot a stream chunk was decoded on, from __verbose on the final result; None if absent."""
        if not isinstance(chunk, dict):
            return None
        verbose = chunk.get("__verbose")
        if not isinstance(verbose, dict):
            return None
        slot = verbose.get("id_slot")
        if type(slot) is not int or slot < 0:
            return None
        return slot

    def release_idle_chat_slot(self, base_url: str, slot: int) -> bool:
        """Erases the cached slot of a round only on the decoding server (a reload renumbers slots) and needs --slot-save-path; a timeout can cost the chat its prefix."""
        if (
            not self._slot_save_dir
            or base_url != self.base_url
            or type(slot) is not int
            or slot < 0
        ):
            return False
        try:
            response = httpx.post(
                f"{base_url}/slots/{slot}",
                params = {"action": "erase"},
                headers = self._auth_headers,
                timeout = 2.0,
                trust_env = False,
                verify = _local_ssl_context(),
            )
            response.raise_for_status()
            result = response.json()
            return (
                result.get("id_slot") == slot
                and type(result.get("n_erased")) is int
                and result["n_erased"] >= 0
            )
        except Exception:
            logger.debug("Approval cache reclamation failed; retaining reservation", exc_info = True)
            return False

    def generate_chat_completion(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        replay_preserve_thinking: bool = True,
        continue_final_message: bool = False,
        seed: Optional[int] = None,
        promote_reasoning_only: bool = True,
        perf_callback: Optional[Callable[[dict], None]] = None,
        reasoning_provenance: Optional[dict] = None,
        context_overflow: Optional[str] = None,
        context_policy: Optional[str] = None,
        compaction_headroom_ratio: Optional[float] = None,
        thread_id: Optional[str] = None,
        tools_withheld: bool = False,
        thinking_budget_tokens: Optional[int] = None,
        _allow_respawn_retry: bool = True,
        compact_context: bool = False,
        chat_template_kwargs: Optional[dict] = None,
    ) -> Generator[Union[str, dict], None, None]:
        """Streams a chat completion via /v1/chat/completions (llama-server applies template and vision); yields cumulative text like InferenceBackend."""
        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")
        if replay_preserve_thinking:
            preserve_thinking = self._replay_preserve_thinking(
                preserve_thinking, messages, enable_thinking, reasoning_effort
            )

        from core.inference.chat_template_helpers import (
            neutralize_control_markup_in_messages,
            trailing_assistant_resume_kind,
        )

        openai_messages = self._build_openai_messages(messages, image_b64)
        _resume_kind = (
            trailing_assistant_resume_kind(openai_messages) if continue_final_message else None
        )
        continue_final_message = _resume_kind is not None
        resumed_reasoning = (
            openai_messages[-1]["reasoning_content"] if _resume_kind == "reasoning_content" else ""
        )

        payload = {
            # llama-server applies the chat template itself (#7066).
            "messages": neutralize_control_markup_in_messages(
                openai_messages, None, self.markup_profile
            ),
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        retry_messages = messages
        retry_image_b64 = image_b64
        retry_max_tokens = max_tokens
        retry_context_overflow = context_overflow
        retry_preflight_context_length = None
        # Progress events let advancing prefills renew the first-token deadline.
        payload["return_progress"] = True
        if _perf_callback_wants_timings(perf_callback):
            payload["timings_per_token"] = True
        if logit_bias:
            payload["logit_bias"] = logit_bias
        # Per-request enable_thinking / reasoning_effort / preserve_thinking, plus the caller's other tags
        _reasoning_kw = _with_template_extras(
            self._request_reasoning_kwargs(enable_thinking, reasoning_effort, preserve_thinking),
            chat_template_kwargs,
        )
        if _reasoning_kw is not None:
            payload["chat_template_kwargs"] = _reasoning_kw
        if thinking_budget_tokens is not None:
            payload["thinking_budget_tokens"] = thinking_budget_tokens
        if continue_final_message:
            # llama-server applies the template; it rejects both flags set true.
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        # Default cap to the model context when known.
        payload["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        if context_overflow == "truncate_oldest" and self._effective_context_length:
            try:
                _before_fit = openai_messages
                # Same value for boundary read and fit, so they agree on epochs.
                _can_reset = _can_reset_epoch(
                    thread_id,
                    _backend_supports_tools(self),
                    tools_withheld = tools_withheld,
                )
                _sticky, _sticky_is_checkpoint, _sticky_manual = _sticky_compaction_details(
                    thread_id,
                    _before_fit,
                    context_policy = context_policy,
                    can_reset = _can_reset,
                    compaction_headroom_ratio = compaction_headroom_ratio,
                )
                openai_messages, truncation = _fit_with_instruction_pins(
                    openai_messages,
                    on_compacting = _compaction_reporter(perf_callback),
                    context_length = self._effective_context_length,
                    max_tokens = payload["max_tokens"],
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            self._countable_messages(fitted), None, self.markup_profile
                        ),
                        None,
                        None,
                        strict = True,
                        chat_template_kwargs = _reasoning_kw,
                        continue_final_message = continue_final_message,
                        should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                    ),
                    estimate_message = self._estimate_countable_message,
                    reserve_tokens = _conversation_recall_reserve(thread_id),
                    sticky_dropped = _sticky,
                    sticky_is_checkpoint = _sticky_is_checkpoint,
                    sticky_manual = _sticky_manual,
                    force_reset = compact_context,
                    keeps_boundary = _keeps_compaction_boundary(thread_id),
                    can_reset = _can_reset,
                    recall_offered = False,
                    **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                )
                if truncation:
                    # Inline: no tools array here, and strict templates reject a lone tool role.
                    _recalled = _archive_and_recall(
                        openai_messages,
                        _before_fit,
                        branch_messages = _before_fit,
                        thread_id = thread_id,
                        style = "inline",
                        # Checkpoint mode recalls once; rolling recalls every turn that evicted.
                        force_recall = bool(truncation.get("checkpoint_started", True)),
                        # A rescued refusal may evict messages; archive them without recall.
                        recall_done = not truncation["fits"],
                        # This single-shot answer never returns to the prompt.
                        recall_budget_tokens = _retrieval_budget(
                            self._effective_context_length,
                            payload["max_tokens"],
                            truncation.get("prompt_tokens_after") or 0,
                        ),
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                self._countable_messages(fitted),
                                None,
                                self.markup_profile,
                            ),
                            None,
                            None,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                            continue_final_message = continue_final_message,
                        ),
                        cancel_check = lambda: bool(cancel_event and cancel_event.is_set()),
                        background = True,
                    )
                    openai_messages = _recalled["conversation"]
                    truncation = {**truncation, **_recalled["counts"]}
                if truncation and _records_boundary(truncation):
                    truncation = {
                        **truncation,
                        **_boundary_metadata(
                            openai_messages, _before_fit, compaction_headroom_ratio
                        ),
                    }
                payload["messages"] = neutralize_control_markup_in_messages(
                    openai_messages, None, self.markup_profile
                )
                # Reuse the fitted request; re-running preflight would duplicate the event.
                retry_messages = openai_messages
                retry_image_b64 = None
                retry_max_tokens = payload["max_tokens"]
                retry_context_overflow = None
                retry_preflight_context_length = self._effective_context_length
                # Re-cost here, not at the breaks: results and nudges appended after the last re-cost left the pool unaware of this final pass, the largest request.
                if truncation:
                    yield {"type": "context_truncated", **truncation}
            except Exception as exc:
                logger.warning("Could not preflight the rolling context window: %s", exc)
        if stop:
            payload["stop"] = stop
        _apply_seeded_llama_request(payload, seed)
        payload["stream_options"] = {"include_usage": True}

        served_by = getattr(self, "_process", None)
        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""
        in_thinking = False
        _stream_done = False
        _metadata_usage = None
        _metadata_timings = None
        _metadata_finish_reason = None

        def _replay_on_replacement_server():
            context_overflow_ = retry_context_overflow
            max_tokens_ = retry_max_tokens
            if (
                retry_preflight_context_length is not None
                and retry_preflight_context_length != self._effective_context_length
            ):
                # Refit the compacted prompt against the replacement server's window;
                # any event now reports only additional evictions.
                context_overflow_ = context_overflow
                if max_tokens is None:
                    max_tokens_ = None
            yield from self.generate_chat_completion(
                retry_messages,
                image_b64 = retry_image_b64,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_tokens = max_tokens_,
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                stop = stop,
                cancel_event = cancel_event,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                seed = seed,
                promote_reasoning_only = promote_reasoning_only,
                perf_callback = perf_callback,
                reasoning_provenance = reasoning_provenance,
                context_overflow = context_overflow_,
                context_policy = context_policy,
                compaction_headroom_ratio = compaction_headroom_ratio,
                # The retry refits and can evict more turns, so it must get the same thread_id to
                # archive them.
                thread_id = thread_id,
                # The retry refits, so it must be told the same about this request's
                # tools as the first attempt was.
                tools_withheld = tools_withheld,
                thinking_budget_tokens = thinking_budget_tokens,
                compact_context = compact_context,
                chat_template_kwargs = chat_template_kwargs,
                _allow_respawn_retry = False,
            )

        try:
            with self._open_stream(url, payload, cancel_event) as (
                response,
                first_token_deadline,
            ):
                buffer = ""
                has_content_tokens = False
                # Track content separately: a literal `</think>` can appear in the answer.
                answer_text = ""
                reasoning_text = ""
                _prov_entry = None
                for raw_chunk in self._iter_text_cancellable(
                    response,
                    cancel_event,
                    first_token_deadline = first_token_deadline,
                ):
                    buffer += raw_chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue
                        if line == "data: [DONE]":
                            if in_thinking:
                                if has_content_tokens:
                                    # Real thinking + content: close the tag
                                    cumulative += "</think>"
                                    _prov_entry = None
                                    yield cumulative
                                else:
                                    # Reasoning-only reply (e.g. Qwen3 always-think): show it as the
                                    # main response.
                                    cumulative = _finalize_reasoning_only_cumulative(
                                        cumulative,
                                        reasoning_text,
                                        _metadata_finish_reason,
                                        promote_reasoning_only,
                                        resumed_reasoning,
                                    )
                                    _prov_entry = None
                                    yield cumulative
                            _stream_done = True
                            break  # exit inner while
                        if not line.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line[6:])
                            _report_live_llama_timings(perf_callback, data)
                            # Diffusion frame from the shim: forward untouched, never into content.
                            if data.get("type") == "diffusion_frame":
                                yield data
                                continue
                            # Capture server timings/usage from final chunks.
                            _chunk_timings = data.get("timings")
                            if _chunk_timings:
                                _metadata_timings = _chunk_timings
                            _chunk_usage = data.get("usage")
                            if _chunk_usage:
                                _metadata_usage = _chunk_usage
                            # Error chunks have no choices; raise so the failure is surfaced.
                            _stream_error = stream_error_from_chunk(data)
                            if _stream_error is not None:
                                raise _stream_error
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                _fr = choices[0].get("finish_reason")
                                if _fr:
                                    _metadata_finish_reason = _fr

                                # Reasoning/thinking tokens: llama-server
                                # sends these as "reasoning_content"; wrap
                                # in <think> tags for the frontend parser.
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    reasoning_text += reasoning
                                    if not in_thinking:
                                        # A leading <think> opened here is genuine reasoning, unlike
                                        # literal tags.
                                        if reasoning_provenance is not None and not cumulative:
                                            reasoning_provenance["wrapped"] = (
                                                reasoning_provenance.get("wrapped", 0) + 1
                                            )
                                            _prov_entry = {"len": 0}
                                            reasoning_provenance.setdefault("wraps", []).append(
                                                _prov_entry
                                            )
                                        cumulative += "<think>"
                                        in_thinking = True
                                    cumulative += reasoning
                                    if _prov_entry is not None:
                                        _prov_entry["len"] += len(reasoning)
                                    yield cumulative

                                token = delta.get("content", "")
                                if token:
                                    has_content_tokens = True
                                    answer_text += token
                                    if in_thinking:
                                        cumulative += "</think>"
                                        in_thinking = False
                                        _prov_entry = None
                                    cumulative += token
                                    yield cumulative
                        except json.JSONDecodeError:
                            logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                    if _stream_done:
                        break  # exit outer for
                _cut = _quote_cut_event(
                    reasoning_text,
                    answer_text,
                    _metadata_finish_reason,
                    promote_reasoning_only,
                )
                if _cut is not None:
                    yield _cut
                if _metadata_usage or _metadata_timings or _metadata_finish_reason:
                    _metadata_usage = _backfill_usage_from_timings(
                        _metadata_usage, _metadata_timings
                    )
                    yield {
                        "type": "metadata",
                        # Never None: consumers call usage.get(...).
                        "usage": _metadata_usage or {},
                        "timings": _metadata_timings,
                        "finish_reason": _metadata_finish_reason,
                    }

        except _LlamaStreamCancelled:
            return
        except httpx.ConnectError as e:
            # Server already down. If this was an MTP crash, recover by
            # reloading without MTP (scheduled in the background) and fail this
            # request. Otherwise the server was likely SIGKILL'd by GPU pressure
            # from a concurrent training run: respawn the same config and retry the
            # generation once (bounded by the private flag, no duplicate output).
            if self._maybe_recover_from_mtp_crash(e):
                raise RuntimeError("Lost connection to llama-server")
            if _allow_respawn_retry and not cumulative and self._respawn_if_dead():
                logger.warning(
                    "llama-server was unreachable; respawned it and retrying the generation"
                )
                yield from _replay_on_replacement_server()
                return
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            if _is_gpu_device_lost(e):
                self._retire_device_lost_server(served_by)
                if _allow_respawn_retry and not cumulative and self._respawn_if_dead():
                    logger.warning(
                        "llama-server lost its GPU device; restarted it and retrying the generation"
                    )
                    yield from _replay_on_replacement_server()
                    return
                raise
            # Died mid-generation: recover MTP, re-raise unchanged for this request.
            self._maybe_recover_from_mtp_crash(e)
            raise
