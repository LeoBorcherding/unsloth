# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat tool: load a downloaded image model, render one image, then put the chat model back.

The whole swap runs inside one tool call, so the chat turn waiting on it resumes on the restored
model. Offered only while "Switch model by request" for images is on."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

USE_LIBRARY_MODEL_TOOL_NAME = "use_library_model"

ERR_NO_PROMPT = "Error: use_library_model needs a prompt."
ERR_NO_LOOP = "Error: Unsloth Studio could not reach its server loop to run this tool. Try again."
ERR_STOPPED = "Error: the model swap was stopped; the chat model was put back."
ERR_TIMEOUT = "Error: the model swap took too long and was stopped."
ERR_NON_GGUF_CHAT = (
    "Error: this image model does not fit beside the chat model, and only a GGUF chat model can be "
    "unloaded and put back mid-chat. Load the image model on the Images page instead."
)
ERR_RESTORE = "The chat model could not be loaded back ({detail}); reload it from the model picker."

_DESCRIPTION = (
    "Render one image with a downloaded image model from the Unsloth Studio library, even when "
    "it is not loaded. Unsloth Studio loads the image model, renders, unloads it and, if it had "
    "to unload the chat model, loads that back, so this can take a minute or more. "
    "Prefer generate_image when it is offered: that model is already loaded. Call without model "
    "to list the downloaded image models."
)

# The tool's choice, as the Keep multiple models loaded mode the load runs under.
_UNLOAD_MODES = {"true": "off", "auto": "fit", "false": "always"}
ERR_UNLOAD_ARG = "Error: unload_chat_model must be true, false or auto."
ERR_KEEP_NO_FIT = (
    "Error: '{model}' does not fit beside the chat model, so it was not loaded. Call again with "
    "unload_chat_model true (or auto) to unload the chat model for this image."
)

# Load + render + reload of a large model; a no-limit chat still gets this ceiling.
_MAX_WAIT_S = 1800.0
# After a stop, how long the swap may still take to put the chat model back.
_RESTORE_WAIT_S = 300.0
_LOAD_POLL_S = 1.0
_POLL_S = 0.25
_LIST_LIMIT = 40


def use_library_model_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": USE_LIBRARY_MODEL_TOOL_NAME,
            "description": _DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Id of a downloaded image model. Omit to list them.",
                    },
                    "prompt": {"type": "string", "description": "The full image prompt."},
                    "width": {"type": "integer", "description": "Width in pixels. Default 1024."},
                    "height": {"type": "integer", "description": "Height in pixels. Default 1024."},
                    "seed": {
                        "type": "integer",
                        "description": "Seed for a repeatable image. Random when omitted.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "What the image should not show, if the model supports it.",
                    },
                    "unload_chat_model": {
                        "type": "string",
                        "enum": ["auto", "true", "false"],
                        "description": (
                            "true: unload the chat model while the image model runs, then load "
                            "it back. false: keep it loaded, and refuse if both do not fit. auto: "
                            "keep it only if both fit. Omit to follow Keep multiple models loaded "
                            "in Settings."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        },
    }


def offers_use_library_model() -> bool:
    from utils.openai_auto_switch_settings import get_media_auto_switch_enabled
    from utils.account_context import is_owner_context

    # It unloads the shared chat model, which only the owner may do from a chat.
    return is_owner_context() and get_media_auto_switch_enabled()


def _listing(task: str) -> str:
    from core.inference.media_model_index import available_media_model_ids

    ids = available_media_model_ids(task)
    if not ids:
        return "Error: no image model is downloaded in Unsloth Studio. Download one on the Images page."
    shown = ", ".join(ids[:_LIST_LIMIT])
    more = f" (and {len(ids) - _LIST_LIMIT} more)" if len(ids) > _LIST_LIMIT else ""
    return (
        f"Downloaded image models: {shown}{more}. Call use_library_model again with one as model."
    )


def _unload_mode(arguments: dict) -> tuple[bool, Optional[str]]:
    """(valid, mode): the load's multi-model mode for unload_chat_model, None for the setting."""
    raw = arguments.get("unload_chat_model")
    if raw is None or raw == "":
        return True, None
    key = str(raw).strip().lower() if not isinstance(raw, bool) else str(raw).lower()
    mode = _UNLOAD_MODES.get(key)
    return mode is not None, mode


def _detail(exc: BaseException) -> str:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        error = detail.get("error")
        detail = detail.get("message") or (
            error.get("message") if isinstance(error, dict) else error
        )
    return str(detail or exc)


def _http_code(exc: BaseException) -> Optional[str]:
    headers = {k.lower(): v for k, v in (getattr(exc, "headers", None) or {}).items()}
    code = headers.get("x-unsloth-error-code")
    if code:
        return code
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        error = detail.get("error")
        if isinstance(error, dict):
            return error.get("code")
        return detail.get("code")
    return None


class _Swap:
    """What this call changed, so the end of the call can put it back."""

    def __init__(self) -> None:
        self.loaded_image = False
        self.chat_identity = None
        self.chat_launch = None
        self.chat_unloaded = False
        # The image model resident before the switch: a refused switch must leave it loaded.
        self.prior_image = None
        # Set when putting things back failed, appended to the tool result.
        self.note = ""


def _resident_image(status) -> Optional[str]:
    return status.get("repo_id") if isinstance(status, dict) and status.get("loaded") else None


async def _prior_still_resident(swap: "_Swap") -> bool:
    from core.inference.gpu_arbiter import DIFFUSION
    from core.inference.media_switch_backends import backend_for

    if swap.prior_image is None:
        return False
    return (
        _resident_image(await asyncio.to_thread(backend_for(DIFFUSION).status)) == swap.prior_image
    )


def _switch_lock():
    from core.inference.gpu_arbiter import DIFFUSION
    from core.inference.media_switch_locks import switch_lock
    return switch_lock(DIFFUSION)


async def _setup_settled(limit: float) -> None:
    """Waits out a switch setup still in preflight, so its load can't start after the put back."""
    end = time.monotonic() + limit
    lock = _switch_lock()
    while lock.locked() and time.monotonic() < end:
        await asyncio.sleep(_LOAD_POLL_S)


async def _await_image_loaded(name: str, pick, deadline: float, abort: threading.Event) -> None:
    from core.inference.gpu_arbiter import DIFFUSION
    from core.inference.media_model_index import satisfied_by
    from core.inference.media_switch_backends import backend_for

    while time.monotonic() < deadline and not abort.is_set():
        backend = backend_for(DIFFUSION)
        if satisfied_by(await asyncio.to_thread(backend.status), name, pick):
            return
        # The switch's setup holds its lock through preflight, before the backend registers a load.
        if not await asyncio.to_thread(backend.loading_repo_ids) and not _switch_lock().locked():
            if satisfied_by(await asyncio.to_thread(backend_for(DIFFUSION).status), name, pick):
                return
            raise RuntimeError(f"'{pick.model_id}' stopped loading before it was ready.")
        await asyncio.sleep(_LOAD_POLL_S)
    raise asyncio.TimeoutError


async def _load_image(name: str, pick, subject: str, deadline: float, abort) -> None:
    from fastapi import HTTPException

    from core.inference.gpu_arbiter import DIFFUSION
    from core.inference.media_auto_switch import maybe_auto_switch_media_model

    try:
        # openai_errors: only that shape carries the refusal code (model_loading, multi_model_always).
        await maybe_auto_switch_media_model(
            name, owner = DIFFUSION, current_subject = subject, openai_errors = True
        )
    except HTTPException as exc:
        # The switch answers within ~90 s and leaves a slow load running: wait the rest out here.
        if _http_code(exc) != "model_loading":
            raise
        await _await_image_loaded(name, pick, deadline, abort)


async def _unload_image(subject: str) -> None:
    from routes.inference import unload_diffusion_model
    await unload_diffusion_model(None, current_subject = subject, via_api_key = False)


async def _restore_chat(swap: _Swap, subject: str) -> Optional[str]:
    """Load the chat model this call evicted back, as the scheduled-task swap does. None on success."""
    from core.inference.llama_keepwarm import (
        end_in_process_request,
        loaded_identity,
        stash_for_reload,
    )
    from core.scheduled_tasks.scheduler import _producer_request
    from routes.inference import (
        _RELOAD_ONLY_MODEL,
        _maybe_auto_switch_model,
        allow_model_switch_for_request,
    )

    if swap.chat_identity is None or loaded_identity() is not None:
        return None
    stash_for_reload(swap.chat_identity, swap.chat_launch)
    request = _producer_request(None, "library-model-restore", threading.Event(), "UTC")
    allow_model_switch_for_request(request.scope)
    restored = False
    try:
        await _maybe_auto_switch_model(_RELOAD_ONLY_MODEL, request, subject)
        restored = loaded_identity() is not None
    except Exception as exc:  # noqa: BLE001 - reported in the tool result
        logger.warning("use_library_model: chat model reload failed: %s", exc, exc_info = True)
        return _detail(exc)
    finally:
        end_in_process_request(request.scope, answered = restored)
    return None if restored else "it did not come back up"


async def _swap_and_render(
    arguments: dict, prompt: str, swap: "_Swap", ticket, abort, deadline: float
):
    from core.inference.media_model_index import IMAGE_TASK, resolve_local_media_model
    from utils.multi_model_settings import multi_model_mode_for_request

    name = str(arguments.get("model") or "").strip()
    if not name:
        listing = await asyncio.to_thread(_listing, IMAGE_TASK)
        # A prompt with no model is a dropped argument, not a request for the list.
        return (
            listing
            if not prompt or listing.startswith("Error")
            else "No model was given, so no image was made. " + listing
        )
    pick = await asyncio.to_thread(resolve_local_media_model, name, task = IMAGE_TASK)
    if pick is None:
        return f"Error: no downloaded image model matches '{name}'. " + await asyncio.to_thread(
            _listing, IMAGE_TASK
        )
    valid, mode = _unload_mode(arguments)
    if not valid:
        return ERR_UNLOAD_ARG
    with multi_model_mode_for_request(mode):
        return await _swap_and_render_as(
            arguments, prompt, swap, ticket, abort, deadline, name, pick, mode
        )


async def _swap_and_render_as(arguments, prompt, swap, ticket, abort, deadline, name, pick, mode):
    from fastapi import HTTPException

    from core.inference.ensure_model import media_load_keeps_chat
    from core.inference.generate_image_tool import (
        ERR_EDIT_ONLY,
        _build_request,
        _generate,
        _http_error,
        offers_txt2img,
    )
    from core.inference.gpu_arbiter import DIFFUSION
    from core.inference.llama_keepwarm import launch_to_replay, loaded_identity
    from core.inference.media_chat_tools import media_chat_status
    from core.inference.media_model_index import IMAGE_TASK, satisfied_by
    from core.inference.media_switch_backends import backend_for
    from utils.account_context import current_account

    subject = current_account().username
    before = await asyncio.to_thread(backend_for(DIFFUSION).status)
    resident = satisfied_by(before, name, pick)
    swap.prior_image = _resident_image(before)
    keeps_chat = True
    if not resident:
        swap.chat_identity = loaded_identity()
        swap.chat_launch = launch_to_replay() if swap.chat_identity is not None else None
        keeps_chat = await asyncio.to_thread(media_load_keeps_chat, DIFFUSION)
        if swap.chat_identity is None and not keeps_chat:
            from core.inference import get_inference_backend
            orchestrator = await asyncio.to_thread(get_inference_backend)
            if getattr(orchestrator, "active_model_name", None):
                return ERR_NON_GGUF_CHAT
    try:
        if not resident:
            if mode == "always" and swap.chat_identity is not None and not keeps_chat:
                return ERR_KEEP_NO_FIT.format(model = pick.model_id)
            swap.loaded_image = True
            try:
                await _load_image(name, pick, subject, deadline, abort)
            finally:
                swap.chat_unloaded = swap.chat_identity is not None and loaded_identity() is None
        if abort.is_set():
            return ERR_STOPPED
        status = media_chat_status(IMAGE_TASK)
        if status is None:
            return f"Error: '{pick.model_id}' did not finish loading."
        if not offers_txt2img(status):
            return ERR_EDIT_ONLY
        try:
            request = _build_request(arguments, prompt, status)
        except (TypeError, ValueError) as exc:
            return f"Error: invalid use_library_model arguments: {exc}"
        try:
            response = await _generate(request, ticket)
        except HTTPException as exc:
            return ERR_STOPPED if abort.is_set() else _http_error(exc)
        return response
    except asyncio.TimeoutError:
        return ERR_TIMEOUT
    except HTTPException as exc:
        # The Always refusal names the setting, which this call overrode.
        if mode == "always" and _http_code(exc) == "multi_model_always":
            return ERR_KEEP_NO_FIT.format(model = pick.model_id)
        return "Error: " + _detail(exc)
    finally:
        # Shielded: a stop must never leave the chat without its model.
        swap.note = await asyncio.shield(_put_back(swap, subject))


async def _put_back(swap: _Swap, subject: str) -> str:
    if swap.loaded_image:
        await _setup_settled(_RESTORE_WAIT_S)
    if swap.loaded_image and not await _prior_still_resident(swap):
        try:
            await _unload_image(subject)
        except Exception as exc:  # noqa: BLE001
            logger.warning("use_library_model: image model unload failed: %s", exc, exc_info = True)
    failure = await _restore_chat(swap, subject)
    return ERR_RESTORE.format(detail = failure) if failure else ""


def _run(arguments: dict, workdir: str, cancel_event, timeout: Optional[float]) -> str:
    from core.inference.generate_image_tool import _PROMPT_ECHO_CHARS, _save_to_workdir
    from core.inference.generation_ticket import GenerationTicket
    from core.inference.media_chat_tools import server_loop
    from utils.account_context import arun_as, current_account

    prompt = str(arguments.get("prompt") or "").strip()
    if not prompt and str(arguments.get("model") or "").strip():
        return ERR_NO_PROMPT
    account = current_account()
    loop = server_loop()
    if loop is None:
        return ERR_NO_LOOP
    limit = _MAX_WAIT_S if timeout is None or timeout >= 9999 else max(1.0, float(timeout))
    deadline = time.monotonic() + limit
    ticket = GenerationTicket()
    abort = threading.Event()
    swap = _Swap()
    future = asyncio.run_coroutine_threadsafe(
        arun_as(account, _swap_and_render(arguments, prompt, swap, ticket, abort, deadline)), loop
    )
    stopped = None
    while not future.done():
        if stopped is None and cancel_event is not None and cancel_event.is_set():
            stopped = ERR_STOPPED
        elif stopped is None and time.monotonic() >= deadline:
            stopped = ERR_TIMEOUT
        if stopped:
            # Not future.cancel(): the coroutine still has to put the chat model back.
            abort.set()
            ticket.stop()
            concurrent.futures.wait([future], timeout = _RESTORE_WAIT_S)
            break
        concurrent.futures.wait([future], timeout = _POLL_S)
    if not future.done():
        return stopped or ERR_TIMEOUT
    result = future.result()
    note = swap.note
    if isinstance(result, str):
        return result + (" " + note if note else "")
    if swap.chat_unloaded:
        note = ("The chat model was unloaded for this image and loaded back. " + note).strip()
    elif swap.loaded_image and swap.chat_identity is not None:
        note = ("The chat model stayed loaded beside the image model. " + note).strip()
    if not result.images:
        return "Error: the image model returned no image." + (" " + note if note else "")
    image = result.images[0]
    saved = _save_to_workdir(image, workdir, account)
    shown = " ".join(prompt.split())[:_PROMPT_ECHO_CHARS]
    model = str(arguments.get("model") or "").strip()
    text = (
        f"Generated a {image.width}x{image.height} image with {model} (seed {image.seed}) and "
        f"saved it as {saved['name']}. Prompt used: {shown}"
    )
    if note:
        text += " " + note
    return (
        text + "\n__FILES__:" + json.dumps([saved]) + "\n__IMAGES__:" + json.dumps([saved["name"]])
    )


def run_use_library_model(
    arguments: Any,
    *,
    workdir: str,
    cancel_event = None,
    timeout: Optional[float] = None,
) -> str:
    """Swap in an image model, render one image into ``workdir``, swap back; always a string."""
    try:
        return _run(
            arguments if isinstance(arguments, dict) else {}, workdir, cancel_event, timeout
        )
    except Exception as exc:  # noqa: BLE001 - a tool result, never a crashed turn
        logger.warning("use_library_model failed: %s", exc, exc_info = True)
        return f"Error: use_library_model failed: {exc}"
