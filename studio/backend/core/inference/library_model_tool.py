# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat tool: load a downloaded image or video model, render one image or clip, then put the chat model back.

The whole swap runs inside one tool call, so the chat turn waiting on it resumes on the restored
model. Offered only while "Switch model by request" for images is on."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
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
    "Error: this model cannot run beside the chat model, and only a GGUF chat model can be "
    "unloaded and put back mid-chat. Run it from its own page instead."
)
ERR_RESTORE = "The chat model could not be loaded back ({detail}); reload it from the model picker."

_DESCRIPTION = (
    "Use any model downloaded in Unsloth Studio, even when it is not loaded: ask another chat "
    "model (LLM) a question and get its answer, or render an image or a video clip. Unsloth "
    "Studio loads that model, runs it, unloads it and, if it had to unload the chat model, loads "
    "that back, so this can take a minute or more. Prefer generate_image or generate_video when "
    "offered: that model is already loaded. Call without model to list the downloaded models."
)

# The tool's choice, as the Keep multiple models loaded mode the load runs under.
_UNLOAD_MODES = {"true": "off", "auto": "fit", "false": "always"}
ERR_UNLOAD_ARG = "Error: unload_chat_model must be true, false or auto."
ERR_KEEP_NO_FIT = (
    "Error: '{model}' does not fit beside the chat model, so it was not loaded. Call again with "
    "unload_chat_model true (or auto) to unload the chat model for this {kind}."
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
                        "description": "Id of a downloaded model (chat, image, video, music, text-to-speech, speech-to-text, decision). Omit to list them.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "For a chat model, the message to send it; for text-to-speech, the text to speak; for an image or video model, the full prompt.",
                    },
                    "width": {"type": "integer", "description": "Width in pixels. Default 1024."},
                    "height": {"type": "integer", "description": "Height in pixels. Default 1024."},
                    "audio": {
                        "type": "string",
                        "description": "For a speech-to-text model: the name of an audio file in this chat.",
                    },
                    "lyrics": {"type": "string", "description": "Song lyrics, for a music model."},
                    "questions": {
                        "type": "object",
                        "description": (
                            "For a decision model: a name per question mapped to {type, instructions, "
                            "criteria}; type noul is yes/no, choice needs criteria per option, score "
                            "needs criteria from 1 to 10."
                        ),
                    },
                    "seconds": {
                        "type": "number",
                        "description": "Clip length in seconds, for a video model.",
                    },
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
                            "true: unload the chat model while the other model runs, then load "
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


def _listing(groups: tuple = ()) -> str:
    """``groups`` adds (kind, ids) pairs the media index does not cover, such as chat models."""
    from core.inference.media_model_index import IMAGE_TASK, VIDEO_TASK, available_media_model_ids

    parts = []
    found = [
        (kind, available_media_model_ids(task))
        for kind, task in (("image", IMAGE_TASK), ("video", VIDEO_TASK))
    ]
    for kind, ids in [*groups, *found]:
        if ids:
            more = f" (and {len(ids) - _LIST_LIMIT} more)" if len(ids) > _LIST_LIMIT else ""
            parts.append(f"Downloaded {kind} models: {', '.join(ids[:_LIST_LIMIT])}{more}.")
    if not parts:
        return (
            "Error: no model is downloaded in Unsloth Studio. Download one from the model "
            "picker, the Images page or the Video page."
        )
    return " ".join(parts) + " Call use_library_model again with one as model."


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
        from core.inference.gpu_arbiter import DIFFUSION

        self.owner = DIFFUSION
        self.kind = "image"
        self.loaded_media = False
        # What this call loaded, so the put back unloads only that.
        self.name = ""
        self.pick = None
        self.chat_identity = None
        self.chat_launch = None
        self.chat_unloaded = False
        self.shared: Optional["_Shared"] = None
        # Another chat was still rendering on the model when this call ended, so it puts things back.
        self.deferred = False
        # Set when putting things back failed, appended to the tool result.
        self.note = ""


class _Shared:
    """The calls using one media backend right now: only the last one out puts things back."""

    def __init__(self) -> None:
        self.users = 0
        # The first call that loaded holds the chat model to bring back; every call that loaded is
        # checked at the end, since a later one may have switched the backend to its own pick.
        self.loader: Optional[_Swap] = None
        self.loaded: list = []


# Keyed by owner. Touched only on the server loop, never across an await, so no lock.
_SHARED: dict = {}


def _enter_shared(swap: "_Swap") -> None:
    swap.shared = _SHARED.setdefault(swap.owner, _Shared())
    swap.shared.users += 1


async def _ours_resident(swap: "_Swap") -> bool:
    """This call's pick is still the resident model: a refused switch left the prior one, and another
    request may have switched in a newer one since."""
    from core.inference.media_model_index import satisfied_by
    from core.inference.media_switch_backends import backend_for

    if swap.pick is None:
        return False
    status = await asyncio.to_thread(backend_for(swap.owner).status)
    return satisfied_by(status, swap.name, swap.pick)


def _switch_lock(owner: str):
    from core.inference.media_switch_locks import switch_lock
    return switch_lock(owner)


async def _setup_settled(owner: str, limit: float) -> None:
    """Waits out a switch setup still in preflight, and any other render on the backend, before the put back."""
    from core.inference.media_switch_backends import backend_for

    end = time.monotonic() + limit
    lock = _switch_lock(owner)
    while time.monotonic() < end:
        progress = getattr(backend_for(owner), "generate_progress", None)
        rendering = bool(progress and ((await asyncio.to_thread(progress)) or {}).get("active"))
        if not lock.locked() and not rendering:
            return
        await asyncio.sleep(_LOAD_POLL_S)


async def _await_loaded(
    owner: str, name: str, pick, deadline: float, abort: threading.Event
) -> None:
    from core.inference.media_model_index import satisfied_by
    from core.inference.media_switch_backends import backend_for

    while time.monotonic() < deadline and not abort.is_set():
        backend = backend_for(owner)
        if satisfied_by(await asyncio.to_thread(backend.status), name, pick):
            return
        # The switch's setup holds its lock through preflight, before the backend registers a load.
        if (
            not await asyncio.to_thread(backend.loading_repo_ids)
            and not _switch_lock(owner).locked()
        ):
            if satisfied_by(await asyncio.to_thread(backend.status), name, pick):
                return
            raise RuntimeError(f"'{pick.model_id}' stopped loading before it was ready.")
        await asyncio.sleep(_LOAD_POLL_S)
    raise asyncio.TimeoutError


async def _load_media(owner: str, name: str, pick, subject: str, deadline: float, abort) -> None:
    from fastapi import HTTPException

    from core.inference.media_auto_switch import maybe_auto_switch_media_model
    try:
        # openai_errors: only that shape carries the refusal code (model_loading, multi_model_always).
        await maybe_auto_switch_media_model(
            name, owner = owner, current_subject = subject, openai_errors = True
        )
    except HTTPException as exc:
        # The switch answers within ~90 s and leaves a slow load running: wait the rest out here.
        if _http_code(exc) != "model_loading":
            raise
        await _await_loaded(owner, name, pick, deadline, abort)


async def _unload_media(owner: str, subject: str) -> None:
    from core.inference.gpu_arbiter import VIDEO
    if owner == VIDEO:
        from routes.video import unload_video_model as unload
    else:
        from routes.inference import unload_diffusion_model as unload
    await unload(None, current_subject = subject, via_api_key = False)


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


# The call's Stop/deadline event, handed to the in-process requests a non-media model runs on.
_ABORT: contextvars.ContextVar = contextvars.ContextVar("library_model_abort", default = None)


def _abort_event() -> threading.Event:
    return _ABORT.get() or threading.Event()


async def _swap_and_render(
    arguments: dict, prompt: str, swap: "_Swap", ticket, abort, deadline: float, workdir: str
):
    from routes.inference import _AUDIO_RUN_CANCEL

    _ABORT.set(abort)
    # TTS and music read their stop from the audio run's event.
    _AUDIO_RUN_CANCEL.set(abort)
    from core.inference.gpu_arbiter import VIDEO
    from core.inference.media_model_index import IMAGE_TASK, VIDEO_TASK, resolve_local_media_model
    from utils.multi_model_settings import multi_model_mode_for_request

    name = str(arguments.get("model") or "").strip()
    if not name and arguments.get("questions"):
        # Only a decision model takes questions, so a dropped model means the one in Settings.
        name = await asyncio.to_thread(_settings_decision_model) or ""
    if not name:
        listing = await _full_listing()
        # A prompt with no model is a dropped argument, not a request for the list.
        return (
            listing
            if not prompt or listing.startswith("Error")
            else "No model was given, so nothing was made. " + listing
        )
    pick = await asyncio.to_thread(resolve_local_media_model, name, task = IMAGE_TASK)
    if pick is None:
        pick = await asyncio.to_thread(resolve_local_media_model, name, task = VIDEO_TASK)
        if pick is not None:
            swap.owner, swap.kind = VIDEO, "video"
    if pick is None:
        from utils.account_context import current_account

        subject = current_account().username
        music = await asyncio.to_thread(_music_model, name)
        if music is not None:
            return await _on_chat_seat(
                arguments,
                music,
                swap,
                lambda beside: _compose_music(
                    music, prompt, arguments, subject, workdir, beside = beside
                ),
            )
        decision = await asyncio.to_thread(_decision_model, name)
        if decision is not None:
            return await _decide_with(decision, prompt, arguments, swap)
        row = await _catalog_row(name)
        task, model = (getattr(row, "task", None), row.id) if row is not None else ("missing", name)
        if task == "missing":
            # audio.cpp voices and Whisper are listed apart from the catalog, as /v1/models does.
            tts, stt = await asyncio.to_thread(_speech_ids)
            wanted = name.strip().lower()
            for ids, kind in ((tts, _TTS_TASK), (stt, _STT_TASK)):
                hit = next((i for i in ids if i.lower() == wanted), None)
                if hit is not None:
                    task, model = kind, hit
                    break
        if task in _LLM_TASKS:
            return await _ask_llm(arguments, prompt, model, swap)
        if task == _TTS_TASK:
            return await _on_chat_seat(
                arguments,
                model,
                swap,
                lambda beside: _speak_tts(model, prompt, subject, workdir, beside = beside),
            )
        if task == _STT_TASK:
            return await _transcribe(model, arguments, workdir)
        return f"Error: no downloaded model matches '{name}'. " + await _full_listing()
    valid, mode = _unload_mode(arguments)
    if not valid:
        return ERR_UNLOAD_ARG
    with multi_model_mode_for_request(mode):
        return await _swap_and_render_as(
            arguments, prompt, swap, ticket, abort, deadline, name, pick, mode, workdir
        )


async def _swap_and_render_as(
    arguments, prompt, swap, ticket, abort, deadline, name, pick, mode, workdir
):
    from fastapi import HTTPException

    from core.inference.ensure_model import media_load_keeps_chat
    from core.inference.llama_keepwarm import launch_to_replay, loaded_identity
    from core.inference.media_model_index import satisfied_by
    from core.inference.media_switch_backends import backend_for
    from utils.account_context import current_account

    subject = current_account().username
    before = await asyncio.to_thread(backend_for(swap.owner).status)
    resident = satisfied_by(before, name, pick)
    keeps_chat = True
    if not resident:
        swap.chat_identity = loaded_identity()
        swap.chat_launch = launch_to_replay() if swap.chat_identity is not None else None
        keeps_chat = await asyncio.to_thread(media_load_keeps_chat, swap.owner)
        if swap.chat_identity is None and not keeps_chat:
            from core.inference import get_inference_backend
            orchestrator = await asyncio.to_thread(get_inference_backend)
            if getattr(orchestrator, "active_model_name", None):
                return ERR_NON_GGUF_CHAT
    _enter_shared(swap)
    try:
        if not resident:
            if mode == "always" and swap.chat_identity is not None and not keeps_chat:
                return ERR_KEEP_NO_FIT.format(model = pick.model_id, kind = swap.kind)
            swap.loaded_media = True
            swap.name, swap.pick = name, pick
            try:
                await _load_media(swap.owner, name, pick, subject, deadline, abort)
            finally:
                swap.chat_unloaded = swap.chat_identity is not None and loaded_identity() is None
        if abort.is_set():
            return ERR_STOPPED
        if swap.kind == "video":
            return _Clip(await _render_video(arguments, prompt, abort, deadline, workdir))
        return await _render_image(arguments, prompt, pick, ticket, abort)
    except asyncio.TimeoutError:
        return ERR_TIMEOUT
    except HTTPException as exc:
        # The Always refusal names the setting, which this call overrode.
        if mode == "always" and _http_code(exc) == "multi_model_always":
            return ERR_KEEP_NO_FIT.format(model = pick.model_id, kind = swap.kind)
        return "Error: " + _detail(exc)
    finally:
        # Shielded: a stop must never leave the chat without its model.
        swap.note = await asyncio.shield(_put_back(swap, subject))


class _Clip(str):
    """A finished video tool result, told apart from an error string."""


async def _render_image(arguments, prompt, pick, ticket, abort):
    from fastapi import HTTPException

    from core.inference.generate_image_tool import (
        ERR_EDIT_ONLY,
        _build_request,
        _generate,
        _http_error,
        offers_txt2img,
    )
    from core.inference.media_chat_tools import media_chat_status
    from core.inference.media_model_index import IMAGE_TASK

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
        return await _generate(request, ticket)
    except HTTPException as exc:
        return ERR_STOPPED if abort.is_set() else _http_error(exc)


async def _render_video(arguments, prompt, abort, deadline, workdir) -> str:
    from core.inference.generate_video_tool import run_generate_video

    fields = {
        k: arguments[k]
        for k in ("seconds", "width", "height", "seed")
        if arguments.get(k) is not None
    }
    # Off the loop: generate_video blocks on its own coroutines, which run on this loop.
    budget = max(1.0, deadline - time.monotonic())
    # The whole budget for the create too: the put back unloads the model, which would cancel a
    # clip whose create was still running when generate_video's own short create wait ran out.
    return await asyncio.to_thread(
        run_generate_video,
        {"prompt": prompt, **fields},
        workdir = workdir,
        cancel_event = abort,
        timeout = budget,
        create_timeout = budget,
    )


async def _put_back(swap: _Swap, subject: str) -> str:
    shared = swap.shared
    loaded = [swap] if swap.loaded_media else []
    if shared is not None:
        shared.users -= 1
        if swap.loaded_media:
            shared.loaded.append(swap)
            if shared.loader is None:
                shared.loader = swap
        if shared.users > 0:
            # Unloading or bringing the chat model back now would end that other render.
            swap.deferred = swap.chat_unloaded
            return ""
        _SHARED.pop(swap.owner, None)
        loaded = shared.loaded
        swap = shared.loader or swap
    if loaded:
        await _setup_settled(swap.owner, _RESTORE_WAIT_S)
    if loaded and any([await _ours_resident(each) for each in loaded]):
        try:
            await _unload_media(swap.owner, subject)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "use_library_model: %s model unload failed: %s", swap.kind, exc, exc_info = True
            )
    failure = await _restore_chat(swap, subject)
    return ERR_RESTORE.format(detail = failure) if failure else ""


def _run(arguments: dict, workdir: str, cancel_event, timeout: Optional[float]) -> str:
    from core.inference.generate_image_tool import _PROMPT_ECHO_CHARS, _save_to_workdir
    from core.inference.generation_ticket import GenerationTicket
    from core.inference.media_chat_tools import server_loop
    from utils.account_context import arun_as, current_account

    prompt = str(arguments.get("prompt") or "").strip()
    if not prompt and not arguments.get("audio") and str(arguments.get("model") or "").strip():
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
        arun_as(
            account, _swap_and_render(arguments, prompt, swap, ticket, abort, deadline, workdir)
        ),
        loop,
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
    if isinstance(result, str) and not (
        isinstance(result, _Clip) and not result.startswith("Error")
    ):
        return result + (" " + note if note else "")
    if swap.deferred:
        note = (
            f"The chat model was unloaded for this {swap.kind}. It comes back when another chat's "
            f"render on the same model finishes. {note}"
        ).strip()
    elif swap.chat_unloaded:
        note = f"The chat model was unloaded for this {swap.kind} and loaded back. {note}".strip()
    elif swap.loaded_media and swap.chat_identity is not None:
        note = f"The chat model stayed loaded beside the {swap.kind} model. {note}".strip()
    if isinstance(result, _Clip):
        text, sep, files = result.partition("\n__FILES__")
        model = str(arguments.get("model") or "").strip()
        return (
            f"{text.replace('Generated a video', f'Generated a video with {model}', 1)} {note}".strip()
            + sep
            + files
        )
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
    """Swap in an image or video model, render one image or clip into ``workdir``, swap back; always a string."""
    try:
        return _run(
            arguments if isinstance(arguments, dict) else {}, workdir, cancel_event, timeout
        )
    except Exception as exc:  # noqa: BLE001 - a tool result, never a crashed turn
        logger.warning("use_library_model failed: %s", exc, exc_info = True)
        return f"Error: use_library_model failed: {exc}"


_LLM_TASKS = (None, "", "text-generation", "image-text-to-text")
_TTS_TASK = "text-to-speech"
_STT_TASK = "automatic-speech-recognition"


async def _catalog_row(name: str):
    """The downloaded model row Studio's picker shows for ``name``, or None."""
    from routes.inference import _cached_local_catalog

    wanted = name.strip().lower()
    for row in await _cached_local_catalog():
        ids = (getattr(row, "id", None), getattr(row, "model_id", None))
        if any(str(i or "").strip().lower() == wanted for i in ids):
            return row
    return None


async def _downloaded_llms() -> tuple:
    """(kind, ids) groups the media index does not list."""
    from routes.inference import _cached_local_catalog

    rows = await _cached_local_catalog()
    groups = (
        ("chat", _LLM_TASKS),
        ("text-to-speech", (_TTS_TASK,)),
        ("speech-to-text", (_STT_TASK,)),
    )
    music = await asyncio.to_thread(_music_ids)
    decisions = await asyncio.to_thread(_decision_ids)
    tts, stt = await asyncio.to_thread(_speech_ids)
    extra = {"text-to-speech": tts, "speech-to-text": stt}
    out = []
    for kind, tasks in groups:
        ids = [row.id for row in rows if getattr(row, "task", None) in tasks]
        out.append((kind, ids + [i for i in extra.get(kind, ()) if i not in ids]))
    return tuple(out) + (("music", music), ("decision", decisions))


def _speech_ids() -> tuple[list[str], list[str]]:
    """(tts, stt) ids the catalog skips: audio.cpp voices and the STT sidecars' models."""
    from routes.inference import _stt_model_objects

    try:
        from core.inference.audio_cpp_models import downloaded_models
        tts = [model.id for model in downloaded_models("tts")]
    except Exception:  # noqa: BLE001 - no audio.cpp here
        tts = []
    return tts, [obj["id"] for obj in _stt_model_objects(0)]


def _evict_chat_model() -> None:
    """As an image load does mid tool call: the chat switch would wait on this chat's own request."""
    from core.inference import gpu_arbiter

    gpu_arbiter._evict_chat()
    gpu_arbiter.release(gpu_arbiter.CHAT)


async def _unload_llm(name: str, subject: str) -> None:
    from core.scheduled_tasks.scheduler import _model_is_loaded
    from models.inference import UnloadRequest
    from routes.inference import _unload_model_impl

    if _model_is_loaded(name):
        # Its own task, so the unload's slot routing cannot stick to this context.
        await asyncio.ensure_future(_unload_model_impl(UnloadRequest(model_path = name), subject))


async def _complete_llm(name: str, prompt: str, subject: str, *, beside: bool) -> tuple[str, bool]:
    """(text, refused): one completion on ``name``, loaded by the chat switch as scheduled tasks do."""
    from fastapi import HTTPException

    from core.inference.gpu_arbiter import MULTI_MODEL_REFUSAL_MARKERS
    from core.inference.llama_keepwarm import end_in_process_request
    from core.scheduled_tasks.scheduler import _producer_request
    from models.inference import ChatCompletionRequest
    from routes.inference import (
        allow_model_switch_for_request,
        load_alongside_for_request,
        produce_openai_chat_completions,
    )

    request = _producer_request(None, "library-model-ask", _abort_event(), "UTC", loaded_only = False)
    allow_model_switch_for_request(request.scope)
    if beside:
        load_alongside_for_request(request.scope)
    payload = ChatCompletionRequest.model_validate(
        {"model": name, "messages": [{"role": "user", "content": prompt}], "stream": False}
    )
    reply = None
    try:
        reply = await produce_openai_chat_completions(
            payload, request, subject, cancel_on_disconnect = False
        )
    except HTTPException as exc:
        text = _detail(exc)
        if any(marker in text for marker in MULTI_MODEL_REFUSAL_MARKERS):
            return text, True
        return "Error: " + text, False
    finally:
        end_in_process_request(
            request.scope,
            answered = reply is not None and int(getattr(reply, "status_code", 200)) < 400,
        )
    data = json.loads(bytes(getattr(reply, "body", b"") or b"{}"))
    if int(getattr(reply, "status_code", 200)) >= 400:
        text = str((data.get("error") or {}).get("message") or data.get("detail") or data)
        refused = any(marker in text for marker in MULTI_MODEL_REFUSAL_MARKERS)
        return (text if refused else "Error: " + text), refused
    message = ((data.get("choices") or [{}])[0].get("message")) or {}
    return str(message.get("content") or "").strip(), False


def _with_note(text: str, note: str) -> str:
    """The note goes above a file envelope, which has to stay last."""
    body, sep, files = text.partition("\n__FILES__")
    return f"{body}\n\n{note}" + sep + files


async def _on_chat_seat(arguments: dict, name: str, swap: "_Swap", run) -> str:
    """A model that loads through the chat switch (an LLM, a TTS model): beside the chat model when it
    stays, else after unloading it. ``run(beside)`` returns (text, refused)."""
    from core.inference.llama_keepwarm import launch_to_replay, loaded_identity
    from core.scheduled_tasks.scheduler import _model_is_loaded
    from utils.account_context import current_account
    from utils.multi_model_settings import get_multi_model_mode, multi_model_mode_for_request

    subject = current_account().username
    valid, mode = _unload_mode(arguments)
    if not valid:
        return ERR_UNLOAD_ARG
    mode = mode or await asyncio.to_thread(get_multi_model_mode)
    if _model_is_loaded(name):
        return (await run(True))[0]
    if mode in ("always", "fit"):
        with multi_model_mode_for_request("always"):
            text, refused = await run(True)
        if not refused:
            await _unload_llm(name, subject)
            if text.startswith("Error"):
                return text
            return _with_note(text, f"{name} ran beside the chat model, which stayed loaded.")
        if mode == "always":
            return ERR_KEEP_NO_FIT.format(model = name, kind = "model")
    swap.chat_identity = loaded_identity()
    swap.chat_launch = launch_to_replay() if swap.chat_identity is not None else None
    if swap.chat_identity is None and await _non_gguf_chat():
        return ERR_NON_GGUF_CHAT
    try:
        await asyncio.to_thread(_evict_chat_model)
        with multi_model_mode_for_request("off"):
            text, _ = await run(False)
        # It holds the chat seat now, and an unload there would wait on this chat's own request.
        await asyncio.to_thread(_evict_chat_model)
        await _unload_llm(name, subject)
    finally:
        # Shielded: a stop must never leave the chat without its model.
        swap.note = await asyncio.shield(_put_back(swap, subject))
    if text.startswith("Error"):
        return text
    return _with_note(text, f"The chat model was unloaded so {name} could run, and loaded back.")


async def _non_gguf_chat() -> bool:
    """A Transformers chat model holds the GPU but has no launch to restore it from."""
    from core.inference import get_inference_backend

    orchestrator = await asyncio.to_thread(get_inference_backend)
    return bool(getattr(orchestrator, "active_model_name", None))


async def _ask_llm(arguments: dict, prompt: str, name: str, swap: "_Swap") -> str:
    from utils.account_context import current_account

    subject = current_account().username

    async def run(beside: bool) -> tuple[str, bool]:
        text, refused = await _complete_llm(name, prompt, subject, beside = beside)
        return (
            text if refused or text.startswith("Error") else f"{name} answered:\n{text}"
        ), refused

    return await _on_chat_seat(arguments, name, swap, run)


async def _speak_tts(
    name: str, text: str, subject: str, workdir: str, *, beside: bool
) -> tuple[str, bool]:
    """(result, refused): one clip from TTS model ``name``, saved into the chat's sandbox."""
    import hashlib
    import os

    from fastapi import HTTPException

    from core.inference.gpu_arbiter import MULTI_MODEL_REFUSAL_MARKERS
    from core.inference.llama_keepwarm import end_in_process_request
    from core.inference.orchestrator import AUDIO_GENERATION_MAX_TOKENS
    from core.scheduled_tasks.scheduler import _producer_request
    from models.inference import ChatCompletionRequest
    from routes.inference import (
        _generate_tts_wav,
        _persist_tts_clip,
        allow_model_switch_for_request,
        load_alongside_for_request,
    )

    request = _producer_request(
        None, "library-model-speak", _abort_event(), "UTC", loaded_only = False
    )
    allow_model_switch_for_request(request.scope)
    if beside:
        load_alongside_for_request(request.scope)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": text}], max_tokens = AUDIO_GENERATION_MAX_TOKENS
    )
    wav = None
    try:
        wav, rate, served, audio_type = await _generate_tts_wav(
            text,
            payload,
            request,
            subject,
            speech_api_default_max_tokens = True,
            requested_model = name,
        )
    except HTTPException as exc:
        detail = _detail(exc)
        if any(marker in detail for marker in MULTI_MODEL_REFUSAL_MARKERS):
            return detail, True
        return "Error: " + detail, False
    finally:
        end_in_process_request(request.scope, answered = wav is not None)
    # The Audio page keeps it too, as for the speech route.
    await asyncio.to_thread(_persist_tts_clip, wav, rate, text, served, audio_type)
    root = os.path.realpath(workdir)
    os.makedirs(root, exist_ok = True)
    file = f"generated-speech-{hashlib.sha1(wav).hexdigest()[:8]}.wav"
    with open(os.path.join(root, file), "wb") as out:
        out.write(wav)
    saved = {"name": file, "size": len(wav)}
    return f"Spoke the text with {name} and saved it as {file}." + "\n__FILES__:" + json.dumps(
        [saved]
    ), False


async def _transcribe(name: str, arguments: dict, workdir: str) -> str:
    """STT runs beside whatever is loaded (no GPU arbiter owner), so there is no chat swap."""
    import os

    from fastapi import HTTPException

    from core.inference import stt_registry
    from core.scheduled_tasks.scheduler import _producer_request
    from routes.inference import _transcribe_audio_result

    audio = str(arguments.get("audio") or "").strip()
    if not audio:
        return "Error: a speech-to-text model needs audio: the name of an audio file in this chat."
    root = os.path.realpath(workdir)
    path = os.path.realpath(os.path.join(root, audio))
    if not path.startswith(root + os.sep) or not os.path.isfile(path):
        return f"Error: no audio file named '{audio}' in this chat."
    raw = await asyncio.to_thread(lambda: open(path, "rb").read())
    before = (await asyncio.to_thread(stt_registry.resident)).get("model")
    request = _producer_request(
        None, "library-model-transcribe", _abort_event(), "UTC", loaded_only = False
    )
    try:
        result = await _transcribe_audio_result(raw, name, None, False, request = request)
    except HTTPException as exc:
        return "Error: " + _detail(exc)
    finally:
        after = (await asyncio.to_thread(stt_registry.resident)).get("model")
        # Only the model this call named: another transcription may have loaded a different one since.
        if after and after != before and _same_stt(after, name):
            await asyncio.to_thread(stt_registry.unload, expected_model = after)
    return f"{name} transcribed {audio}:\n{str(result.get('text') or '').strip()}"


def _same_stt(resident: str, name: str) -> bool:
    """Whisper reports its short key ("tiny"); the listing names the repo."""
    try:
        from core.inference.stt_sidecar import STT_MODELS
    except Exception:  # noqa: BLE001
        STT_MODELS = {}
    ids = {resident.lower(), str(STT_MODELS.get(resident, "")).lower()}
    return name.strip().lower() in ids


async def _full_listing() -> str:
    return await asyncio.to_thread(_listing, await _downloaded_llms())


async def _decide_with(name: str, prompt: str, arguments: dict, swap: "_Swap") -> str:
    """A decision model cannot load while the chat model holds the GPU, so only false keeps it loaded,
    and false is then refused."""
    from fastapi import HTTPException

    from core.inference.llama_keepwarm import launch_to_replay, loaded_identity
    from core.systemone import catalog, laya_runtime
    from routes.systemone import QuestionIn, _decide, _require_enabled
    from utils.account_context import current_account
    from utils.multi_model_settings import get_multi_model_mode

    questions = arguments.get("questions")
    if not isinstance(questions, dict) or not questions:
        return (
            "Error: a decision model needs questions, for example "
            '{"urgent": {"type": "noul", "instructions": "Does this need a reply today?"}}.'
        )
    valid, mode = _unload_mode(arguments)
    if not valid:
        return ERR_UNLOAD_ARG
    mode = mode or await asyncio.to_thread(get_multi_model_mode)
    try:
        await asyncio.to_thread(_require_enabled)
        parsed = {key: QuestionIn.model_validate(value) for key, value in questions.items()}
    except HTTPException as exc:
        return "Error: " + _detail(exc)
    except Exception as exc:  # noqa: BLE001 - a malformed question from the model
        return f"Error: invalid questions: {exc}"
    checkpoint = await asyncio.to_thread(catalog.resolve, name)
    holds_gpu = loaded_identity() is not None
    non_gguf = not holds_gpu and await _non_gguf_chat()
    if (holds_gpu or non_gguf) and mode == "always":
        return ERR_KEEP_NO_FIT.format(model = name, kind = "model")
    if non_gguf:
        return ERR_NON_GGUF_CHAT
    subject = current_account().username
    if holds_gpu:
        swap.chat_identity = loaded_identity()
        swap.chat_launch = launch_to_replay()
    try:
        if holds_gpu:
            await asyncio.to_thread(_evict_chat_model)
        result = await _decide(checkpoint, prompt, parsed)
        result.pop("_backend", None)
    except HTTPException as exc:
        return "Error: " + _detail(exc)
    finally:
        await asyncio.to_thread(laya_runtime.unload)
        swap.note = await asyncio.shield(_put_back(swap, subject))
    text = f"{name} decided:\n{json.dumps(result, ensure_ascii = False)}"
    if holds_gpu:
        return _with_note(
            text, f"The chat model was unloaded so {name} could run, and loaded back."
        )
    return text


async def _compose_music(
    name: str, prompt: str, arguments: dict, subject: str, workdir: str, *, beside: bool
) -> tuple[str, bool]:
    """(result, refused): one track from music model ``name``, copied into the chat's sandbox."""
    import os
    import shutil

    from fastapi import HTTPException

    from core.inference import audio_gallery
    from core.inference.gpu_arbiter import MULTI_MODEL_REFUSAL_MARKERS
    from core.inference.llama_keepwarm import end_in_process_request
    from core.scheduled_tasks.scheduler import _producer_request
    from models.inference import AudioRunRequest
    from routes.inference import (
        _run_music_workflow,
        allow_model_switch_for_request,
        load_alongside_for_request,
    )

    request = _producer_request(
        None, "library-model-music", _abort_event(), "UTC", loaded_only = False
    )
    allow_model_switch_for_request(request.scope)
    if beside:
        load_alongside_for_request(request.scope)
    lyrics = str(arguments.get("lyrics") or "")
    # The music workflow needs a mode; with no lyrics a song is instrumental.
    fields = {"workflow": "music", "model": name, "text": prompt, "mode": "song"}
    fields["instrumental"] = not lyrics
    if arguments.get("seconds") is not None:
        fields["duration_s"] = float(arguments["seconds"])
    if lyrics:
        fields["lyrics"] = lyrics
    response = None
    try:
        response = await _run_music_workflow(AudioRunRequest(**fields), request, subject)
    except HTTPException as exc:
        detail = _detail(exc)
        if any(marker in detail for marker in MULTI_MODEL_REFUSAL_MARKERS):
            return detail, True
        return "Error: " + detail, False
    finally:
        end_in_process_request(request.scope, answered = response is not None)
    clips = [getattr(clip, "id", None) for clip in (getattr(response, "clips", None) or [])]
    source = audio_gallery.audio_path(clips[0]) if clips and clips[0] else None
    if source is None or not os.path.isfile(str(source)):
        return "Error: the music model returned no track.", False
    root = os.path.realpath(workdir)
    os.makedirs(root, exist_ok = True)
    file = f"generated-music-{str(clips[0])[:8]}{os.path.splitext(str(source))[1] or '.wav'}"
    shutil.copyfile(str(source), os.path.join(root, file))
    saved = {"name": file, "size": os.path.getsize(os.path.join(root, file))}
    return f"Made a track with {name} and saved it as {file}." + "\n__FILES__:" + json.dumps(
        [saved]
    ), False


def _music_ids() -> list[str]:
    try:
        from core.inference.audio_cpp_models import downloaded_models

        # The id names the model; repo_id is audio.cpp's umbrella repo, the same for every model.
        return [model.id for model in downloaded_models("music")]
    except Exception:  # noqa: BLE001 - no audio.cpp here means no music models
        return []


def _music_model(name: str) -> Optional[str]:
    wanted = name.strip().lower()
    try:
        from core.inference.audio_cpp_models import downloaded_models
        for model in downloaded_models("music"):
            if wanted == str(model.id).lower():
                return model.id
    except Exception:  # noqa: BLE001
        return None
    return None


def _decision_ids() -> list[str]:
    try:
        from core.systemone import catalog, laya_runtime
        from utils import systemone_settings

        if not systemone_settings.get_enabled():
            return []
        return [
            name for name in catalog.CHECKPOINTS if laya_runtime.is_cached(catalog.resolve(name))
        ]
    except Exception:  # noqa: BLE001 - no decision runtime here
        return []


def _settings_decision_model() -> Optional[str]:
    try:
        from utils import systemone_settings
        return _decision_model(systemone_settings.get_model())
    except Exception:  # noqa: BLE001 - no decision runtime here
        return None


def _decision_model(name: str) -> Optional[str]:
    return name if name in _decision_ids() else None
