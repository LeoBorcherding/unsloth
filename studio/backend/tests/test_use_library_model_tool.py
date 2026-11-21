# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""use_library_model swaps a downloaded image model in for one image and always puts the chat model back."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.inference import (
    ensure_model,
    generate_image_tool,
    library_model_tool as tool,
    llama_keepwarm,
    media_chat_tools,
    media_model_index,
    media_switch_backends,
)
from core.inference.gpu_arbiter import DIFFUSION, VIDEO
from core.inference.media_model_index import IMAGE_TASK, VIDEO_TASK
from routes import inference
from utils import openai_auto_switch_settings as auto_switch
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
PICK = media_model_index.MediaModelPick("org/image-model", "/models/org/image-model")
CHAT = ("/models/chat.gguf", "Q4_K_M", "org/chat-GGUF")
STATUS = {
    "loaded": True,
    "repo_id": "org/image-model",
    "family": "z-image",
    "workflows": ["txt2img"],
}


class Backend:
    def __init__(self, resident = False):
        self.resident = resident

    def status(self):
        return {"loaded": self.resident, "repo_id": "org/image-model" if self.resident else None}

    def loading_repo_ids(self):
        return []


@pytest.fixture
def server_loop(monkeypatch):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target = loop.run_forever, daemon = True)
    thread.start()
    monkeypatch.setattr(media_chat_tools, "_server_loop", loop)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


@pytest.fixture
def world(monkeypatch, server_loop, tmp_path):
    """A GGUF chat model loaded, one downloaded image model, nothing image resident."""
    state = SimpleNamespace(
        calls = [],
        backend = Backend(),
        chat = CHAT,
        generate_error = None,
        generate_block = None,
        restore_failure = None,
        fits_beside = False,
        mlx_chat = None,
    )

    monkeypatch.setattr(
        media_model_index,
        "resolve_local_media_model",
        lambda name, task: PICK if name == "org/image-model" and task == IMAGE_TASK else None,
    )
    monkeypatch.setattr(
        media_model_index,
        "available_media_model_ids",
        lambda task: ["org/image-model"] if task == IMAGE_TASK else [],
    )
    monkeypatch.setattr(
        media_model_index, "satisfied_by", lambda status, name, pick: bool(status.get("loaded"))
    )
    monkeypatch.setattr(media_switch_backends, "backend_for", lambda owner: state.backend)
    monkeypatch.setattr(llama_keepwarm, "loaded_identity", lambda: state.chat)
    monkeypatch.setattr(llama_keepwarm, "launch_to_replay", lambda backend = None: {"kwargs": {}})
    monkeypatch.setattr(ensure_model, "media_load_keeps_chat", lambda owner: state.fits_beside)
    monkeypatch.setattr(
        media_chat_tools,
        "media_chat_status",
        lambda task: STATUS if state.backend.resident else None,
    )

    async def load_image(owner, name, pick, subject, deadline, abort):
        state.calls.append(("load", name, subject))
        state.chat = None if not state.fits_beside else state.chat
        state.backend.resident = True

    async def unload_image(owner, subject):
        state.calls.append(("unload",))
        state.backend.resident = False

    async def restore_chat(swap, subject):
        state.calls.append(("restore", swap.chat_identity))
        if state.restore_failure:
            return state.restore_failure
        if swap.chat_identity is not None:
            state.chat = swap.chat_identity
        return None

    async def generate(request, ticket):
        state.calls.append(("generate", request.prompt))
        if state.generate_block is not None:
            while not ticket.cancelled():
                await asyncio.sleep(0.02)
            raise HTTPException(status_code = 409, detail = "Image generation was cancelled.")
        if state.generate_error is not None:
            raise state.generate_error
        return SimpleNamespace(
            images = [SimpleNamespace(id = "abcdef0123", width = 1024, height = 1024, seed = 7)]
        )

    async def no_row(name):
        return None

    async def no_llms():
        return ()

    monkeypatch.setattr(tool, "_catalog_row", no_row)
    monkeypatch.setattr(tool, "_downloaded_llms", no_llms)
    monkeypatch.setattr(tool, "_music_model", lambda name: None)
    monkeypatch.setattr(tool, "_decision_model", lambda name: None)
    monkeypatch.setattr(tool, "_load_media", load_image)
    monkeypatch.setattr(tool, "_unload_media", unload_image)
    monkeypatch.setattr(tool, "_restore_chat", restore_chat)
    monkeypatch.setattr(generate_image_tool, "_generate", generate)
    monkeypatch.setattr(
        generate_image_tool,
        "_save_to_workdir",
        lambda image, workdir, account: {"name": f"generated-image-{image.id[:8]}.png", "size": 3},
    )
    state.workdir = str(tmp_path / "work")
    return state


def _call(world, arguments, **kwargs):
    out = {}
    thread = threading.Thread(
        target = lambda: out.update(
            r = run_as(OWNER, tool.run_use_library_model, arguments, workdir = world.workdir, **kwargs)
        )
    )
    thread.start()
    thread.join(60)
    return out["r"]


def test_full_swap_unloads_chat_renders_and_puts_the_chat_model_back(world):
    result = _call(world, {"model": "org/image-model", "prompt": "a red fox"})
    assert result.startswith("Generated a 1024x1024 image with org/image-model (seed 7)")
    assert '__IMAGES__:["generated-image-abcdef01.png"]' in result
    assert [c[0] for c in world.calls] == ["load", "generate", "unload", "restore"]
    assert world.calls[-1] == ("restore", CHAT)
    assert world.chat == CHAT and not world.backend.resident


def test_a_resident_image_model_is_used_without_any_swap(world):
    world.backend.resident = True
    result = _call(world, {"model": "org/image-model", "prompt": "a red fox"})
    assert result.startswith("Generated a")
    assert [c[0] for c in world.calls] == ["generate", "restore"]
    # Nothing was evicted, so the restore has no identity to bring back.
    assert world.calls[-1] == ("restore", None)
    assert world.backend.resident


def test_a_model_that_fits_beside_the_chat_is_unloaded_but_the_chat_is_untouched(world):
    world.fits_beside = True
    _call(world, {"model": "org/image-model", "prompt": "a red fox"})
    assert [c[0] for c in world.calls] == ["load", "generate", "unload", "restore"]
    assert world.chat == CHAT


def test_no_model_lists_the_downloaded_ones(world):
    result = _call(world, {"prompt": "a red fox"})
    assert result.startswith(
        "No model was given, so nothing was made. Downloaded image models: org/image-model."
    )
    assert world.calls == []


def test_an_unknown_model_is_refused_with_the_list(world):
    result = _call(world, {"model": "org/nope", "prompt": "a red fox"})
    assert result.startswith("Error: no downloaded model matches 'org/nope'.")
    assert "org/image-model" in result
    assert world.calls == []


def test_a_model_without_a_prompt_is_refused(world):
    assert _call(world, {"model": "org/image-model"}) == tool.ERR_NO_PROMPT


def test_a_failed_render_still_puts_the_chat_model_back(world):
    world.generate_error = HTTPException(status_code = 500, detail = "out of memory")
    result = _call(world, {"model": "org/image-model", "prompt": "a red fox"})
    assert result == "Error: out of memory"
    assert [c[0] for c in world.calls] == ["load", "generate", "unload", "restore"]
    assert world.chat == CHAT


def test_a_stop_mid_render_still_puts_the_chat_model_back(world):
    world.generate_block = True
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    result = _call(world, {"model": "org/image-model", "prompt": "a red fox"}, cancel_event = cancel)
    assert result == tool.ERR_STOPPED
    assert [c[0] for c in world.calls] == ["load", "generate", "unload", "restore"]
    assert world.chat == CHAT


def test_a_failed_restore_tells_the_model(world):
    world.restore_failure = "it did not come back up"
    result = _call(world, {"model": "org/image-model", "prompt": "a red fox"})
    assert result.startswith("Generated a")
    assert "The chat model could not be loaded back (it did not come back up)" in result


def test_a_non_gguf_chat_that_would_be_evicted_is_refused_before_anything_loads(world, monkeypatch):
    import core.inference as core_inference

    world.chat = None
    monkeypatch.setattr(
        core_inference, "get_inference_backend", lambda: SimpleNamespace(active_model_name = "mlx/x")
    )
    result = _call(world, {"model": "org/image-model", "prompt": "a red fox"})
    assert result == tool.ERR_NON_GGUF_CHAT
    assert world.calls == []


def test_restore_chat_is_a_no_op_when_nothing_was_evicted_or_the_chat_is_back(monkeypatch):
    swap = tool._Swap()
    assert asyncio.run(tool._restore_chat(swap, "owner")) is None
    swap.chat_identity = CHAT
    monkeypatch.setattr(llama_keepwarm, "loaded_identity", lambda: CHAT)
    assert asyncio.run(tool._restore_chat(swap, "owner")) is None


def _select(enabled):
    payload = SimpleNamespace(enabled_tools = enabled, rag_scope = None, bypass_permissions = False)

    async def go():
        return await inference._select_request_tools(
            payload, tools_on = True, mcp_allowed = False, supports_vision = True
        )

    return {t["function"]["name"] for t in asyncio.run(go())}


def test_offered_only_to_the_owner_with_switch_model_by_request_on(monkeypatch):
    monkeypatch.setattr(inference, "_enabled_agent_skills", lambda: [])
    monkeypatch.setattr(inference, "_thread_has_conversation_archive", lambda _: False)
    monkeypatch.setattr(media_chat_tools, "media_chat_status", lambda task: None)
    monkeypatch.setattr(auto_switch, "get_media_auto_switch_enabled", lambda: False)
    assert "use_library_model" not in _select(["use_library_model"])
    monkeypatch.setattr(auto_switch, "get_media_auto_switch_enabled", lambda: True)
    assert "use_library_model" in _select(["use_library_model"])
    assert "use_library_model" not in _select(["web_search"])
    assert "use_library_model" not in run_as(ALICE, _select, ["use_library_model"])


@pytest.fixture
def modes(world, monkeypatch):
    """Records the Keep multiple models loaded mode each image load ran under."""
    from utils import multi_model_settings

    monkeypatch.setattr(multi_model_settings, "_read_mode", lambda: "setting")
    multi_model_settings.invalidate_multi_model_cache()
    seen = []
    load = tool._load_media

    async def load_image(owner, name, pick, subject, deadline, abort):
        seen.append(multi_model_settings.get_multi_model_mode())
        await load(owner, name, pick, subject, deadline, abort)

    monkeypatch.setattr(tool, "_load_media", load_image)
    yield seen
    multi_model_settings.invalidate_multi_model_cache()


@pytest.mark.parametrize(
    "value, mode",
    [
        ("true", "off"),
        (True, "off"),
        ("auto", "fit"),
        ("AUTO", "fit"),
        ("false", "always"),
        (False, "always"),
    ],
)
def test_unload_chat_model_runs_the_load_under_the_matching_mode(world, modes, value, mode):
    world.fits_beside = mode != "off"
    _call(world, {"model": "org/image-model", "prompt": "a fox", "unload_chat_model": value})
    assert modes == [mode]


def test_omitted_unload_chat_model_follows_the_setting(world, modes):
    _call(world, {"model": "org/image-model", "prompt": "a fox"})
    assert modes == ["setting"]


def test_the_override_ends_with_the_call(world, modes):
    from utils.multi_model_settings import get_multi_model_mode
    _call(world, {"model": "org/image-model", "prompt": "a fox", "unload_chat_model": "true"})
    assert get_multi_model_mode() == "setting"


def test_an_invalid_unload_chat_model_is_refused_before_anything_loads(world):
    result = _call(
        world, {"model": "org/image-model", "prompt": "a fox", "unload_chat_model": "maybe"}
    )
    assert result == tool.ERR_UNLOAD_ARG
    assert world.calls == []


def test_unload_true_says_the_chat_model_was_unloaded_and_put_back(world):
    result = _call(
        world, {"model": "org/image-model", "prompt": "a fox", "unload_chat_model": "true"}
    )
    assert "The chat model was unloaded for this image and loaded back." in result
    assert world.chat == CHAT


def test_keep_that_fits_leaves_the_chat_model_loaded_throughout(world):
    world.fits_beside = True
    result = _call(
        world, {"model": "org/image-model", "prompt": "a fox", "unload_chat_model": "false"}
    )
    assert "The chat model stayed loaded beside the image model." in result
    assert [c[0] for c in world.calls] == ["load", "generate", "unload", "restore"]
    assert world.chat == CHAT


def test_keep_that_does_not_fit_is_refused_and_loads_nothing(world):
    result = _call(
        world, {"model": "org/image-model", "prompt": "a fox", "unload_chat_model": "false"}
    )
    assert result.startswith("Error: 'org/image-model' does not fit beside the chat model")
    assert [c[0] for c in world.calls] == ["restore"]
    assert world.chat == CHAT and not world.backend.resident


def test_the_schema_offers_the_three_choices():
    prop = tool.use_library_model_schema()["function"]["parameters"]["properties"][
        "unload_chat_model"
    ]
    assert prop["enum"] == ["auto", "true", "false"]


def _preflight_world(monkeypatch):
    backend = Backend()
    monkeypatch.setattr(media_switch_backends, "backend_for", lambda owner: backend)
    monkeypatch.setattr(
        media_model_index, "satisfied_by", lambda status, name, pick: bool(status.get("loaded"))
    )
    monkeypatch.setattr(tool, "_LOAD_POLL_S", 0.01)
    return backend


def test_wait_outlasts_a_setup_still_in_preflight(monkeypatch):
    backend = _preflight_world(monkeypatch)

    async def scenario():
        lock = tool._switch_lock(DIFFUSION)
        await lock.acquire()

        async def setup():
            await asyncio.sleep(0.1)
            backend.resident = True
            lock.release()

        task = asyncio.ensure_future(setup())
        await tool._await_loaded(DIFFUSION, "org/image-model", PICK, 1e18, threading.Event())
        await task

    asyncio.run(scenario())
    assert backend.resident


def test_wait_reports_a_load_that_stopped(monkeypatch):
    _preflight_world(monkeypatch)
    with pytest.raises(RuntimeError, match = "stopped loading"):
        asyncio.run(tool._await_loaded(DIFFUSION, "org/image-model", PICK, 1e18, threading.Event()))


def test_put_back_waits_for_the_setup_before_restoring(monkeypatch):
    _preflight_world(monkeypatch)
    calls = []

    async def unload_image(owner, subject):
        calls.append("unload")

    async def restore_chat(swap, subject):
        calls.append("restore")

    monkeypatch.setattr(tool, "_unload_media", unload_image)
    monkeypatch.setattr(tool, "_restore_chat", restore_chat)

    async def scenario():
        lock = tool._switch_lock(DIFFUSION)
        await lock.acquire()
        swap = tool._Swap()
        swap.loaded_media = True
        put_back = asyncio.ensure_future(tool._put_back(swap, "alice"))
        await asyncio.sleep(0.1)
        assert calls == []
        lock.release()
        await put_back

    asyncio.run(scenario())
    assert calls == ["unload", "restore"]


def test_a_slow_switch_is_waited_out(monkeypatch):
    from core.inference import media_auto_switch
    from core.inference.media_switch_errors import slow_switch

    seen = {}

    async def switch(name, *, owner, current_subject, openai_errors):
        raise slow_switch("image", openai_errors)

    async def wait(owner, name, pick, deadline, abort):
        seen["waited"] = True

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", switch)
    monkeypatch.setattr(tool, "_await_loaded", wait)
    asyncio.run(
        tool._load_media(DIFFUSION, "org/image-model", PICK, "alice", 1e18, threading.Event())
    )
    assert seen == {"waited": True}


@pytest.mark.parametrize(
    "unload, expected", [("false", "does not fit beside"), (None, "set to Always")]
)
def test_always_refusal_names_the_argument_only_when_it_set_always(
    world, monkeypatch, unload, expected
):
    from core.inference.media_switch_errors import refuse

    async def refused(owner, name, pick, subject, deadline, abort):
        exc = refuse(
            "This model needs about 22.9 GB. Keep multiple models loaded is set to Always.",
            status_code = 409,
            openai_errors = True,
            code = "multi_model_always",
        )
        raise exc

    monkeypatch.setattr(tool, "_load_media", refused)
    world.fits_beside = True
    arguments = {"model": "org/image-model", "prompt": "a fox"}
    if unload is not None:
        arguments["unload_chat_model"] = unload
    assert expected in _call(world, arguments)


def test_no_model_and_no_prompt_is_a_plain_listing(world):
    assert _call(world, {}).startswith("Downloaded image models: org/image-model.")


def test_a_refused_switch_leaves_the_resident_image_model_loaded(world, monkeypatch):
    from core.inference.media_switch_errors import refuse

    class Other:
        resident = True

        def status(self):
            return {"loaded": True, "repo_id": "org/other-model"}

        def loading_repo_ids(self):
            return []

    world.backend = Other()
    world.fits_beside = True
    monkeypatch.setattr(
        media_model_index,
        "satisfied_by",
        lambda status, name, pick: status.get("repo_id") == "org/image-model",
    )

    async def refused(owner, name, pick, subject, deadline, abort):
        raise refuse("no room", status_code = 409, openai_errors = True, code = "multi_model_always")

    monkeypatch.setattr(tool, "_load_media", refused)
    result = _call(
        world, {"model": "org/image-model", "prompt": "a fox", "unload_chat_model": "false"}
    )
    assert "does not fit beside" in result
    assert ("unload",) not in world.calls


VPICK = media_model_index.MediaModelPick("org/video-model", "/models/org/video-model")


@pytest.fixture
def video(world, monkeypatch):
    """org/video-model downloaded as a video model; records the owner each load and unload used."""
    from core.inference import generate_video_tool

    state = SimpleNamespace(owners = [], render = None)
    monkeypatch.setattr(
        media_model_index,
        "resolve_local_media_model",
        lambda name, task: {
            ("org/image-model", IMAGE_TASK): PICK,
            ("org/video-model", VIDEO_TASK): VPICK,
        }.get((name, task)),
    )
    monkeypatch.setattr(
        media_model_index,
        "available_media_model_ids",
        lambda task: {IMAGE_TASK: ["org/image-model"], VIDEO_TASK: ["org/video-model"]}[task],
    )
    load, unload = tool._load_media, tool._unload_media

    async def load_media(owner, *args):
        state.owners.append(("load", owner))
        await load(owner, *args)

    async def unload_media(owner, subject):
        state.owners.append(("unload", owner))
        await unload(owner, subject)

    def render(
        arguments,
        *,
        workdir,
        cancel_event = None,
        timeout = None,
        create_timeout = None,
    ):
        state.render = arguments
        state.create_timeout = (timeout, create_timeout)
        return (
            "Generated a video (5s, 832x480) saved as generated-video-abcd1234.mp4. Prompt: "
            + arguments["prompt"]
            + '\n__FILES__:[{"name": "generated-video-abcd1234.mp4", "size": 3}]'
        )

    monkeypatch.setattr(tool, "_load_media", load_media)
    monkeypatch.setattr(tool, "_unload_media", unload_media)
    monkeypatch.setattr(generate_video_tool, "run_generate_video", render)
    return state


def test_a_video_model_swaps_in_on_the_video_backend(world, video):
    result = _call(
        world, {"model": "org/video-model", "prompt": "waves at dusk", "seconds": 5, "seed": 3}
    )
    assert video.owners == [("load", VIDEO), ("unload", VIDEO)]
    assert video.render == {"prompt": "waves at dusk", "seconds": 5, "seed": 3}
    timeout, create_timeout = video.create_timeout
    assert create_timeout == timeout and create_timeout > 60
    text, files = result.split("\n__FILES__")
    assert text.startswith("Generated a video with org/video-model (5s, 832x480)")
    assert text.endswith("The chat model was unloaded for this video and loaded back.")
    assert "generated-video-abcd1234.mp4" in files
    assert world.chat == CHAT
    assert ("generate", "waves at dusk") not in world.calls


def test_a_video_model_honours_unload_chat_model_false(world, video):
    result = _call(
        world, {"model": "org/video-model", "prompt": "waves", "unload_chat_model": "false"}
    )
    assert result.startswith(tool.ERR_KEEP_NO_FIT.format(model = "org/video-model", kind = "video"))
    assert video.owners == []


def test_an_image_model_still_uses_the_image_backend(world, video):
    _call(world, {"model": "org/image-model", "prompt": "a fox"})
    assert video.owners == [("load", DIFFUSION), ("unload", DIFFUSION)]


def test_the_listing_shows_image_and_video_models(world, video):
    assert _call(world, {}) == (
        "Downloaded image models: org/image-model. Downloaded video models: org/video-model. "
        "Call use_library_model again with one as model."
    )


def test_another_gguf_of_the_same_repo_is_not_taken_for_the_prior_model(world, monkeypatch):
    class Repo:
        def __init__(self):
            self.file = "model-Q8_0.gguf"
            self.resident = True

        def status(self):
            return {"loaded": True, "repo_id": "org/image-model", "gguf_filename": self.file}

        def loading_repo_ids(self):
            return []

    world.backend = Repo()
    world.fits_beside = True
    monkeypatch.setattr(
        media_model_index,
        "satisfied_by",
        lambda status, name, pick: status.get("gguf_filename") == "model-Q4_K_M.gguf",
    )

    async def load_other_file(owner, name, pick, subject, deadline, abort):
        world.backend.file = "model-Q4_K_M.gguf"

    monkeypatch.setattr(tool, "_load_media", load_other_file)
    _call(world, {"model": "org/image-model", "prompt": "a fox"})
    assert ("unload",) in world.calls


def test_only_the_last_chat_on_a_shared_model_puts_things_back(monkeypatch):
    _preflight_world(monkeypatch)
    calls = []

    async def unload_media(owner, subject):
        calls.append("unload")

    async def restore_chat(swap, subject):
        calls.append(("restore", swap.chat_identity))

    monkeypatch.setattr(tool, "_unload_media", unload_media)
    monkeypatch.setattr(tool, "_restore_chat", restore_chat)

    async def scenario():
        loader, reader = tool._Swap(), tool._Swap()
        loader.loaded_media, loader.chat_identity, loader.chat_unloaded = True, CHAT, True
        tool._enter_shared(loader)
        tool._enter_shared(reader)
        # The chat that loaded the model finishes first: the other chat is still rendering on it.
        await tool._put_back(loader, "alice")
        assert calls == [] and loader.deferred
        await tool._put_back(reader, "bob")
        assert tool._SHARED == {}

    asyncio.run(scenario())
    assert calls == ["unload", ("restore", CHAT)]


def test_put_back_waits_for_another_render_on_the_backend(monkeypatch):
    backend = _preflight_world(monkeypatch)
    state = {"active": True}
    backend.generate_progress = lambda: {"active": state["active"]}
    calls = []

    async def unload_media(owner, subject):
        calls.append("unload")

    async def restore_chat(swap, subject):
        calls.append("restore")

    monkeypatch.setattr(tool, "_unload_media", unload_media)
    monkeypatch.setattr(tool, "_restore_chat", restore_chat)

    async def scenario():
        swap = tool._Swap()
        swap.loaded_media = True
        tool._enter_shared(swap)
        put_back = asyncio.ensure_future(tool._put_back(swap, "alice"))
        await asyncio.sleep(0.1)
        assert calls == []
        state["active"] = False
        await put_back

    asyncio.run(scenario())
    assert calls == ["unload", "restore"]


@pytest.fixture
def llm(world, monkeypatch):
    """org/other-llm is a downloaded chat model; records loads, evictions and unloads."""
    from core.scheduled_tasks import scheduler
    from utils import multi_model_settings

    state = SimpleNamespace(calls = [], fits = True, loaded = set(), setting = "off", modes = [])

    async def row(name):
        return (
            SimpleNamespace(id = "org/other-llm", task = "text-generation")
            if name == "org/other-llm"
            else None
        )

    async def llms():
        return (("chat", ["org/other-llm"]),)

    async def complete(name, prompt, subject, *, beside):
        mode = multi_model_settings.get_multi_model_mode()
        state.modes.append(mode)
        if beside and mode == "always" and not state.fits:
            state.calls.append(("refused", name))
            return "Keep multiple models loaded is set to Always", True
        state.calls.append(("ask", name, beside))
        state.loaded.add(name)
        return f"echo: {prompt}", False

    async def unload(name, subject):
        state.calls.append(("unload", name))
        state.loaded.discard(name)

    def evict():
        state.calls.append(("evict",))
        world.chat = None

    monkeypatch.setattr(tool, "_catalog_row", row)
    monkeypatch.setattr(tool, "_downloaded_llms", llms)
    monkeypatch.setattr(tool, "_complete_llm", complete)
    monkeypatch.setattr(tool, "_unload_llm", unload)
    monkeypatch.setattr(tool, "_evict_chat_model", evict)
    monkeypatch.setattr(scheduler, "_model_is_loaded", lambda name: name in state.loaded)
    monkeypatch.setattr(multi_model_settings, "_read_mode", lambda: state.setting)
    multi_model_settings.invalidate_multi_model_cache()
    yield state
    multi_model_settings.invalidate_multi_model_cache()


def test_an_llm_with_unload_true_runs_after_the_chat_model_leaves(world, llm):
    result = _call(world, {"model": "org/other-llm", "prompt": "hi", "unload_chat_model": "true"})
    assert llm.calls == [("evict",), ("ask", "org/other-llm", False), ("unload", "org/other-llm")]
    assert llm.modes == ["off"]
    assert result.startswith("org/other-llm answered:\necho: hi")
    assert result.endswith(
        "The chat model was unloaded so org/other-llm could run, and loaded back."
    )
    assert world.chat == CHAT
    assert ("restore", CHAT) in world.calls


def test_an_llm_with_unload_false_runs_beside_the_chat_model(world, llm):
    result = _call(world, {"model": "org/other-llm", "prompt": "hi", "unload_chat_model": "false"})
    assert llm.calls == [("ask", "org/other-llm", True), ("unload", "org/other-llm")]
    assert llm.modes == ["always"]
    assert result.endswith("org/other-llm ran beside the chat model, which stayed loaded.")
    assert world.chat == CHAT


def test_an_llm_that_does_not_fit_is_refused_under_false(world, llm):
    llm.fits = False
    result = _call(world, {"model": "org/other-llm", "prompt": "hi", "unload_chat_model": "false"})
    assert result.startswith(tool.ERR_KEEP_NO_FIT.format(model = "org/other-llm", kind = "model"))
    assert ("evict",) not in llm.calls
    assert world.chat == CHAT


def test_auto_falls_back_to_unloading_when_the_llm_does_not_fit(world, llm):
    llm.fits = False
    result = _call(world, {"model": "org/other-llm", "prompt": "hi", "unload_chat_model": "auto"})
    assert llm.calls == [
        ("refused", "org/other-llm"),
        ("evict",),
        ("ask", "org/other-llm", False),
        ("unload", "org/other-llm"),
    ]
    assert "loaded back" in result
    assert world.chat == CHAT


def test_omitted_follows_the_setting_for_an_llm(world, llm):
    llm.setting = "fit"
    result = _call(world, {"model": "org/other-llm", "prompt": "hi"})
    assert llm.calls[0] == ("ask", "org/other-llm", True)
    assert "stayed loaded" in result


def test_an_llm_already_loaded_is_asked_without_a_swap(world, llm):
    llm.loaded.add("org/other-llm")
    result = _call(world, {"model": "org/other-llm", "prompt": "hi", "unload_chat_model": "true"})
    assert llm.calls == [("ask", "org/other-llm", True)]
    assert result == "org/other-llm answered:\necho: hi"


def test_the_listing_includes_chat_models(world, llm):
    assert _call(world, {}).startswith(
        "Downloaded chat models: org/other-llm. Downloaded image models"
    )


def test_a_tts_model_speaks_beside_the_chat_model_and_keeps_the_file_envelope_last(
    world, monkeypatch
):
    from core.scheduled_tasks import scheduler

    async def row(name):
        return SimpleNamespace(id = "org/tts", task = "text-to-speech")

    spoken = []

    async def speak(name, text, subject, workdir, *, beside):
        spoken.append((name, text, beside))
        return (
            'Spoke the text with org/tts and saved it as a.wav.\n__FILES__:[{"name": "a.wav", "size": 1}]',
            False,
        )

    async def unload(name, subject):
        spoken.append(("unload", name))

    monkeypatch.setattr(tool, "_catalog_row", row)
    monkeypatch.setattr(tool, "_speak_tts", speak)
    monkeypatch.setattr(tool, "_unload_llm", unload)
    monkeypatch.setattr(scheduler, "_model_is_loaded", lambda name: False)
    result = _call(
        world, {"model": "org/tts", "prompt": "hello there", "unload_chat_model": "false"}
    )
    assert spoken == [("org/tts", "hello there", True), ("unload", "org/tts")]
    body, files = result.split("\n__FILES__")
    assert body.endswith("org/tts ran beside the chat model, which stayed loaded.")
    assert "a.wav" in files


def test_an_stt_model_transcribes_an_audio_file_from_the_chat_and_is_unloaded(
    world, monkeypatch, tmp_path
):
    from core.inference import stt_registry

    async def row(name):
        return SimpleNamespace(id = "org/asr", task = "automatic-speech-recognition")

    async def transcribe(
        raw,
        model,
        language,
        fast,
        request = None,
        **kwargs,
    ):
        loaded["model"] = model
        return {"text": f"heard {len(raw)} bytes"}

    loaded, unloaded = {}, []
    monkeypatch.setattr(tool, "_catalog_row", row)
    monkeypatch.setattr(inference, "_transcribe_audio_result", transcribe)
    monkeypatch.setattr(
        stt_registry, "resident", lambda: {"model": loaded.get("model")} if loaded else {}
    )
    monkeypatch.setattr(stt_registry, "unload", lambda **kw: unloaded.append(kw))
    import os

    os.makedirs(world.workdir, exist_ok = True)
    with open(os.path.join(world.workdir, "clip.wav"), "wb") as f:
        f.write(b"12345")
    result = _call(world, {"model": "org/asr", "audio": "clip.wav"})
    assert result == "org/asr transcribed clip.wav:\nheard 5 bytes"
    assert unloaded == [{"expected_model": "org/asr"}]
    assert world.calls == []


def test_an_stt_model_refuses_a_path_outside_the_chat(world, monkeypatch):
    async def row(name):
        return SimpleNamespace(id = "org/asr", task = "automatic-speech-recognition")

    monkeypatch.setattr(tool, "_catalog_row", row)
    result = _call(world, {"model": "org/asr", "audio": "../../etc/passwd"})
    assert result == "Error: no audio file named '../../etc/passwd' in this chat."


def test_a_music_model_uses_the_chat_seat_and_returns_its_track(world, monkeypatch):
    from core.scheduled_tasks import scheduler

    made = []

    async def compose(name, prompt, arguments, subject, workdir, *, beside):
        made.append((name, prompt, arguments.get("seconds"), beside))
        return 'Made a track with org/music and saved it as t.wav.\n__FILES__:[{"name": "t.wav", "size": 1}]', False

    async def unload(name, subject):
        made.append(("unload", name))

    def evict():
        made.append(("evict",))
        world.chat = None

    monkeypatch.setattr(tool, "_music_model", lambda name: "org/music" if name == "org/music" else None)
    monkeypatch.setattr(tool, "_compose_music", compose)
    monkeypatch.setattr(tool, "_unload_llm", unload)
    monkeypatch.setattr(tool, "_evict_chat_model", evict)
    monkeypatch.setattr(scheduler, "_model_is_loaded", lambda name: False)
    result = _call(world, {"model": "org/music", "prompt": "lofi beat", "seconds": 20, "unload_chat_model": "true"})
    assert made == [("evict",), ("org/music", "lofi beat", 20, False), ("unload", "org/music")]
    body, files = result.split("\n__FILES__")
    assert body.endswith("The chat model was unloaded so org/music could run, and loaded back.")
    assert world.chat == CHAT


def test_a_decision_model_unloads_the_chat_model_unless_told_to_keep_it(world, monkeypatch):
    from core.systemone import catalog, laya_runtime
    from routes import systemone

    seen = []

    async def decide(checkpoint, state, questions):
        seen.append((state, sorted(questions)))
        return {"urgent": {"p": 0.9}, "_backend": "x"}

    def evict():
        seen.append("evict")
        world.chat = None

    monkeypatch.setattr(tool, "_decision_model", lambda name: "laya" if name == "laya" else None)
    monkeypatch.setattr(tool, "_evict_chat_model", evict)
    monkeypatch.setattr(systemone, "_decide", decide)
    monkeypatch.setattr(systemone, "_require_enabled", lambda: None)
    monkeypatch.setattr(catalog, "resolve", lambda name: object())
    monkeypatch.setattr(laya_runtime, "unload", lambda: seen.append("unload"))
    questions = {"urgent": {"type": "noul", "instructions": "Reply today?"}}

    kept = _call(world, {"model": "laya", "prompt": "an email", "questions": questions, "unload_chat_model": "false"})
    assert kept == tool.ERR_KEEP_NO_FIT.format(model = "laya", kind = "model")
    assert seen == []

    result = _call(world, {"model": "laya", "prompt": "an email", "questions": questions, "unload_chat_model": "auto"})
    assert seen == ["evict", ("an email", ["urgent"]), "unload"]
    assert result.startswith('laya decided:\n{"urgent": {"p": 0.9}}')
    assert "loaded back" in result
    assert world.chat == CHAT
