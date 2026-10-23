# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""IP-Adapter image prompts: the family gate, the load / scale / unload lifecycle on a mocked pipe, the request
model, the route's engine gate and the recorded recipe."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from core.inference import diffusion_ip_adapter as ipa


class FakeDenoiser:
    def __init__(self):
        self.attn_processors = {"down.attn1": object(), "down.attn2": object()}
        self.set_calls = []

    def set_attn_processor(self, procs):
        self.set_calls.append(procs)
        self.attn_processors = dict(procs)


class FakePipe:
    def __init__(self, fail_load = False):
        self.unet = FakeDenoiser()
        self.image_encoder = None
        self.loads = []
        self.scales = []
        self.unloads = 0
        self.fail_load = fail_load

    def load_ip_adapter(self, repo, **kwargs):
        self.loads.append((repo, kwargs))
        self.unet.attn_processors = {k: ("ip", k) for k in self.unet.attn_processors}
        if self.fail_load:
            raise OSError("download failed")
        self.image_encoder = SimpleNamespace(
            to = lambda device: setattr(self, "encoder_device", device)
        )

    def set_ip_adapter_scale(self, scale):
        self.scales.append(scale)

    def unload_ip_adapter(self):
        self.unloads += 1
        self.image_encoder = None
        self.unet.attn_processors = {k: "default" for k in self.unet.attn_processors}


SDXL = ipa.adapter_for("sdxl")


def _apply(pipe, scale):
    return ipa.apply(pipe, SDXL, scale, denoiser_attr = "unet", device = "cuda", cache_dir = "/c")


def test_only_sdxl_has_an_adapter():
    assert SDXL is not None and SDXL.repo_id == "h94/IP-Adapter"
    assert SDXL.weight_name == "ip-adapter_sdxl_vit-h.safetensors"
    assert ipa.adapter_for(" SDXL ") is SDXL
    for family in ("flux.1", "flux.2", "qwen-image", "z-image", None, ""):
        assert ipa.adapter_for(family) is None
    assert ipa.adapter_repos() == ["h94/IP-Adapter"]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, True),
        ({"engine": "sd_cpp"}, False),
        ({"family": "flux.1"}, False),
        ({"model_kind": "gguf"}, False),
        ({"compiled": True}, False),
        ({"transformer_quant": "int8"}, False),
        ({"transformer_quant": "bnb-4bit"}, True),
    ],
)
def test_supports_ip_adapter_matrix(kwargs, expected):
    args = dict(
        engine = "diffusers",
        family = "sdxl",
        model_kind = "pipeline",
        transformer_quant = None,
        compiled = False,
    )
    args.update(kwargs)
    assert ipa.supports_ip_adapter(**args) is expected


def test_loads_once_then_only_rescales():
    pipe = FakePipe()
    assert _apply(pipe, 0.6) is True
    assert _apply(pipe, 0.3) is False
    assert len(pipe.loads) == 1
    repo, kwargs = pipe.loads[0]
    assert repo == "h94/IP-Adapter"
    assert kwargs["subfolder"] == "sdxl_models"
    assert kwargs["image_encoder_folder"] == "models/image_encoder"
    assert kwargs["cache_dir"] == "/c"
    assert pipe.scales == [0.6, 0.3]
    assert pipe.encoder_device == "cuda"
    assert ipa.loaded_id(pipe) == SDXL.id


def test_clear_unloads_and_restores_the_exact_processors():
    pipe = FakePipe()
    before = dict(pipe.unet.attn_processors)
    assert ipa.clear(pipe, denoiser_attr = "unet") is False
    _apply(pipe, 0.6)
    assert ipa.clear(pipe, denoiser_attr = "unet") is True
    assert pipe.unloads == 1 and pipe.image_encoder is None
    assert pipe.unet.attn_processors == before
    assert all(a is b for a, b in zip(pipe.unet.attn_processors.values(), before.values()))
    assert ipa.loaded_id(pipe) is None
    # A second clear is a no-op, and the next request loads again.
    assert ipa.clear(pipe, denoiser_attr = "unet") is False
    assert _apply(pipe, 0.5) is True and len(pipe.loads) == 2


def test_a_failed_load_leaves_the_pipe_as_it_was():
    pipe = FakePipe(fail_load = True)
    before = dict(pipe.unet.attn_processors)
    with pytest.raises(OSError):
        _apply(pipe, 0.6)
    assert ipa.loaded_id(pipe) is None
    assert pipe.unet.attn_processors == before
    assert pipe.scales == []


def test_request_model_bounds():
    from pydantic import ValidationError

    from models.inference import DiffusionGenerateRequest

    req = DiffusionGenerateRequest(prompt = "p", ip_adapter = {"image": "data:image/png;base64,AAAA"})
    assert req.ip_adapter.scale == 0.6
    assert DiffusionGenerateRequest(prompt = "p").ip_adapter is None
    for bad in (
        {"image": "x", "scale": 1.5},
        {"image": "x", "scale": -0.1},
        {"image": ""},
        {"scale": 0.5},
    ):
        with pytest.raises(ValidationError):
            DiffusionGenerateRequest(prompt = "p", ip_adapter = bad)


def test_hires_refuses_an_image_prompt():
    from core.inference.diffusion_hires import _check_request
    _check_request({"ip_adapter": ("img", 0.0)})
    with pytest.raises(ValueError, match = "image prompt"):
        _check_request({"ip_adapter": ("img", 0.6)})


def test_managed_accounts_authorize_the_adapter_repo():
    from hub.services.models import account_access
    from models.inference import DiffusionGenerateRequest

    req = DiffusionGenerateRequest(prompt = "p", ip_adapter = {"image": "x", "scale": 0.6})
    assert account_access.media_adapter_references(req) == ["h94/IP-Adapter"]
    off = DiffusionGenerateRequest(prompt = "p", ip_adapter = {"image": "x", "scale": 0})
    assert account_access.media_adapter_references(off) == []


def test_route_passes_the_image_prompt_only_to_an_engine_that_takes_it():
    from models.inference import DiffusionGenerateRequest
    from routes.inference import _ip_adapter_kwargs

    def diffusers_generate(*, prompt, ip_adapter = None): ...

    def native_generate(*, prompt): ...

    req = DiffusionGenerateRequest(prompt = "p", ip_adapter = {"image": "img", "scale": 0.4})
    assert _ip_adapter_kwargs(diffusers_generate, req) == {"ip_adapter": ("img", 0.4)}
    with pytest.raises(ValueError, match = "diffusers engine only"):
        _ip_adapter_kwargs(native_generate, req)
    off = DiffusionGenerateRequest(prompt = "p", ip_adapter = {"image": "img", "scale": 0})
    assert _ip_adapter_kwargs(native_generate, off) == {}
    assert _ip_adapter_kwargs(native_generate, DiffusionGenerateRequest(prompt = "p")) == {}


def get_family(name):
    from core.inference.diffusion_families import _FAMILIES
    return next(f for f in _FAMILIES if f.name == name)


def _prepare_state(**overrides):
    state = SimpleNamespace(
        family = get_family("sdxl"),
        kind = "pipeline",
        transformer_quant = None,
        speed_optims = (),
        offload_policy = "none",
        calibrated_placement = False,
        pipe = FakePipe(),
        device = "cpu",
        hf_token = None,
        cuda_graphs = (),
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _png_b64():
    import base64
    import io

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize(
    "overrides, workflow, message",
    [
        ({"family": None}, "txt2img", "not supported for the 'FLUX.1"),
        ({}, "img2img", "plain text-to-image only, not the img2img"),
        ({}, "controlnet", "plain text-to-image only, not the controlnet"),
        ({"speed_optims": ("compiled",)}, "txt2img", "Speed off"),
        ({"transformer_quant": "fp8"}, "txt2img", "Speed off"),
        ({"offload_policy": "group"}, "txt2img", "fully on the GPU"),
        ({"calibrated_placement": True}, "txt2img", "fully on the GPU"),
    ],
)
def test_prepare_refuses_before_any_download(overrides, workflow, message):
    from core.inference.diffusion import DiffusionBackend

    if overrides.get("family", "x") is None:
        overrides = {"family": get_family("flux.1")}
    state = _prepare_state(**overrides)
    with pytest.raises(ValueError, match = message):
        DiffusionBackend()._ip_adapter_prepare(
            state, workflow, (_png_b64(), 0.6), threading.Event()
        )
    assert state.pipe.loads == []


def test_prepare_refuses_a_bad_image_before_loading():
    from core.inference.diffusion import DiffusionBackend

    state = _prepare_state()
    with pytest.raises(ValueError):
        DiffusionBackend()._ip_adapter_prepare(
            state, "txt2img", ("not-an-image", 0.6), threading.Event()
        )
    assert state.pipe.loads == []


def test_prepare_refuses_when_the_card_has_no_room(monkeypatch):
    from core.inference import diffusion
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(diffusion, "_free_device_mib", lambda device: 1024)
    state = _prepare_state()
    with pytest.raises(ValueError, match = "Not enough free GPU memory"):
        DiffusionBackend()._ip_adapter_prepare(
            state, "txt2img", (_png_b64(), 0.6), threading.Event()
        )
    assert state.pipe.loads == []


def test_prepare_loads_and_returns_the_decoded_image(monkeypatch):
    from core.inference import diffusion
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(diffusion, "_free_device_mib", lambda device: 64 * 1024)
    state = _prepare_state()
    entry, image = DiffusionBackend()._ip_adapter_prepare(
        state, "txt2img", (_png_b64(), 0.6), threading.Event()
    )
    assert entry is SDXL and image.size == (32, 32) and image.mode == "RGB"
    assert len(state.pipe.loads) == 1 and state.pipe.scales == [0.6]
    # Loaded: no second memory check or load on the next request.
    monkeypatch.setattr(diffusion, "_free_device_mib", lambda device: 0)
    DiffusionBackend()._ip_adapter_prepare(state, "txt2img", (_png_b64(), 0.2), threading.Event())
    assert len(state.pipe.loads) == 1 and state.pipe.scales == [0.6, 0.2]


class _Engine:
    """A diffusers-shaped engine: takes ip_adapter, reports what it applied."""

    def __init__(self):
        self.calls = []

    def status(self):
        return {
            "loaded": True,
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "family": "sdxl",
        }

    def generate(
        self,
        *,
        ip_adapter = None,
        **kwargs,
    ):
        self.calls.append(dict(kwargs, ip_adapter = ip_adapter))
        applied = ip_adapter is not None and ip_adapter[1] > 0
        return {
            "images": [Image.new("RGB", (64, 64), "blue")],
            "seed": 7,
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "workflow": "txt2img",
            "ip_adapter": SDXL.id if applied else None,
            "ip_adapter_scale": ip_adapter[1] if applied else None,
        }


def _run(monkeypatch, engine, request):
    from core.inference import diffusion_engine_router
    from routes import inference

    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: engine)
    return asyncio.run(inference._generate_diffusion_image_core(request))


def test_route_records_the_image_prompt_in_the_gallery_and_png(monkeypatch):
    from core.inference import image_gallery
    from models.inference import DiffusionGenerateRequest

    engine = _Engine()
    req = DiffusionGenerateRequest(
        prompt = "a lighthouse",
        steps = 4,
        guidance = 5.0,
        seed = 7,
        ip_adapter = {"image": _png_b64(), "scale": 0.6},
    )
    res = _run(monkeypatch, engine, req)
    assert engine.calls[0]["ip_adapter"] == (req.ip_adapter.image, 0.6)
    record = res.images[0]
    assert record.ip_adapter == SDXL.id and record.ip_adapter_scale == 0.6
    png = Image.open(image_gallery.image_path(record.id))
    meta = json.loads(png.text["unsloth"])
    assert meta["ip_adapter"] == SDXL.id and meta["ip_adapter_scale"] == 0.6
    # The image prompt itself is not persisted, like every other conditioning image.
    assert req.ip_adapter.image not in png.text["unsloth"]

    plain = _run(
        monkeypatch, engine, DiffusionGenerateRequest(prompt = "a lighthouse", steps = 4, guidance = 5.0)
    )
    assert engine.calls[1]["ip_adapter"] is None
    assert plain.images[0].ip_adapter is None and plain.images[0].ip_adapter_scale is None


def test_route_refuses_an_image_prompt_on_an_engine_without_it(monkeypatch):
    from models.inference import DiffusionGenerateRequest

    class Native(_Engine):
        def generate(self, **kwargs):
            raise AssertionError("must be refused before the engine runs")

    req = DiffusionGenerateRequest(
        prompt = "p", steps = 4, guidance = 5.0, ip_adapter = {"image": _png_b64(), "scale": 0.6}
    )
    with pytest.raises(HTTPException) as exc:
        _run(monkeypatch, Native(), req)
    assert exc.value.status_code == 400 and "diffusers engine only" in exc.value.detail
