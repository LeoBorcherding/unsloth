# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Images page sampler on the native sd.cpp engine: status options and --sampling-method on both transports."""

from __future__ import annotations

import dataclasses

import pytest

from core.inference.diffusion_sampler import SD_CPP_SAMPLERS, sd_cpp_sampling_method

from .test_sd_cpp_backend import _FakeEngine, _FakeServer, _loaded_backend


_REPOS = {"qwen-image": "unsloth/Qwen-Image-GGUF", "z-image": "unsloth/Z-Image-Turbo-GGUF"}


def _backend(fam_name, engine = None):
    b = _loaded_backend(fam_name = fam_name, engine = engine)
    b._state = dataclasses.replace(b._state, repo_id = _REPOS[fam_name])
    return b


def _oneshot_params(sampler = None, fam_name = "qwen-image", **kw):
    engine = _FakeEngine()
    b = _backend(fam_name, engine)
    out = b.generate(prompt = "p", steps = 4, guidance = 4.0, seed = 1, sampler = sampler, **kw)
    return engine.calls[-1][1], out


def _server_sample_params(sampler = None, fam_name = "qwen-image"):
    server = _FakeServer("sd-server")
    b = _backend(fam_name)
    b._state = dataclasses.replace(b._state, mode = "server", server = server)
    b.generate(prompt = "p", steps = 4, guidance = 4.0, seed = 1, sampler = sampler)
    return server.payloads[-1].get("sample_params", {})


def test_status_offers_the_native_samplers():
    samplers = _backend("qwen-image").status()["samplers"]
    assert [o["id"] for o in samplers] == ["default", "euler", "euler_a", "heun", "dpmpp_2m"]
    assert all(o["schedule_types"] == ["automatic"] for o in samplers)


def test_a_few_step_load_withholds_dpmpp_2m():
    # Z-Image-Turbo's recipe is 8 steps; sd.cpp's DPM++ 2M renders noise there.
    assert "dpmpp_2m" not in [o["id"] for o in _backend("z-image").status()["samplers"]]
    with pytest.raises(ValueError, match = "DPM\\+\\+ 2M sampler is not available"):
        _oneshot_params("dpmpp_2m", fam_name = "z-image")
    assert sd_cpp_sampling_method("dpmpp_2m", None, "euler", None) == "dpm++2m"


@pytest.mark.parametrize(
    "sampler, method",
    [(None, "euler"), ("default", "euler"), ("heun", "heun"), ("euler_a", "euler_a"), ("dpmpp_2m", "dpm++2m")],
)
def test_oneshot_argv_takes_the_request_sampler(sampler, method):
    params, out = _oneshot_params(sampler)
    assert params.sampling_method == method
    assert out["sampler"] == (None if sampler in (None, "default") else sampler)


@pytest.mark.parametrize("sampler, method", [(None, "euler"), ("heun", "heun"), ("dpmpp_2m", "dpm++2m")])
def test_server_body_takes_the_request_sampler(sampler, method):
    assert _server_sample_params(sampler)["sample_method"] == method


def test_default_keeps_a_family_without_a_pinned_method_on_sd_cpp_defaults():
    # z-image pins no method: the default request sends none, a pick sends its name.
    assert _oneshot_params(None, fam_name = "z-image")[0].sampling_method is None
    assert _oneshot_params("euler_a", fam_name = "z-image")[0].sampling_method == "euler_a"


def test_unsupported_choices_are_refused():
    with pytest.raises(ValueError, match = "UniPC sampler is not available on the native"):
        _oneshot_params("unipc")
    with pytest.raises(ValueError, match = "Schedule types are not adjustable"):
        _oneshot_params("euler", schedule_type = "karras")
    assert sd_cpp_sampling_method(None, "automatic", "euler") == "euler"


def test_every_native_name_is_an_sd_cpp_sampling_method():
    # stable-diffusion.cpp src/stable-diffusion.cpp sample_method_to_str (master-982-0518b34).
    known = {"euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "lcm"}
    assert {method for _label, method in SD_CPP_SAMPLERS.values()} <= known


def test_route_passes_the_sampler_to_the_native_engine():
    from routes.inference import _sampler_kwargs

    b = _backend("qwen-image")
    assert _sampler_kwargs(b.generate, "heun", None) == {"sampler": "heun", "schedule_type": None}
