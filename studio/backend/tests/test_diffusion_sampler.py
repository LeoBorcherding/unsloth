# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Images page sampler choice and the v-prediction markers of single-file SDXL checkpoints."""

from __future__ import annotations

import types

import pytest

diffusers = pytest.importorskip("diffusers")

from core.inference.diffusion_sampler import (
    apply_checkpoint_prediction_type,
    build_scheduler,
    checkpoint_prediction_overrides,
    sampler_options,
)
from core.inference.image_gallery import _params_text

# stabilityai/stable-diffusion-xl-base-1.0 scheduler/scheduler_config.json
_SDXL = dict(
    beta_end = 0.012,
    beta_schedule = "scaled_linear",
    beta_start = 0.00085,
    clip_sample = False,
    interpolation_type = "linear",
    num_train_timesteps = 1000,
    prediction_type = "epsilon",
    sample_max_value = 1.0,
    set_alpha_to_one = False,
    skip_prk_steps = True,
    steps_offset = 1,
    timestep_spacing = "leading",
    trained_betas = None,
    use_karras_sigmas = False,
)


def _sdxl(**overrides):
    return diffusers.EulerDiscreteScheduler.from_config({**_SDXL, **overrides})


def test_sdxl_offers_the_popular_samplers():
    options = {o["id"]: o["schedule_types"] for o in sampler_options(_sdxl())}
    assert list(options) == ["default", "euler", "euler_a", "dpmpp_2m", "dpmpp_2m_sde"]
    assert options["dpmpp_2m"] == ["automatic", "uniform", "karras", "exponential"]
    assert options["euler_a"] == ["automatic", "uniform"]


# Qwen/Qwen-Image scheduler/scheduler_config.json: dynamic mu, exponential time shift, terminal stretch.
_QWEN = dict(
    base_image_seq_len = 256,
    base_shift = 0.5,
    invert_sigmas = False,
    max_image_seq_len = 8192,
    max_shift = 0.9,
    num_train_timesteps = 1000,
    shift = 1.0,
    shift_terminal = 0.02,
    stochastic_sampling = False,
    time_shift_type = "exponential",
    use_beta_sigmas = False,
    use_dynamic_shifting = True,
    use_exponential_sigmas = False,
    use_karras_sigmas = False,
)
# Tongyi-MAI/Z-Image-Turbo scheduler/scheduler_config.json: static shift.
_ZIMAGE = dict(num_train_timesteps = 1000, use_dynamic_shifting = False, shift = 3.0)
_FLOW_IDS = ["default", "euler", "euler_a", "unipc"]


def _flow(config):
    return diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)


# Alpha-VLLM/Lumina-Image-2.0 scheduler/scheduler_config.json: static shift 6.
_LUMINA = dict(
    base_image_seq_len = 256,
    base_shift = 0.5,
    invert_sigmas = False,
    max_image_seq_len = 4096,
    max_shift = 1.15,
    num_train_timesteps = 1000,
    shift = 6.0,
    shift_terminal = None,
    use_dynamic_shifting = False,
)


def _pipeline_timesteps(scheduler, steps = 12, mu = 0.83):
    # FLUX / FLUX.2 / Qwen-Image / Lumina-2 / Krea-2: an np.linspace grid plus mu through retrieve_timesteps.
    import numpy as np
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import retrieve_timesteps

    retrieve_timesteps(scheduler, steps, "cpu", sigmas = np.linspace(1.0, 1 / steps, steps), mu = mu)
    return scheduler


def _zimage_timesteps(scheduler, steps = 8, mu = None):
    # Z-Image: a Python list (get_default_z_image_sigmas returns .tolist()).
    from diffusers.pipelines.z_image import pipeline_z_image as z

    kwargs = {} if mu is None else {"mu": mu}
    z.retrieve_timesteps(scheduler, steps, "cpu", sigmas = z.get_default_z_image_sigmas(steps), **kwargs)
    return scheduler


def _ideogram_timesteps(scheduler, steps = 10, mu = None):
    # Ideogram-4: its own logit-normal grid as a list, set_timesteps called directly, no mu.
    from diffusers.pipelines.ideogram4 import pipeline_ideogram4 as ideo

    scheduler.set_timesteps(sigmas = ideo._logit_normal_sigmas(steps, 0.4).tolist(), device = "cpu")
    return scheduler


def _no_grid_timesteps(scheduler, steps = 10, mu = None):
    # HiDream's UniPC branch: steps only, the scheduler's own default grid.
    scheduler.set_timesteps(steps, device = "cpu")
    return scheduler


_CALLS = [
    (_pipeline_timesteps, _QWEN),
    (_pipeline_timesteps, _ZIMAGE),
    (_pipeline_timesteps, _LUMINA),
    (_zimage_timesteps, _ZIMAGE),
    (_zimage_timesteps, _QWEN),
    (_ideogram_timesteps, _ZIMAGE),
    (_no_grid_timesteps, _LUMINA),
]


@pytest.mark.parametrize("config", [_QWEN, _ZIMAGE, {}])
def test_flow_matching_offers_the_flow_samplers(config):
    options = sampler_options(_flow(config))
    assert [o["id"] for o in options] == _FLOW_IDS
    assert all(o["schedule_types"] == ["automatic"] for o in options)
    assert sampler_options(None) == []


@pytest.mark.parametrize("call, config", _CALLS, ids = lambda v: getattr(v, "__name__", ""))
@pytest.mark.parametrize("sampler", ["euler", "euler_a", "unipc"])
def test_flow_samplers_keep_the_resident_schedule(call, config, sampler):
    from core.inference.diffusion_sampler import bind_flow_sampling

    resident = _flow(config)
    built, resolved = build_scheduler(resident, sampler, None)
    assert resolved is None
    # Pipelines read these off the swapped scheduler to compute mu.
    for key in ("base_shift", "max_shift", "base_image_seq_len", "max_image_seq_len"):
        assert built.config.get(key) == resident.config.get(key)
    bind_flow_sampling(built, None)
    mu = 0.83 if config.get("use_dynamic_shifting") else None
    expected = call(resident, mu = mu)
    got = call(built, mu = mu)
    assert got.sigmas.shape == expected.sigmas.shape
    assert (got.sigmas - expected.sigmas).abs().max() < 1e-5
    # The model is fed the same (float) timesteps the resident would give it.
    assert (got.timesteps.float() - expected.timesteps.float()).abs().max() < 1e-3


def test_unbound_unipc_cannot_take_a_list_grid():
    # What the bind fixes: UniPC's own shift math multiplies the list Z-Image passes.
    built, _ = build_scheduler(_flow(_ZIMAGE), "unipc", None)
    with pytest.raises(TypeError):
        _zimage_timesteps(built)


def test_unipc_carries_the_flow_shift():
    built, _ = build_scheduler(_flow(_ZIMAGE), "unipc", None)
    assert type(built).__name__ == "UniPCMultistepScheduler"
    assert built.config.use_flow_sigmas is True
    assert built.config.prediction_type == "flow_prediction"
    assert built.config.flow_shift == 3.0
    assert built.config.use_dynamic_shifting is False
    dynamic, _ = build_scheduler(_flow(_QWEN), "unipc", None)
    assert dynamic.config.use_dynamic_shifting is True
    assert dynamic.config.shift_terminal == 0.02
    assert dynamic.config.time_shift_type == "exponential"


def test_euler_a_is_stochastic_flow_euler():
    built, _ = build_scheduler(_flow(_QWEN), "euler_a", None)
    assert type(built).__name__ == "FlowMatchEulerDiscreteScheduler"
    assert built.config.stochastic_sampling is True
    assert build_scheduler(built, "euler", None)[0].config.stochastic_sampling is False


@pytest.mark.parametrize("sampler", ["euler", "euler_a", "unipc"])
def test_flow_samplers_land_on_the_clean_sample(sampler):
    # A rectified-flow model that predicts the exact velocity: every correct sampler ends on x0.
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_sampler import bind_flow_sampling

    x0 = torch.randn(2, 16, 4, generator = torch.Generator().manual_seed(1))
    noise = torch.randn(2, 16, 4, generator = torch.Generator().manual_seed(2))
    built, _ = build_scheduler(_flow(_QWEN), sampler, None)
    bind_flow_sampling(built, [torch.Generator().manual_seed(s) for s in (5, 6)])
    _pipeline_timesteps(built, steps = 10)
    sample = noise.clone()
    for i, t in enumerate(built.timesteps):
        velocity = (sample - x0) / built.sigmas[i]
        sample = built.step(velocity, t, sample, return_dict = False)[0]
    assert (sample - x0).abs().max() < 1e-4


def _euler_a_run(seeds):
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_sampler import bind_flow_sampling

    built, _ = build_scheduler(_flow(_QWEN), "euler_a", None)
    bind_flow_sampling(built, [torch.Generator().manual_seed(s) for s in seeds])
    _pipeline_timesteps(built, steps = 6)
    sample = torch.ones(len(seeds), 8, 4)
    for t in built.timesteps[:-1]:
        # Zero velocity: the renoise is all that moves the sample.
        sample = built.step(torch.zeros_like(sample), t, sample, return_dict = False)[0]
    return sample


def test_euler_a_noise_follows_the_request_seeds():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    first = _euler_a_run([7, 8])
    torch.manual_seed(1)  # the global RNG must not matter
    again = _euler_a_run([7, 8])
    assert torch.equal(first, again)
    # Per image: image 0 of seeds [7, 8] is image 0 of seeds [7, 9].
    assert torch.equal(_euler_a_run([7, 9])[0], first[0])
    assert not torch.equal(_euler_a_run([7, 9])[1], first[1])


def test_unipc_gets_the_flow_forward_process_for_img2img():
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_sampler import bind_flow_sampling

    resident = _pipeline_timesteps(_flow(_ZIMAGE))
    built, _ = build_scheduler(resident, "unipc", None)
    assert not hasattr(built, "scale_noise")
    bind_flow_sampling(built, None)
    _pipeline_timesteps(built)
    sample, noise = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    for scheduler in (resident, built):
        scheduler.set_begin_index(4)
    expected = resident.scale_noise(sample, resident.timesteps[4:5], noise)
    got = built.scale_noise(sample, built.timesteps[4:5], noise)
    assert (got - expected).abs().max() < 1e-5


def test_shift_override_on_a_dynamic_unipc_is_static():
    from core.inference.diffusion_flow_shift import scheduler_with_flow_shift
    from core.inference.diffusion_sampler import bind_flow_sampling

    built, _ = build_scheduler(_flow(_QWEN), "unipc", None)
    shifted = scheduler_with_flow_shift(built, 3.1)
    bind_flow_sampling(shifted, None)
    assert shifted.config.flow_shift == 3.1
    assert shifted.config.use_dynamic_shifting is False
    assert shifted.config.shift_terminal is None
    static = scheduler_with_flow_shift(_flow(_QWEN), 3.1)
    got = _pipeline_timesteps(shifted).sigmas
    expected = _pipeline_timesteps(static).sigmas
    assert (got - expected).abs().max() < 1e-5


@pytest.mark.parametrize("cls", ["DPMSolverMultistepScheduler", "FlowMatchHeunDiscreteScheduler"])
def test_left_out_flow_schedulers_cannot_take_the_pipeline_call(cls):
    flow = {"use_flow_sigmas": True, "prediction_type": "flow_prediction"}
    scheduler = getattr(diffusers, cls).from_config({**_QWEN, **flow})
    with pytest.raises(ValueError, match = "custom sigmas"):
        _pipeline_timesteps(scheduler)


def test_unipc_is_withheld_where_it_cannot_follow_the_resident():
    karras = _flow({**_QWEN, "use_karras_sigmas": True})
    assert [o["id"] for o in sampler_options(karras)] == ["default", "euler", "euler_a"]
    with pytest.raises(ValueError, match = "UniPC sampler is not available"):
        build_scheduler(karras, "unipc", None)


def test_flow_configured_dpm_offers_nothing():
    flow_dpm = diffusers.DPMSolverMultistepScheduler(
        use_flow_sigmas = True, prediction_type = "flow_prediction"
    )
    assert sampler_options(flow_dpm) == []


def test_uniform_clears_an_inherited_lu_schedule():
    dpm = diffusers.DPMSolverMultistepScheduler.from_config({**_SDXL, "use_lu_lambdas": True})
    built, _ = build_scheduler(dpm, "dpmpp_2m", "uniform")
    assert built.config.use_lu_lambdas is False
    assert built.config.use_karras_sigmas is False


def test_default_keeps_the_shipped_scheduler():
    assert build_scheduler(_sdxl(), None, None) == (None, None)
    assert build_scheduler(_sdxl(), "default", "automatic") == (None, None)


@pytest.mark.parametrize(
    "sampler, schedule, cls, algorithm, flags",
    [
        ("dpmpp_2m", "automatic", "DPMSolverMultistepScheduler", "dpmsolver++", "karras"),
        ("dpmpp_2m_sde", None, "DPMSolverMultistepScheduler", "sde-dpmsolver++", "exponential"),
        ("dpmpp_2m", "uniform", "DPMSolverMultistepScheduler", "dpmsolver++", "uniform"),
        ("euler", "karras", "EulerDiscreteScheduler", None, "karras"),
        ("default", "exponential", "EulerDiscreteScheduler", None, "exponential"),
    ],
)
def test_build_scheduler(sampler, schedule, cls, algorithm, flags):
    built, resolved = build_scheduler(_sdxl(use_karras_sigmas = True), sampler, schedule)
    assert type(built).__name__ == cls
    assert resolved == flags
    assert built.config.use_karras_sigmas is (flags == "karras")
    assert built.config.use_exponential_sigmas is (flags == "exponential")
    if algorithm:
        assert built.config.algorithm_type == algorithm


def test_euler_a_automatic_keeps_the_config_spacing():
    built, resolved = build_scheduler(_sdxl(), "euler_a", None)
    assert type(built).__name__ == "EulerAncestralDiscreteScheduler"
    assert resolved is None
    assert built.config.timestep_spacing == "leading"


def test_prediction_settings_carry_over():
    vpred = _sdxl(
        prediction_type = "v_prediction",
        rescale_betas_zero_snr = True,
        timestep_spacing = "trailing",
    )
    for sampler in ("euler_a", "dpmpp_2m", "dpmpp_2m_sde"):
        built, _ = build_scheduler(vpred, sampler, None)
        assert built.config.prediction_type == "v_prediction"
        assert built.config.rescale_betas_zero_snr is True
        assert built.config.timestep_spacing == "trailing"


def test_zero_snr_automatic_avoids_exponential():
    vpred = _sdxl(prediction_type = "v_prediction", rescale_betas_zero_snr = True)
    built, resolved = build_scheduler(vpred, "dpmpp_2m_sde", "automatic")
    assert resolved == "karras"
    assert built.config.use_karras_sigmas is True
    assert build_scheduler(vpred, "dpmpp_2m_sde", "exponential")[1] == "exponential"


@pytest.mark.parametrize("sampler", ["dpmpp_2m", "dpmpp_2m_sde"])
def test_zero_snr_karras_runs_every_step_without_a_pipeline_begin_index(sampler):
    # The SDXL ControlNet pipeline never calls set_begin_index; the duplicate first timestep overran.
    torch = pytest.importorskip("torch")
    vpred = _sdxl(
        prediction_type = "v_prediction", rescale_betas_zero_snr = True, timestep_spacing = "trailing"
    )
    built, _ = build_scheduler(vpred, sampler, "karras")
    built.set_timesteps(28)
    assert built.timesteps[0] == built.timesteps[1]
    sample = torch.randn(1, 4, 8, 8)
    generator = torch.Generator().manual_seed(0)
    for t in built.timesteps:
        sample = built.step(torch.zeros_like(sample), t, sample, generator = generator).prev_sample
    assert torch.isfinite(sample).all()


def test_unavailable_choices_are_refused():
    with pytest.raises(ValueError, match = "does not apply"):
        build_scheduler(_sdxl(), "euler_a", "karras")
    with pytest.raises(ValueError, match = "not available"):
        build_scheduler(diffusers.FlowMatchEulerDiscreteScheduler(), "dpmpp_2m", None)
    with pytest.raises(ValueError, match = "not available"):
        build_scheduler(_sdxl(), "unipc", None)
    with pytest.raises(ValueError, match = "Heun sampler is not available"):
        build_scheduler(_flow(_QWEN), "heun", None)
    with pytest.raises(ValueError, match = "does not apply"):
        build_scheduler(diffusers.FlowMatchEulerDiscreteScheduler(), "euler", "karras")
    with pytest.raises(ValueError, match = "Unknown sampler"):
        build_scheduler(_sdxl(), "lms", None)


def _checkpoint(path, *markers):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    tensors = {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(2)}
    tensors.update({m: torch.zeros(0) for m in markers})
    save_file(tensors, str(path))
    return str(path)


def test_vpred_ztsnr_markers(tmp_path):
    pipe = types.SimpleNamespace(scheduler = _sdxl())
    path = _checkpoint(tmp_path / "noob.safetensors", "v_pred", "ztsnr")
    assert apply_checkpoint_prediction_type(pipe, path) is True
    config = pipe.scheduler.config
    assert type(pipe.scheduler).__name__ == "EulerDiscreteScheduler"
    assert config.prediction_type == "v_prediction"
    assert config.rescale_betas_zero_snr is True
    assert config.timestep_spacing == "trailing"
    assert apply_checkpoint_prediction_type(pipe, path) is False


def test_vpred_marker_without_ztsnr(tmp_path):
    path = _checkpoint(tmp_path / "vpred.safetensors", "v_pred")
    assert checkpoint_prediction_overrides(path) == {"prediction_type": "v_prediction"}


def test_unmarked_and_unreadable_checkpoints_keep_epsilon(tmp_path):
    pipe = types.SimpleNamespace(scheduler = _sdxl())
    plain = _checkpoint(tmp_path / "sdxl.safetensors")
    assert apply_checkpoint_prediction_type(pipe, plain) is False
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    assert apply_checkpoint_prediction_type(pipe, str(junk)) is False
    assert apply_checkpoint_prediction_type(pipe, None) is False
    assert pipe.scheduler.config.prediction_type == "epsilon"


def _meta(**extra):
    return dict(
        prompt = "a cat", steps = 25, guidance = 7, seed = 1, width = 1024, height = 1024, model = "m", **extra
    )


def test_parameters_text_names_the_sampler():
    text = _params_text(_meta(sampler = "dpmpp_2m", schedule_type = "karras"))
    assert "Steps: 25, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7," in text
    assert "Sampler" not in _params_text(_meta())
    assert "Schedule type" not in _params_text(_meta(sampler = "euler_a"))
    assert "Sampler: UniPC," in _params_text(_meta(sampler = "unipc"))
    assert "Sampler: Heun," in _params_text(_meta(sampler = "heun"))


def test_route_refuses_a_sampler_the_engine_cannot_run():
    from routes.inference import _sampler_kwargs

    def native_generate(*, prompt):
        return None

    def diffusers_generate(
        *,
        prompt,
        sampler = None,
        schedule_type = None,
    ):
        return None

    assert _sampler_kwargs(native_generate, None, None) == {}
    assert _sampler_kwargs(native_generate, "default", "automatic") == {}
    with pytest.raises(ValueError, match = "diffusers engine only"):
        _sampler_kwargs(native_generate, "dpmpp_2m", None)
    assert _sampler_kwargs(diffusers_generate, "euler", "karras") == {
        "sampler": "euler",
        "schedule_type": "karras",
    }
