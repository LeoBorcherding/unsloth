# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image prompts through Unsloth's real diffusers backend: tiny SDXL plus a random tiny IP-Adapter built on disk in
the h94 layout, so the real ``load_ip_adapter`` / ``unload_ip_adapter`` paths run."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

from PIL import Image  # noqa: E402

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

TINY_SDXL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
SIZE = 256
EMBED = 32


def _b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _style():
    arr = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)
    arr[..., 0] = np.linspace(0, 255, SIZE, dtype = np.uint8)[None, :]
    arr[..., 2] = 200
    return Image.fromarray(arr, "RGB")


def _write_adapter(root: Path, unet) -> None:
    """h94's layout: sdxl_models/<weights>.safetensors and models/image_encoder/, random weights at tiny sizes."""
    from diffusers.models.attention_processor import IPAdapterAttnProcessor
    from diffusers.models.embeddings import ImageProjection
    from safetensors.torch import save_file

    torch.manual_seed(0)
    cfg = unet.config
    tensors = {}
    key_id = 1
    for name in unet.attn_processors.keys():
        if not name.endswith("attn1.processor"):
            if name.startswith("mid_block"):
                hidden = cfg.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                hidden = list(reversed(cfg.block_out_channels))[int(name[len("up_blocks.")])]
            else:
                hidden = cfg.block_out_channels[int(name[len("down_blocks.")])]
            sd = IPAdapterAttnProcessor(
                hidden_size = hidden, cross_attention_dim = cfg.cross_attention_dim, scale = 1.0
            ).state_dict()
            tensors[f"ip_adapter.{key_id}.to_k_ip.weight"] = sd["to_k_ip.0.weight"] * 4
            tensors[f"ip_adapter.{key_id}.to_v_ip.weight"] = sd["to_v_ip.0.weight"] * 4
            key_id += 2
    proj = ImageProjection(
        cross_attention_dim = cfg.cross_attention_dim, image_embed_dim = EMBED, num_image_text_embeds = 4
    ).state_dict()
    tensors["image_proj.proj.weight"] = proj["image_embeds.weight"]
    tensors["image_proj.proj.bias"] = proj["image_embeds.bias"]
    tensors["image_proj.norm.weight"] = proj["norm.weight"]
    tensors["image_proj.norm.bias"] = proj["norm.bias"]
    (root / "sdxl_models").mkdir(parents = True)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(root / "sdxl_models" / "ip-adapter_sdxl_vit-h.safetensors"),
    )
    encoder = transformers.CLIPVisionModelWithProjection(
        transformers.CLIPVisionConfig(
            hidden_size = 32,
            intermediate_size = 37,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            image_size = 32,
            patch_size = 4,
            projection_dim = EMBED,
        )
    )
    encoder.save_pretrained(str(root / "models" / "image_encoder"))


@pytest.fixture(scope = "module")
def backend(tmp_path_factory):
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(TINY_SDXL, local_files_only = True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny SDXL pipeline not in the local hub cache: {exc}")
    from core.inference.diffusion import DiffusionBackend

    b = DiffusionBackend()
    # CPU fp32: on MPS bf16 the tiny random adapter overflows to NaN now and then.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            DiffusionBackend,
            "_pick_device_and_dtype",
            lambda self, ordinal = None: ("cpu", torch.float32),
        )
        b.load_pipeline(path, family_override = "sdxl", speed_mode = "off")
    adapter_root = tmp_path_factory.mktemp("ip_adapter_repo")
    _write_adapter(adapter_root, b._state.pipe.unet)
    b._test_adapter_root = str(adapter_root)
    yield b
    b.unload()


@pytest.fixture
def local_adapter(backend, monkeypatch):
    from dataclasses import replace

    from core.inference import diffusion_ip_adapter as ipa

    entry = replace(ipa.adapter_for("sdxl"), repo_id = backend._test_adapter_root, approx_mib = 0)
    monkeypatch.setitem(ipa._BY_FAMILY, "sdxl", entry)
    loads = []
    cls = type(backend._state.pipe)
    original = cls.load_ip_adapter

    def counting(self, *args, **kwargs):
        loads.append(args[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "load_ip_adapter", counting)
    return entry, loads


def _render(backend, **extra):
    out = backend.generate(
        prompt = "a lighthouse", steps = 3, guidance = 5.0, seed = 11, width = SIZE, height = SIZE, **extra
    )
    return out, np.asarray(out["images"][0].convert("RGB"), dtype = np.int16)


def test_status_offers_image_prompts_on_sdxl(backend):
    assert backend.status()["supports_ip_adapter"] is True


def test_load_once_scale_per_request_and_unload_restores_plain_renders(backend, local_adapter):
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0

    from core.inference import diffusion_ip_adapter as ipa

    entry, loads = local_adapter
    pipe = backend._state.pipe
    processors_before = dict(pipe.unet.attn_processors)
    plain, ref = _render(backend)
    assert plain["ip_adapter"] is None
    _, again = _render(backend)
    assert np.array_equal(ref, again)

    style = _b64(_style())
    out, with_ip = _render(backend, ip_adapter = (style, 0.6))
    assert out["workflow"] == "txt2img"
    assert out["ip_adapter"] == entry.id and out["ip_adapter_scale"] == 0.6
    assert ipa.loaded_id(pipe) == entry.id and pipe.image_encoder is not None
    ip_procs = [
        p for p in pipe.unet.attn_processors.values() if isinstance(p, IPAdapterAttnProcessor2_0)
    ]
    assert ip_procs and all(p.scale == [0.6] for p in ip_procs)
    assert np.abs(with_ip - ref).max() > 0

    _render(backend, ip_adapter = (style, 0.2))
    assert loads == [entry.repo_id]
    assert all(p.scale == [0.2] for p in ip_procs)

    after, plain_again = _render(backend)
    assert after["ip_adapter"] is None
    assert ipa.loaded_id(pipe) is None and pipe.image_encoder is None
    assert all(pipe.unet.attn_processors[k] is v for k, v in processors_before.items())
    assert pipe.unet.encoder_hid_proj is None
    assert np.array_equal(ref, plain_again)


def test_scale_zero_is_plain_text_to_image(backend, local_adapter):
    from core.inference import diffusion_ip_adapter as ipa

    _, loads = local_adapter
    out, _ = _render(backend, ip_adapter = (_b64(_style()), 0.0))
    assert out["ip_adapter"] is None and loads == [] and ipa.loaded_id(backend._state.pipe) is None


def test_image_prompt_with_img2img_is_refused(backend, local_adapter):
    _, loads = local_adapter
    with pytest.raises(ValueError, match = "plain text-to-image only"):
        backend.generate(
            prompt = "a lighthouse",
            steps = 3,
            guidance = 5.0,
            seed = 11,
            init_image = _b64(_style()),
            strength = 0.5,
            ip_adapter = (_b64(_style()), 0.6),
        )
    assert loads == []
