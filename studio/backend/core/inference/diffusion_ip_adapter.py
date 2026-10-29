# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""IP-Adapter image prompts: the adapter each family uses, a capability gate, and its lifecycle on the resident pipe.

Like a ControlNet the adapter is a companion asset, fetched into the hub cache on first use. It is loaded onto the
resident pipeline once (``pipe._unsloth_ip_adapter`` names it), its scale is set per request, and a request without
an image prompt unloads it and puts the denoiser's own attention processors back, so later renders are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

_BLOCKED_QUANT = ("int8", "fp8", "nvfp4", "mxfp8")


@dataclass(frozen = True)
class IpAdapterEntry:
    id: str
    repo_id: str
    subfolder: str
    weight_name: str
    # Repo-relative (it holds a "/"), so diffusers does not look for it under ``subfolder``.
    image_encoder_folder: str
    # Image encoder + adapter resident in fp16, for the free-memory check before the first load.
    approx_mib: int


# h94/IP-Adapter is Apache-2.0. The ViT-H variant: its encoder is ~1.2 GiB in fp16, the bigG one ~3.5 GiB. FLUX.1 is
# absent on purpose: XLabs' adapter is under the FLUX.1-dev non-commercial licence.
_BY_FAMILY: dict[str, IpAdapterEntry] = {
    "sdxl": IpAdapterEntry(
        id = "h94/IP-Adapter:sdxl_vit-h",
        repo_id = "h94/IP-Adapter",
        subfolder = "sdxl_models",
        weight_name = "ip-adapter_sdxl_vit-h.safetensors",
        image_encoder_folder = "models/image_encoder",
        approx_mib = 2000,
    ),
}


def adapter_for(family: Optional[str]) -> Optional[IpAdapterEntry]:
    return _BY_FAMILY.get((family or "").strip().lower())


def adapter_repos() -> list[str]:
    """Every repo an image prompt can download, for the managed-account model check."""
    return sorted({entry.repo_id for entry in _BY_FAMILY.values()})


def supports_ip_adapter(
    *,
    engine: str,
    family: Optional[str],
    model_kind: Optional[str],
    transformer_quant: Optional[str],
    compiled: bool,
) -> bool:
    """diffusers, a family with an adapter, not GGUF / torchao, and not compiled (swapping attention processors under
    a compiled denoiser would recompile or replay stale graphs)."""
    if engine != "diffusers" or adapter_for(family) is None:
        return False
    if model_kind == "gguf" or compiled:
        return False
    if transformer_quant and str(transformer_quant).strip().lower() in _BLOCKED_QUANT:
        return False
    return True


def loaded_id(pipe: Any) -> Optional[str]:
    return getattr(pipe, "_unsloth_ip_adapter", None)


def apply(
    pipe: Any,
    entry: IpAdapterEntry,
    scale: float,
    *,
    denoiser_attr: str,
    device: Any,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Load ``entry`` onto ``pipe`` unless it already is, then set ``scale``. True when this call loaded it."""
    loaded = False
    if loaded_id(pipe) != entry.id:
        clear(pipe, denoiser_attr = denoiser_attr)
        denoiser = getattr(pipe, denoiser_attr)
        saved = dict(denoiser.attn_processors)
        try:
            pipe.load_ip_adapter(
                entry.repo_id,
                subfolder = entry.subfolder,
                weight_name = entry.weight_name,
                image_encoder_folder = entry.image_encoder_folder,
                cache_dir = cache_dir,
                token = token or None,
            )
            encoder = getattr(pipe, "image_encoder", None)
            if encoder is not None:
                # diffusers places it on pipe.device, which reads cpu for a pipe with offload hooks.
                encoder.to(device)
        except BaseException:
            _unload(pipe, denoiser, saved)
            raise
        pipe._unsloth_ip_adapter = entry.id
        pipe._unsloth_ip_saved_processors = saved
        loaded = True
    pipe.set_ip_adapter_scale(float(scale))
    return loaded


def clear(pipe: Any, *, denoiser_attr: str) -> bool:
    """Unload the adapter if one is attached. True when something was unloaded."""
    if loaded_id(pipe) is None:
        return False
    saved = getattr(pipe, "_unsloth_ip_saved_processors", None)
    pipe._unsloth_ip_adapter = None
    pipe._unsloth_ip_saved_processors = None
    _unload(pipe, getattr(pipe, denoiser_attr), saved)
    return True


def _unload(pipe: Any, denoiser: Any, saved: Optional[dict]) -> None:
    try:
        pipe.unload_ip_adapter()
    finally:
        # diffusers resets every processor to a fresh default; put back the exact ones the load ran with.
        if saved:
            denoiser.set_attn_processor(saved)
