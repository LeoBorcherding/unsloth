# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, media architecture preflight."""

import os

from typing import Optional
from utils.gguf_archs import (
    is_audio_cpp_gguf_architecture,
    is_speech_gguf_architecture,
    SPEECH_GGUF_ARCHS as _SPEECH_GGUF_ARCHS,
)

from .log import logger
from utils import gguf_archs as _gguf_archs
from utils.gguf_archs import is_foreign_asr_gguf_architecture


class _PreflightMixin:
    # ── Lifecycle ─────────────────────────────────────────────────

    _IMAGE_ARCHES = _gguf_archs.DIFFUSION_IMAGE_GGUF_ARCHS
    _AMBIGUOUS_IMAGE_ARCHES = _gguf_archs.AMBIGUOUS_DIFFUSION_GGUF_ARCHS
    _VIDEO_ARCHES = _gguf_archs.VIDEO_GGUF_ARCHS
    _UNRUNNABLE_MEDIA_ARCHES = _gguf_archs.UNSUPPORTED_DIFFUSION_GGUF_ARCHS
    _DIFFUSION_ARCHES = (
        _IMAGE_ARCHES | _AMBIGUOUS_IMAGE_ARCHES | _VIDEO_ARCHES | _UNRUNNABLE_MEDIA_ARCHES
    )
    _SPEECH_ARCHES = _SPEECH_GGUF_ARCHS

    _PLACEHOLDER_ARCHES = _gguf_archs.PLACEHOLDER_DIFFUSION_GGUF_ARCHS

    @staticmethod
    def _video_arch_is_pickable(
        arch: Optional[str], gguf_path: Optional[str], model_identifier: Optional[str]
    ) -> bool:
        """Asks arch, repo id, then filename in _arch_to_task's order; an MoE counts only if it names one expert of a high/low noise pair, and the engine can build it."""
        try:
            from core.inference.video_families import detect_video_family

            from routes.models import _video_family_buildable

            fam = detect_video_family("", override = (arch or "").strip().lower() or None)
            if fam is None:
                for needle in (model_identifier, os.path.basename(gguf_path or "")):
                    if not needle:
                        continue
                    fam = detect_video_family(needle)
                    if fam is not None:
                        break
            if fam is None:
                return False
            from core.inference.video_moe_pair import moe_pick_pairs

            # An MoE expert is offered when its name pairs the partner the loader assembles it with.
            return moe_pick_pairs(fam, gguf_path) and _video_family_buildable(fam)
        except Exception as e:  # noqa: BLE001 -- never lose the page over a probe failure
            logger.debug("Family probe failed for video arch: %s", e)
            return True

    @staticmethod
    def _image_arch_is_pickable(
        gguf_path: Optional[str],
        model_identifier: Optional[str],
        arch: Optional[str] = None,
    ) -> bool:
        """The Images picker's fail-open buildability gate (Qwen-Image-2.1 needs diffusers 0.41)."""
        try:
            from routes.models import _diffusion_gguf_family_hints, _gguf_family_buildable
            return _gguf_family_buildable(
                _diffusion_gguf_family_hints(
                    arch, (model_identifier, os.path.basename(gguf_path or ""))
                )
            )
        except Exception as e:  # noqa: BLE001 -- never lose the page over a probe failure
            logger.debug("Family probe failed for image arch: %s", e)
            return True

    @staticmethod
    def _ambiguous_image_arch_is_pickable(
        gguf_path: Optional[str], model_identifier: Optional[str]
    ) -> bool:
        """Matches routes.models._arch_to_task so the refusal never names a missing Images row; imported lazily (routes imports this module); fails open on probe error."""
        try:
            from core.inference.diffusion_engine_router import family_buildable_here
            from core.inference.diffusion_families import (
                detect_family_for_pick,
                family_gguf_loadable,
            )

            for needle in (model_identifier, os.path.basename(gguf_path or "")):
                if not needle:
                    continue
                fam = detect_family_for_pick(needle)
                if fam is not None:
                    return family_gguf_loadable(fam) and family_buildable_here(
                        fam, model_kind = "gguf"
                    )
            return False
        except Exception as e:  # noqa: BLE001 -- never lose the page over a probe failure
            logger.debug("Family probe failed for ambiguous image arch: %s", e)
            return True

    def _non_chat_gguf_refusal(self, gguf_path: str) -> Optional[str]:
        """Header-only refusal of adapter and media GGUFs: an adapter, a media architecture, or none
        at all once the header parsed."""
        # Metadata, not filenames: an adapter may declare a chat arch; a merged LoRA is a full model.
        # Checked before the walk gate: a type that was read is definitive even from a partial prefix.
        if getattr(self, "_gguf_type", None) == "adapter":
            return (
                "This GGUF is a LoRA adapter and needs a matching base model. "
                "Select a full model GGUF for chat, or use this adapter with its base model."
            )
        arch = (self._architecture or "").strip().lower()
        # A media arch is KV 0 so a partial remote prefix can carry it; the walk still gates the no-arch fallback, since an incomplete read looks like none.
        if (
            arch not in self._DIFFUSION_ARCHES
            and not is_speech_gguf_architecture(arch)
            and not is_audio_cpp_gguf_architecture(arch)
            and not is_foreign_asr_gguf_architecture(arch)
            and not getattr(self, "_gguf_header_parsed", False)
        ):
            return None
        if arch in self._PLACEHOLDER_ARCHES:
            # A placeholder arch counts as no architecture, so the name-based branch below decides the page.
            arch = ""
        if arch:
            if is_audio_cpp_gguf_architecture(arch):
                return (
                    "This is an audio GGUF (speech, music or transcription), which cannot run as "
                    "a chat model. Open it from the Audio page instead."
                )
            if is_foreign_asr_gguf_architecture(arch):
                return (
                    f"This is a '{arch}' speech-recognition GGUF (parakeet.cpp / LocalAI), which "
                    "llama.cpp cannot load. Updating llama.cpp will not help. For Parakeet in "
                    "Unsloth, open the Audio page and use the curated audio.cpp model "
                    "(Parakeet-TDT-0.6B-v3-GGUF)."
                )
            if is_speech_gguf_architecture(arch):
                # Points at the Transformers build: the Audio page cannot list a speech GGUF either.
                return (
                    "This is a text-to-speech GGUF, which llama.cpp cannot load: it has no "
                    "CSM decoder. Run the model's Transformers build (for example "
                    "unsloth/csm-1b) from the Audio page instead."
                )
            if arch in self._UNRUNNABLE_MEDIA_ARCHES:
                return (
                    f"This is an image / video generation GGUF (architecture '{arch}'), "
                    "which cannot run as a chat model. Unsloth Studio cannot run this "
                    "architecture at all: neither the Images page nor the Video page "
                    "accepts it."
                )
            if arch in self._VIDEO_ARCHES:
                # wan alone is not enough: the canonical classifier resolves the family by name and rejects an MoE the GGUF loader cannot assemble.
                if self._video_arch_is_pickable(arch, gguf_path, self._model_identifier):
                    return (
                        f"This is a text-to-video GGUF (architecture '{arch}'), which "
                        "cannot run as a chat model. Open it from the Video page instead."
                    )
                return (
                    f"This is a text-to-video GGUF (architecture '{arch}'), which cannot "
                    "run as a chat model. Unsloth cannot assemble this particular model "
                    "from a GGUF, so the Video page does not list it either."
                )
            if arch in self._IMAGE_ARCHES:
                if self._image_arch_is_pickable(gguf_path, self._model_identifier, arch):
                    return (
                        f"This is an image-generation GGUF (architecture '{arch}'), which "
                        "cannot run as a chat model. Open it from the Images page instead."
                    )
                return (
                    f"This is an image-generation GGUF (architecture '{arch}'), which "
                    "cannot run as a chat model. No image engine installed here can build "
                    "this model family, so the Images page does not list it either. "
                    "Updating Unsloth may add support for it."
                )
            if arch in self._AMBIGUOUS_IMAGE_ARCHES:
                if self._ambiguous_image_arch_is_pickable(gguf_path, self._model_identifier):
                    return (
                        f"This is an image-generation GGUF (architecture '{arch}'), which "
                        "cannot run as a chat model. Open it from the Images page instead."
                    )
                return (
                    f"This is an image-generation GGUF (architecture '{arch}'), which "
                    "cannot run as a chat model. Unsloth could not tell which model family "
                    f"this file belongs to -- '{arch}' is shared by several -- so the "
                    "Images page cannot assemble it either."
                )
            return None

        # A later gguf-split shard declares no architecture; llama.cpp's own first-split error is better.
        # Requires split.no > 0 and nothing declared, since shard 1 has the split keys too.
        if self._architecture is None and (self._gguf_split_index or 0) > 0:
            return None

        # No architecture declared: not loadable by llama-server whatever it is; name a page only when the identifiers make that clear.
        needles = [n for n in (self._model_identifier, os.path.basename(gguf_path or "")) if n]
        # Name a page only if its picker would actually list the model.
        try:
            from core.inference.video_families import detect_video_family
            if any(detect_video_family(n) is not None for n in needles):
                if self._video_arch_is_pickable(None, gguf_path, self._model_identifier):
                    return (
                        "This is a text-to-video model, not a chat model: its GGUF carries "
                        "no llama.cpp model metadata, so llama-server cannot load it. Open "
                        "it from the Video page instead."
                    )
                return (
                    "This is a text-to-video model, not a chat model, and its GGUF carries "
                    "no llama.cpp model metadata. Unsloth cannot assemble this particular "
                    "model from a GGUF either, so the Video page does not list it."
                )
        except Exception as e:  # noqa: BLE001 -- naming the page is a nicety, never a gate
            logger.debug("Video family probe failed during non-chat GGUF preflight: %s", e)
        try:
            from core.inference.diffusion_families import detect_family
            if any(detect_family(n) is not None for n in needles):
                if self._ambiguous_image_arch_is_pickable(gguf_path, self._model_identifier):
                    return (
                        "This is an image-generation model, not a chat model: its GGUF "
                        "carries no llama.cpp model metadata, so llama-server cannot load "
                        "it. Open it from the Images page instead."
                    )
                return (
                    "This is an image-generation model, not a chat model, and its GGUF "
                    "carries no llama.cpp model metadata. Unsloth cannot assemble this "
                    "particular model from a GGUF either, so the Images page does not list "
                    "it."
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Image family probe failed during non-chat GGUF preflight: %s", e)
        return (
            "This GGUF carries no llama.cpp model metadata (no general.architecture), so "
            "llama-server cannot load it as a chat model. Image and video GGUFs are "
            "packaged this way -- try the Images or Video page, or pick a chat GGUF."
        )

    # Distro packages for the Linux prebuilt's shared libs, for the loader-failure hint.
    _SHARED_LIB_PACKAGES = {
        "libgomp.so.1": " (Debian/Ubuntu: libgomp1, Fedora/RHEL: libgomp)",
    }

    # Libraries Unsloth ships in build/bin; a loader failure naming one means a broken managed
    # runtime, not a missing package.
    _BUNDLED_LIB_PREFIXES = ("libllama", "libggml", "libmtmd")
