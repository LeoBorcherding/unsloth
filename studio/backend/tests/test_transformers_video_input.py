# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video input on a safetensors VLM whose processor reads video (LFM3-Omni)."""

import base64
import os

import pytest

from core.inference.inference import (
    InferenceBackend,
    _processor_reads_video,
    _write_video_clips,
)


class _Processor:
    video_token = "<|video|>"

    def __init__(self, video_processor = object(), places_marker = True, raises = False):
        self.video_processor = video_processor
        self._places = places_marker
        self._raises = raises

    def apply_chat_template(self, messages, **kwargs):
        if self._raises:
            raise ValueError("template rejects video parts")
        parts = messages[0]["content"]
        marker = self.video_token if self._places and parts[0]["type"] == "video" else ""
        return f"<|im_start|>user\n{marker}x<|im_end|>\n"


def test_reads_video_needs_processor_and_template_marker():
    assert _processor_reads_video(_Processor()) is True
    assert _processor_reads_video(_Processor(video_processor = None)) is False
    assert _processor_reads_video(_Processor(places_marker = False)) is False
    assert _processor_reads_video(_Processor(raises = True)) is False
    assert _processor_reads_video(object()) is False


def test_clips_written_from_raw_and_data_uri_base64():
    data = b"\x00\x00\x00\x18ftypmp42"
    raw = base64.b64encode(data).decode()
    paths = _write_video_clips([raw, "data:video/mp4;base64," + raw])
    try:
        assert len(paths) == 2
        for path in paths:
            with open(path, "rb") as handle:
                assert handle.read() == data
    finally:
        for path in paths:
            os.remove(path)


def test_playlist_clip_refused_and_earlier_clips_removed(monkeypatch):
    import tempfile

    written = []

    original = tempfile.NamedTemporaryFile

    def tracking(*args, **kwargs):
        handle = original(*args, **kwargs)
        written.append(handle.name)
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracking)
    good = base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()
    playlist = base64.b64encode(b"#EXTM3U\n#EXTINF:1,\nfile:///etc/passwd\n").decode()
    with pytest.raises(Exception, match = "Playlists"):
        _write_video_clips([good, playlist])
    assert written and not any(os.path.exists(path) for path in written)


def test_worker_sees_videos_on_the_transformers_backend():
    from core.inference.worker import _backend_declares

    assert _backend_declares(InferenceBackend, "videos")


def test_video_only_turn_resolves_reasoning_markers():
    """A video-only turn renders through the processor template like an image turn, so its reasoning must be split."""
    import inspect

    from core.inference import inference as inference_module

    body = inspect.getsource(inference_module.InferenceBackend._generate_vision_response)
    call = body.index("detect_reasoning_channel_markers(processor")
    assert "if attached or videos" in body[call : call + 400]
