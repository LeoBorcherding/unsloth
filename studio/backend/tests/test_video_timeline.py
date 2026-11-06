# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The video timeline's join: trims, frame clock, generated-part overlaps, canvas, and the route."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

av = pytest.importorskip("av")

from core.inference import video_timeline as vt  # noqa: E402


def _clip(count, fps = 24, size = (64, 48), base = 0):
    """Frame i is solid gray 4 * (base + i), so a decoded frame names its source index."""
    buf = io.BytesIO()
    with av.open(buf, "w", format = "mp4") as out:
        stream = out.add_stream("libx264", rate = fps)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        stream.options = {"qp": "0"}
        for i in range(count):
            pixels = np.full((size[1], size[0], 3), (4 * (base + i)) % 256, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format = "rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def _labels(mp4):
    with av.open(io.BytesIO(mp4)) as c:
        frames = [f.to_ndarray(format = "rgb24") for f in c.decode(video = 0)]
    return [int(round(float(f.mean()) / 4)) for f in frames], frames[0].shape


def test_segments_join_in_order_on_one_canvas():
    a, b = _clip(10, base = 0), _clip(6, base = 40, size = (96, 48))
    mp4, counts = vt.compose([a, b], vt.parse_segments([{"clip": 0}, {"clip": 1}], 2), width = 64, height = 48)
    labels, shape = _labels(mp4)
    assert counts == [10, 6] and labels == list(range(10)) + list(range(40, 46))
    assert shape == (48, 64, 3)


def test_in_and_out_points_trim_on_the_frame_clock():
    seg = vt.parse_segments([{"clip": 0, "start": 0.25, "end": 0.5}], 1)
    mp4, counts = vt.compose([_clip(24)], seg, width = 64, height = 48)
    assert counts == [6] and _labels(mp4)[0] == [6, 7, 8, 9, 10, 11]


def test_a_generated_part_drops_the_frames_it_shares_with_its_neighbours():
    # A bridge repeats the 5 frames before it and the 8 after it.
    seg = vt.parse_segments([{"clip": 0, "skip_head": 5, "skip_tail": 8}], 1)
    mp4, counts = vt.compose([_clip(20)], seg, width = 64, height = 48)
    assert counts == [7] and _labels(mp4)[0] == list(range(5, 12))


@pytest.mark.parametrize("fps, count, expected", [(48, 24, 12), (12, 6, 12), (30, 30, 24)])
def test_every_clip_lands_on_the_output_frame_rate(fps, count, expected):
    mp4, counts = vt.compose([_clip(count, fps = fps)], [vt.parse_segments([{"clip": 0}], 1)[0]], width = 64, height = 48)
    assert counts == [expected]


def test_the_canvas_is_covered_not_stretched():
    from PIL import Image

    wide = Image.new("RGB", (200, 50))
    wide.paste((255, 255, 255), (0, 0, 50, 50))  # a white strip at the left edge, cropped away
    out = vt.fit(wide, 100, 50)
    assert out.size == (100, 50) and np.asarray(out)[:, :10].mean() < 5


@pytest.mark.parametrize(
    "segments, count, fragment",
    [
        ([], 1, "at least one"),
        ([{"clip": 1}], 1, "names clip 1"),
        ([{"clip": True}], 1, "names clip"),
        ([{"clip": 0, "start": -1}], 1, "start must be"),
        ([{"clip": 0, "start": 2, "end": 1}], 1, "after the in point"),
        ([{"clip": 0, "skip_head": 1.5}], 1, "whole number"),
        ([{"clip": 0}] * 33, 1, "at most 32"),
        ("nope", 1, "at least one"),
    ],
)
def test_bad_segments_are_refused(segments, count, fragment):
    with pytest.raises(vt.TimelineError, match = fragment):
        vt.parse_segments(segments, count)


def test_unreadable_trimmed_away_and_long_inputs_are_refused(monkeypatch):
    with pytest.raises(vt.TimelineError, match = "could not be read"):
        vt.compose([b"junk"], vt.parse_segments([{"clip": 0}], 1))
    with pytest.raises(vt.TimelineError, match = "no frames"):
        vt.compose([_clip(4)], vt.parse_segments([{"clip": 0, "skip_head": 4}], 1), width = 64, height = 48)
    monkeypatch.setattr(vt, "MAX_CLIP_SECONDS", 0.2)
    with pytest.raises(vt.TimelineError, match = "longer than 0.2 seconds"):
        vt.compose([_clip(24)], vt.parse_segments([{"clip": 0}], 1), width = 64, height = 48)
    monkeypatch.setattr(vt, "MAX_CLIP_SECONDS", 60.0)
    monkeypatch.setattr(vt, "MAX_TOTAL_SECONDS", 0.5)
    with pytest.raises(vt.TimelineError, match = "timeline runs longer"):
        vt.compose([_clip(24)], vt.parse_segments([{"clip": 0}], 1), width = 64, height = 48)


def test_route_joins_and_reports_frames():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import video_timeline as route

    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    client = TestClient(app)
    url = "/api/inference/video/timeline/compose"
    files = [("clips", ("a.mp4", _clip(10), "video/mp4")), ("clips", ("b.mp4", _clip(4, base = 30), "video/mp4"))]
    spec = {"segments": [{"clip": 1}, {"clip": 0, "end": 0.25}], "width": 64, "height": 48}
    res = client.post(url, data = {"spec": json.dumps(spec)}, files = files)
    assert res.status_code == 200 and res.headers["content-type"] == "video/mp4", res.text
    assert json.loads(res.headers["x-segment-frames"]) == [4, 6]
    assert _labels(res.content)[0] == [30, 31, 32, 33, 0, 1, 2, 3, 4, 5]
    for bad, fragment in (
        ({"segments": [{"clip": 0}], "width": 63}, "even"),
        ({"segments": [{"clip": 0}], "fps": 0}, "fps"),
        ({"segments": [{"clip": 5}]}, "names clip 5"),
    ):
        res = client.post(url, data = {"spec": json.dumps(bad)}, files = files)
        assert res.status_code == 400 and fragment in res.json()["detail"]
    res = client.post(url, data = {"spec": "not json"}, files = files)
    assert res.status_code == 400 and "JSON object" in res.json()["detail"]


def test_the_compose_upload_gets_the_timeline_body_cap():
    import main

    assert main._VIDEO_TIMELINE_UPLOAD_PATH == "/api/inference/video/timeline/compose"
    assert main._VIDEO_TIMELINE_UPLOAD_PATH in main._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS
    from utils.upload_limits import VIDEO_TIMELINE_UPLOAD_MAX_BYTES, upload_request_limit_bytes

    assert main._get_upload_passthrough_request_max_bytes(main._VIDEO_TIMELINE_UPLOAD_PATH) == (
        upload_request_limit_bytes(VIDEO_TIMELINE_UPLOAD_MAX_BYTES)
    )
