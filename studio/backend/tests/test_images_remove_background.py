# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Remove background on the Images page and over the API: the cut-out from a SAM 3 mask (alpha, grow, feather, trim),
the default subject, and the route (upload or gallery source, saved or returned, errors). The segmenter is stubbed."""

from __future__ import annotations

import base64
import io
import threading

import pytest

from core.inference import diffusion_segment as seg

URL = "/api/inference/images/remove-background"


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG")
    return buffer.getvalue()


def _data_url(image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png(image)).decode()


def _open(data_url: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def _source(width = 64, height = 48):
    from PIL import Image
    return Image.new("RGB", (width, height), (200, 30, 30))


def _box_mask(size, box):
    from PIL import Image, ImageDraw

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill = 255)
    return mask


class _Stub:
    """Answers every segment with ``box`` filled (or nothing found when ``box`` is None)."""

    def __init__(self, box = (16, 8, 39, 31)):
        self.box = box
        self.calls = []

    def segment(self, image, prompt, *, threshold, allow_gpu):
        self.calls.append((image.mode, image.size, prompt, threshold))
        instances = (
            [] if self.box is None else [seg.Instance(prompt, 0.9, tuple(map(float, self.box)))]
        )
        from PIL import Image

        mask = _box_mask(image.size, self.box) if self.box else Image.new("L", image.size, 0)
        return seg.SegmentResult(
            mask_png = _png(mask),
            width = image.size[0],
            height = image.size[1],
            instances = instances,
            concepts = [{"text": prompt, "max_detections": 1, "found": len(instances)}],
        )


@pytest.fixture
def stub(monkeypatch):
    s = _Stub()
    monkeypatch.setattr(seg, "get_segmenter", lambda: s)
    return s


# ── the cut-out ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_cutout_alpha_is_the_mask():
    image = _source()
    cut, alpha = seg.apply_cutout(image, _box_mask(image.size, (16, 8, 39, 31)))
    assert cut.mode == "RGBA" and cut.size == image.size and alpha.size == image.size
    assert cut.getpixel((20, 20)) == (200, 30, 30, 255)
    assert cut.getpixel((2, 2))[3] == 0
    assert alpha.getbbox() == (16, 8, 40, 32)


def test_cutout_mask_at_another_size_is_resized():
    image = _source(64, 48)
    cut, alpha = seg.apply_cutout(image, _box_mask((32, 24), (8, 4, 19, 15)))
    assert cut.size == (64, 48) and alpha.size == (64, 48)
    assert cut.getpixel((32, 20))[3] == 255 and cut.getpixel((2, 2))[3] == 0


def test_grow_shrink_feather_trim():
    image = _source()
    mask = _box_mask(image.size, (16, 8, 39, 31))
    _, grown = seg.apply_cutout(image, mask, grow = 4)
    assert grown.getbbox() == (12, 4, 44, 36)
    _, shrunk = seg.apply_cutout(image, mask, grow = -4)
    assert shrunk.getbbox() == (20, 12, 36, 28)
    _, soft = seg.apply_cutout(image, mask, feather = 3)
    assert 0 < soft.getpixel((15, 20)) < 255 and soft.getpixel((28, 20)) == 255
    cut, alpha = seg.apply_cutout(image, mask, trim = True)
    assert cut.size == alpha.size == (24, 24)
    with pytest.raises(ValueError):
        seg.apply_cutout(image, mask, grow = 65)
    with pytest.raises(ValueError):
        seg.apply_cutout(image, mask, feather = -1)


def test_dilate_matches_a_max_filter():
    import numpy as np
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(0)
    keep = rng.random((40, 50)) > 0.97
    radius = 3
    expected = (
        np.asarray(
            Image.fromarray(keep.astype(np.uint8) * 255).filter(
                ImageFilter.MaxFilter(2 * radius + 1)
            )
        )
        > 0
    )
    assert (seg._dilate(keep, radius) == expected).all()


def test_existing_alpha_is_kept():
    from PIL import Image

    image = Image.new("RGBA", (64, 48), (10, 20, 30, 255))
    image.paste((10, 20, 30, 0), (0, 0, 20, 48))
    cut, _ = seg.apply_cutout(image, Image.new("L", (64, 48), 255))
    assert cut.getpixel((5, 5))[3] == 0 and cut.getpixel((40, 5))[3] == 255


# ── remove_background ───────────────────────────────────────────────────────────────────────────────────────────────
def test_default_subject_and_flattened_input(stub):
    from PIL import Image

    image = Image.new("RGBA", (64, 48), (200, 30, 30, 255))
    cut, alpha, result = seg.remove_background(image, "  ")
    assert stub.calls == [("RGB", (64, 48), seg.DEFAULT_SUBJECT, seg.DEFAULT_THRESHOLD)]
    assert cut.size == (64, 48) and result.instances
    seg.remove_background(image, "car:2", threshold = 0.3)
    assert stub.calls[-1][2:] == ("car:2", 0.3)


def test_nothing_found_raises(stub):
    stub.box = None
    with pytest.raises(seg.NothingFoundError, match = "elephant"):
        seg.remove_background(_source(), "elephant")


# ── route ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class _Engine:
    def __init__(self):
        self._generate_lock = threading.Lock()
        self._loading = None


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import core.inference.diffusion_engine_router as engine_router
    import routes.inference as inference_routes
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from core.inference import image_gallery
    from routes.inference import studio_router

    engine = _Engine()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: engine)
    monkeypatch.setattr(image_gallery, "gallery_dir", lambda: tmp_path)
    monkeypatch.setattr(inference_routes, "_training_is_active", lambda: True)
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app), engine


def test_route_upload_unsaved_returns_the_png(client, stub):
    http, _ = client
    r = http.post(URL, json = {"image": _data_url(_source()), "prompt": "car", "save": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] is None and body["prompt"] == "car"
    assert (body["width"], body["height"]) == (64, 48) and body["instances"][0]["prompt"] == "car"
    cut = _open(body["b64_png"])
    assert cut.mode == "RGBA" and cut.size == (64, 48)
    assert cut.getpixel((2, 2))[3] == 0 and cut.getpixel((20, 20))[3] == 255
    assert _open(body["mask"]).size == (64, 48)


def test_route_saves_to_gallery_and_carries_the_recipe(client, stub, tmp_path):
    from core.inference import image_gallery

    http, _ = client
    r = http.post(URL, json = {"image": _data_url(_source()), "trim": True})
    assert r.status_code == 200, r.text
    saved = r.json()["image"]
    assert saved["workflow"] == "remove_background" and (saved["width"], saved["height"]) == (
        24,
        24,
    )
    assert r.json()["prompt"] == seg.DEFAULT_SUBJECT and r.json()["b64_png"] is None
    path, _ = image_gallery.owned_image(saved["id"])
    from PIL import Image

    with Image.open(path) as im:
        assert im.mode == "RGBA" and im.size == (24, 24)

    source = image_gallery.save(
        _source(),
        {
            "prompt": "a red car",
            "width": 64,
            "height": 48,
            "steps": 8,
            "guidance": 1.0,
            "seed": 42,
            "model": "unsloth/Z-Image-Turbo-GGUF",
            "created_at": 1.0,
        },
    )
    r = http.post(URL, json = {"image_id": source["id"], "prompt": "car"})
    assert r.status_code == 200, r.text
    saved = r.json()["image"]
    assert saved["id"] != source["id"]
    assert (saved["prompt"], saved["seed"], saved["model"]) == (
        "a red car",
        42,
        "unsloth/Z-Image-Turbo-GGUF",
    )
    assert saved["workflow"] == "remove_background"
    assert stub.calls[-1][:3] == ("RGB", (64, 48), "car")


def test_route_errors(client, stub, monkeypatch):
    http, engine = client
    image = _data_url(_source())
    assert http.post(URL, json = {"image_id": "missing"}).status_code == 404
    assert http.post(URL, json = {"image_id": "../x"}).status_code == 404
    assert http.post(URL, json = {}).status_code == 422
    assert http.post(URL, json = {"image": image, "image_id": "x"}).status_code == 422
    assert http.post(URL, json = {"image": image, "grow": 99}).status_code == 422
    assert http.post(URL, json = {"image": "!!!"}).status_code == 400
    assert http.post(URL, json = {"image": image, "prompt": " , , "}).status_code == 400
    stub.box = None
    r = http.post(URL, json = {"image": image, "prompt": "elephant"})
    assert r.status_code == 422 and "elephant" in r.json()["detail"]
    stub.box = (1, 1, 5, 5)
    with engine._generate_lock:
        r = http.post(URL, json = {"image": image})
    assert r.status_code == 409 and "generating" in r.json()["detail"]

    def not_downloaded():
        raise seg.SegmenterNotDownloadedError("The SAM 3.1 segmenter is not downloaded.")

    monkeypatch.setattr(seg, "get_segmenter", lambda: _raising(not_downloaded))
    r = http.post(URL, json = {"image": image})
    assert r.status_code == 409 and "not downloaded" in r.json()["detail"]


def _raising(raise_fn):
    class _Raising:
        def segment(self, *args, **kwargs):
            raise_fn()

    return _Raising()
