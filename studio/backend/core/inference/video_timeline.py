# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video timeline: trim clips and join them into one MP4 on a shared canvas and frame clock. Runs no model; the
timeline page renders each generated part with the existing video and restyle routes and joins them here."""

from __future__ import annotations

import io
from typing import Any, Iterable, Iterator, Optional

WIDTH, HEIGHT, FPS = 832, 480, 24
MAX_SEGMENTS = 32
MAX_CLIP_SECONDS = 60.0
MAX_TOTAL_SECONDS = 300.0
MAX_SIDE = 4096


class TimelineError(ValueError):
    """A bad timeline request; the route answers 400 with the message."""


def parse_segments(raw: Any, clip_count: int) -> list[dict[str, Any]]:
    """Validate the segment list: ``clip`` index, optional ``start`` / ``end`` seconds, ``skip_head`` /
    ``skip_tail`` frames dropped after the trim (a generated part repeats the frames it was conditioned on)."""
    if not isinstance(raw, list) or not raw:
        raise TimelineError("Send at least one segment.")
    if len(raw) > MAX_SEGMENTS:
        raise TimelineError(f"A timeline holds at most {MAX_SEGMENTS} segments.")
    out = []
    for i, seg in enumerate(raw, 1):
        if not isinstance(seg, dict):
            raise TimelineError(f"Segment {i} must be an object.")
        clip = seg.get("clip")
        if isinstance(clip, bool) or not isinstance(clip, int) or not 0 <= clip < clip_count:
            raise TimelineError(
                f"Segment {i} names clip {clip!r}, but {clip_count} clip(s) were sent."
            )
        start, end = seg.get("start", 0.0) or 0.0, seg.get("end")
        for name, value in (("start", start), ("end", end)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            ):
                raise TimelineError(f"Segment {i}: {name} must be a number of seconds, 0 or more.")
        if end is not None and end <= start:
            raise TimelineError(f"Segment {i}: the out point must come after the in point.")
        skips = {}
        for name in ("skip_head", "skip_tail"):
            value = seg.get(name, 0) or 0
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TimelineError(
                    f"Segment {i}: {name} must be a whole number of frames, 0 or more."
                )
            skips[name] = value
        out.append(
            {
                "clip": clip,
                "start": float(start),
                "end": None if end is None else float(end),
                **skips,
            }
        )
    return out


def fit(image: Any, width: int, height: int) -> Any:
    """Cover the canvas and centre-crop, never stretch."""
    from PIL import Image, ImageOps

    image = image.convert("RGB")
    if image.size == (width, height):
        return image
    return ImageOps.fit(image, (width, height), Image.Resampling.BICUBIC)


def clip_frames(
    data: bytes,
    *,
    start: float = 0.0,
    end: Optional[float] = None,
    fps: int = FPS,
) -> Iterator[Any]:
    """Frames of ``data`` from ``start`` to ``end`` seconds on an ``fps`` clock, picked by presentation time (held when
    the source is slower, dropped when faster), as decoded ``av.VideoFrame`` objects."""
    import av

    try:
        container = av.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 -- any open failure is a bad upload
        raise TimelineError("A clip could not be read as a video.") from exc
    with container:
        if not container.streams.video:
            raise TimelineError("A clip has no video track.")
        stream = container.streams.video[0]
        rate = float(stream.average_rate or stream.guessed_rate or fps) or float(fps)
        origin: Optional[float] = None
        held, tick = None, 0
        try:
            for index, frame in enumerate(container.decode(stream)):
                if max(frame.width, frame.height) > MAX_SIDE:
                    raise TimelineError(
                        f"A clip is {frame.width}x{frame.height}; the limit is {MAX_SIDE} px a side."
                    )
                t = frame.time if frame.time is not None else index / rate
                if origin is None:
                    origin = t
                t -= origin
                if t > MAX_CLIP_SECONDS + 1e-6:
                    raise TimelineError(
                        f"A clip runs longer than {MAX_CLIP_SECONDS:g} seconds. Trim it first."
                    )
                while held is not None and start + tick / fps < t - 1e-6:
                    if end is not None and start + tick / fps >= end - 1e-6:
                        return
                    yield held
                    tick += 1
                held = frame
            # The last frame lasts one nominal frame.
            if held is not None:
                stop = (t + 1 / rate) if end is None else min(end, t + 1 / rate)
                while start + tick / fps < stop - 1e-6:
                    yield held
                    tick += 1
        except av.error.FFmpegError as exc:
            raise TimelineError("A clip could not be decoded.") from exc


def _segment_frames(clips: list[bytes], seg: dict[str, Any], fps: int) -> Iterator[Any]:
    head, tail = seg["skip_head"], seg["skip_tail"]
    buffer: list[Any] = []
    for index, frame in enumerate(
        clip_frames(clips[seg["clip"]], start = seg["start"], end = seg["end"], fps = fps)
    ):
        if index < head:
            continue
        buffer.append(frame)
        if len(buffer) > tail:
            yield buffer.pop(0)


def compose(
    clips: list[bytes],
    segments: Iterable[dict[str, Any]],
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> tuple[bytes, list[int]]:
    """One H.264 MP4 of every segment in order, and the frame count each contributed."""
    import av

    segments = list(segments)
    counts: list[int] = []
    limit = int(MAX_TOTAL_SECONDS * fps)
    total = 0
    buf = io.BytesIO()
    with av.open(buf, "w", format = "mp4") as out:
        stream = out.add_stream("libx264", rate = fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        for seg in segments:
            n = 0
            last_id, last_fitted = None, None
            for frame in _segment_frames(clips, seg, fps):
                if total >= limit:
                    raise TimelineError(
                        f"The timeline runs longer than {MAX_TOTAL_SECONDS:g} seconds."
                    )
                # A held source frame is fitted once.
                if frame is not last_id:
                    last_id, last_fitted = frame, fit(frame.to_image(), width, height)
                for packet in stream.encode(av.VideoFrame.from_image(last_fitted)):
                    out.mux(packet)
                n += 1
                total += 1
            counts.append(n)
        if total == 0:
            raise TimelineError("The timeline has no frames: every segment was trimmed away.")
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue(), counts
