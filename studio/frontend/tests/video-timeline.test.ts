// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  BRIDGE_TARGET_FRAMES,
  composeSpec,
  newSeconds,
  renderOrder,
  segmentProblem,
  sharedFrames,
  TIMELINE_LENGTHS,
  trimProblem,
  trimSpec,
  type TimelineKind,
} from "../src/features/video/video-timeline.ts";

const shape = (kind: TimelineKind, hasFile = true, prompt = "p") => ({ kind, hasFile, prompt });

test("a generated part drops the frames it shares with its neighbours when joined", () => {
  assert.deepEqual(sharedFrames("real"), { head: 0, tail: 0 });
  assert.deepEqual(sharedFrames("restyle"), { head: 0, tail: 0 });
  assert.deepEqual(sharedFrames("continue"), { head: 5, tail: 0 });
  assert.deepEqual(sharedFrames("bridge"), { head: 5, tail: BRIDGE_TARGET_FRAMES });
  assert.deepEqual(composeSpec(["real", "continue", "bridge", "real"]).segments, [
    { clip: 0, skip_head: 0, skip_tail: 0 },
    { clip: 1, skip_head: 5, skip_tail: 0 },
    { clip: 2, skip_head: 5, skip_tail: 8 },
    { clip: 3, skip_head: 0, skip_tail: 0 },
  ]);
});

test("lengths are on Cosmos3's 4k + 1 lattice and leave new footage for a bridge", () => {
  for (const frames of TIMELINE_LENGTHS) {
    assert.equal((frames - 1) % 4, 0);
    assert.ok(newSeconds("bridge", frames) > 0);
  }
  assert.equal(newSeconds("continue", 93), 88 / 24);
  assert.equal(newSeconds("bridge", 93), 80 / 24);
});

test("uploaded clips render before generated parts, which render front to back", () => {
  assert.deepEqual(renderOrder(["real", "continue", "bridge", "restyle", "continue"]), [0, 3, 1, 2, 4]);
});

test("each part says what it is missing", () => {
  const parts = [shape("real"), shape("continue"), shape("bridge"), shape("real")];
  assert.deepEqual(
    parts.map((_, i) => segmentProblem(parts, i)),
    [null, null, null, null],
  );
  assert.equal(segmentProblem([shape("real", false)], 0), "Choose a clip");
  assert.match(segmentProblem([shape("restyle", true, " ")], 0) ?? "", /restyle should look/);
  assert.match(segmentProblem([shape("continue")], 0) ?? "", /part before/);
  assert.match(segmentProblem([shape("real"), shape("continue", true, "")], 1) ?? "", /What happens/i);
  assert.match(segmentProblem([shape("real"), shape("bridge")], 1) ?? "", /right after it/);
  assert.match(segmentProblem([shape("real"), shape("bridge"), shape("continue")], 1) ?? "", /right after it/);
  assert.equal(segmentProblem([shape("real"), shape("bridge"), shape("restyle")], 1), null);
});

test("trim points are checked against each other and the clip length", () => {
  assert.equal(trimProblem(0, null), null);
  assert.equal(trimProblem(1, 2.5, 6), null);
  assert.match(trimProblem(-1, null) ?? "", /in point/);
  assert.match(trimProblem(2, 1) ?? "", /after the in point/);
  assert.match(trimProblem(Number.NaN, null) ?? "", /in point/);
  assert.match(trimProblem(7, null, 6) ?? "", /only 6.0 s/);
  assert.deepEqual(trimSpec(-0.5, 3).segments, [{ clip: 0, start: 0, end: 3 }]);
});

test("the Video page offers Timeline as a page mode and sends bridge_video", async () => {
  const page = await readFile(new URL("../src/features/video/video-page.tsx", import.meta.url), "utf8");
  assert.match(page, /value: "timeline", label: "Timeline"/);
  assert.match(page, /<VideoTimelinePanel/);
  const panel = await readFile(new URL("../src/features/video/timeline-panel.tsx", import.meta.url), "utf8");
  assert.match(panel, /bridge_video: await blobToDataUrl\(after\)/);
  assert.match(panel, /continuation_keep: "last"/);
});
