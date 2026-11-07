// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  BRIDGE_TARGET_FRAMES,
  composeSpec,
  enginesNeeded,
  kindInfo,
  newSeconds,
  renderOrder,
  SEGMENT_KINDS,
  SEGMENT_LENGTHS,
  segmentProblem,
  trimProblem,
  trimSpec,
  type SegmentKind,
} from "../src/features/video-editing/segments.ts";

const shape = (kind: SegmentKind, hasFile = true, prompt = "p") => ({ kind, hasFile, prompt });
const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");

test("a generated segment drops the frames it shares with its neighbours when joined", () => {
  assert.deepEqual(kindInfo("real").shared, { head: 0, tail: 0 });
  assert.deepEqual(kindInfo("restyle").shared, { head: 0, tail: 0 });
  assert.deepEqual(kindInfo("continue").shared, { head: 5, tail: 0 });
  assert.deepEqual(kindInfo("bridge").shared, { head: 5, tail: BRIDGE_TARGET_FRAMES });
  assert.deepEqual(composeSpec(["real", "continue", "bridge", "real"]).segments, [
    { clip: 0, skip_head: 0, skip_tail: 0 },
    { clip: 1, skip_head: 5, skip_tail: 0 },
    { clip: 2, skip_head: 5, skip_tail: 8 },
    { clip: 3, skip_head: 0, skip_tail: 0 },
  ]);
});

test("each kind names the engine that renders it, so another family can plug in as an engine", () => {
  assert.deepEqual(
    Object.fromEntries(SEGMENT_KINDS.map((k) => [k.value, k.engine])),
    { real: "upload", restyle: "cosmos3-sim-transfer", continue: "cosmos3-nano", bridge: "cosmos3-nano" },
  );
  assert.deepEqual([...enginesNeeded(["real", "real"])], ["upload"]);
  assert.ok(enginesNeeded(["real", "bridge", "real"]).has("cosmos3-nano"));
});

test("lengths are on Cosmos3's 4k + 1 lattice and leave new footage for a bridge", () => {
  for (const frames of SEGMENT_LENGTHS) {
    assert.equal((frames - 1) % 4, 0);
    assert.ok(newSeconds("bridge", frames) > 0);
  }
  assert.equal(newSeconds("continue", 93), 88 / 24);
  assert.equal(newSeconds("bridge", 93), 80 / 24);
});

test("uploaded clips render before generated segments, which render front to back", () => {
  assert.deepEqual(renderOrder(["real", "continue", "bridge", "restyle", "continue"]), [0, 3, 1, 2, 4]);
});

test("each segment says what it is missing", () => {
  const list = [shape("real"), shape("continue"), shape("bridge"), shape("real")];
  assert.deepEqual(
    list.map((_, i) => segmentProblem(list, i)),
    [null, null, null, null],
  );
  assert.equal(segmentProblem([shape("real", false)], 0), "Choose a clip");
  assert.equal(segmentProblem([shape("restyle", true, " ")], 0), "Describe the new look");
  assert.match(segmentProblem([shape("continue")], 0) ?? "", /segment before/);
  assert.equal(segmentProblem([shape("real"), shape("continue", true, "")], 1), "Describe what happens");
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

test("Video editing is its own page next to Video, and a bridge sends bridge_video", async () => {
  const router = await read("../src/app/router.tsx");
  assert.match(router, /videoRoute,\s*videoEditingRoute,/);
  const route = await read("../src/app/routes/video-editing.tsx");
  assert.match(route, /path: "\/video-editing"/);
  const nav = await read("../src/features/settings/stores/appearance-custom-store.ts");
  assert.match(nav, /"video",\s*"editing",/);
  const sidebar = await read("../src/components/app-sidebar.tsx");
  assert.match(sidebar, /editing: \{\s*icon: VideoReplayIcon,/);
  const store = await read("../src/features/video-editing/editing-store.ts");
  assert.match(store, /bridge_video: await blobToDataUrl\(after\)/);
  assert.match(store, /continuation_keep: "last"/);
  // A Cancel during the start request still stops the GPU job it started.
  assert.match(store, /await startedJob\(cancelVideoGeneration\)/);
  assert.match(store, /await startedJob\(cancelVideoRestyle\)/);
  const videoPage = await read("../src/features/video/video-page.tsx");
  assert.doesNotMatch(videoPage, /Timeline/);
});
