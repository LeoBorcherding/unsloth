// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createResidentDriftWatch } from "../src/features/api-monitor/resident-drift.ts";
import {
  requestResidentStatusRefresh,
  subscribeResidentStatusRefresh,
} from "../src/features/hub/lib/resident-status-refresh.ts";
import { fakeTargets, readSrc } from "./helpers/kit.ts";

const MODEL = "unsloth/Qwen3-8B-GGUF";
const loadedAt = (loadedContextLength: number | null, modelLoading = false) => ({
  residentCheckpoint: MODEL,
  loadedContextLength,
  modelLoading,
});

test("an API reload at a new context re-reads status under a focused tab", () => {
  const targets = fakeTargets();
  let reads = 0;
  const unsubscribe = subscribeResidentStatusRefresh(() => {
    reads += 1;
  }, targets);
  const drifted = createResidentDriftWatch();
  // UI loaded at 4096, then POST /api/inference/load reloaded it at 16384; no focus or visibility event.
  const monitor = { active_model: MODEL, context_length: 16384 };
  if (drifted(monitor, loadedAt(4096))) requestResidentStatusRefresh();
  assert.equal(reads, 1);
  // Same stale pair again: one re-read, not one per poll.
  if (drifted(monitor, loadedAt(4096))) requestResidentStatusRefresh();
  assert.equal(reads, 1);
  unsubscribe();
  requestResidentStatusRefresh();
  assert.equal(reads, 1);
});

test("the drift watch stays quiet when in sync, loading or unknown, and re-arms after a sync", () => {
  const drifted = createResidentDriftWatch();
  assert.equal(drifted({ active_model: MODEL, context_length: 4096 }, loadedAt(4096)), false);
  assert.equal(drifted({ active_model: MODEL, context_length: 16384 }, loadedAt(4096, true)), false);
  assert.equal(drifted({ active_model: null, context_length: 16384 }, loadedAt(4096)), false);
  assert.equal(drifted({ active_model: MODEL, context_length: null }, loadedAt(4096)), false);
  assert.equal(drifted({ active_model: MODEL, context_length: 16384 }, loadedAt(null)), false);
  assert.equal(
    drifted(
      { active_model: MODEL, context_length: 16384 },
      { ...loadedAt(4096), residentCheckpoint: null },
    ),
    false,
  );
  assert.equal(drifted({ active_model: MODEL, context_length: 8192 }, loadedAt(4096)), true);
  assert.equal(drifted({ active_model: MODEL, context_length: 8192 }, loadedAt(8192)), false);
  assert.equal(drifted({ active_model: MODEL, context_length: 4096 }, loadedAt(8192)), true);
});

test("a drift the refresh did not clear is retried after a while, not every poll", () => {
  let clock = 0;
  const drifted = createResidentDriftWatch(() => clock);
  const monitor = { active_model: MODEL, context_length: 16384 };
  assert.equal(drifted(monitor, loadedAt(4096)), true);
  clock = 1_500;
  assert.equal(drifted(monitor, loadedAt(4096)), false);
  // The status read failed, so the store still says 4096.
  clock = 15_000;
  assert.equal(drifted(monitor, loadedAt(4096)), true);
  clock = 16_500;
  assert.equal(drifted(monitor, loadedAt(4096)), false);
});

test("a live value that keeps changing while drifted still waits out the retry gap", () => {
  let clock = 0;
  const drifted = createResidentDriftWatch(() => clock);
  assert.equal(drifted({ active_model: MODEL, context_length: 16384 }, loadedAt(4096)), true);
  clock = 1_500;
  assert.equal(drifted({ active_model: MODEL, context_length: 8192 }, loadedAt(4096)), false);
  clock = 3_000;
  assert.equal(drifted({ active_model: MODEL, context_length: 16384 }, loadedAt(4096)), false);
  clock = 15_000;
  assert.equal(drifted({ active_model: MODEL, context_length: 8192 }, loadedAt(4096)), true);
});

test("both monitor polls feed the drift watch", () => {
  for (const file of [
    "features/api-monitor/use-api-monitor.ts",
    "features/api-monitor/api-monitor-overlay.tsx",
  ]) {
    const source = readSrc(file);
    assert.match(source, /createResidentDriftWatch\(\)/, file);
    assert.match(
      source,
      /\(next, useChatRuntimeStore\.getState\(\)\)\) \{\s*requestResidentStatusRefresh\(\);/,
      file,
    );
  }
});
