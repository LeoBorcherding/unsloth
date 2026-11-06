// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  KEEP_GROUPS,
  keepNewest,
  shownIndex,
} from "../src/features/studio/sections/charts/rl-sample-groups.ts";

const range = (from: number, to: number) =>
  Array.from({ length: to - from + 1 }, (_, i) => ({ seq: from + i }));

test("a sample being browsed stays put when older groups are trimmed", () => {
  let groups = range(1, KEEP_GROUPS);
  const selected = groups[4].seq;
  groups = keepNewest(groups, range(KEEP_GROUPS + 1, KEEP_GROUPS + 2));
  assert.equal(groups.length, KEEP_GROUPS);
  assert.equal(groups[shownIndex(groups, selected)].seq, selected);
});

test("no selection follows the newest group; a trimmed one falls to the oldest kept", () => {
  const groups = keepNewest(
    range(1, KEEP_GROUPS),
    range(KEEP_GROUPS + 1, KEEP_GROUPS + 3),
  );
  assert.equal(shownIndex(groups, null), groups.length - 1);
  assert.equal(shownIndex(groups, 1), 0);
});

test("a browsed sample stays put across many polls that each trim", () => {
  // Replays a long run: the user parks on seq 5, then every poll adds two groups.
  let groups = range(1, KEEP_GROUPS);
  const selected = 5;
  for (let poll = 0; poll < 40; poll++) {
    const next = groups[groups.length - 1].seq + 1;
    groups = keepNewest(groups, range(next, next + 1), selected);
    assert.equal(
      groups[shownIndex(groups, selected)].seq,
      selected,
      `poll ${poll}`,
    );
    assert.ok(groups.length <= KEEP_GROUPS + 1);
  }
  // Back to following the newest: the parked group is let go on the next trim.
  groups = keepNewest(groups, range(1000, 1001), null);
  assert.ok(!groups.some((g) => g.seq === selected));
});

test("a poll that overlaps the last one adds no group twice", () => {
  const held = range(1, 3);
  const groups = keepNewest(held, range(2, 5));
  assert.deepEqual(
    groups.map((g) => g.seq),
    [1, 2, 3, 4, 5],
  );
});

test("the sample card clears only for a new run, not when the run ends", async () => {
  // No DOM in these tests: the reset effect's dependency list is the behaviour.
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL(
      "../src/features/studio/sections/charts/rl-samples.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const reset =
    /setGroups\(\[\]\);[\s\S]*?lastSeq\.current = 0;\s*\}, \[([^\]]*)\]\);/.exec(
      src,
    );
  assert.equal(reset?.[1], "jobId");
});
