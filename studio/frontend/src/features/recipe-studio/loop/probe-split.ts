// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BenchmarkResult } from "../types";
import type { LoopFailedExample, LoopScore, LoopSplitScore } from "./types";

/** Tags each held-out row sent to the benchmark, so a failure can be traced to its split. */
export const HOLDOUT_ROW_ID_KEY = "__loop_holdout_row_id";

export const PROBE_SHARE = 0.5;

export type HoldoutRow = { id: number; row: Record<string, unknown> };

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Splits the held-out rows into a probe split (its failures feed the next round) and a test split
 * (never fed back; the loop's score). Seeded by row id, so every round gets the same split.
 */
export function splitProbeTest(
  rows: HoldoutRow[],
  seed: number,
  probeShare: number = PROBE_SHARE,
): { probe: HoldoutRow[]; test: HoldoutRow[] } {
  const ordered = [...rows].sort((a, b) => a.id - b.id);
  const random = mulberry32(seed);
  for (let i = ordered.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [ordered[i], ordered[j]] = [ordered[j], ordered[i]];
  }
  const probeCount = Math.min(
    ordered.length - 1,
    Math.max(1, Math.round(ordered.length * probeShare)),
  );
  return { probe: ordered.slice(0, probeCount), test: ordered.slice(probeCount) };
}

export function taggedRows(rows: HoldoutRow[]): Record<string, unknown>[] {
  return rows.map(({ id, row }) => ({ ...row, [HOLDOUT_ROW_ID_KEY]: id }));
}

function splitScore(result: BenchmarkResult): LoopSplitScore {
  const { candidate } = result;
  if (candidate.score === null || candidate.scored === 0) {
    throw new Error(
      candidate.errors > 0
        ? `Every benchmark row errored (${candidate.errors}). Check that the adapter loaded in Chat.`
        : "The benchmark scored no rows.",
    );
  }
  return { score: candidate.score, passed: candidate.passed, scored: candidate.scored };
}

/**
 * A round's score from its two benchmark runs. Only failures on probe rows go forward: a row that
 * is not tagged with a probe row id is dropped, so nothing from the test split reaches the seed.
 */
export function buildRoundScore(input: {
  test: BenchmarkResult;
  probe: BenchmarkResult;
  probeRowIds: Iterable<number>;
  baseline: number | null;
}): LoopScore {
  const test = splitScore(input.test);
  const probe = splitScore(input.probe);
  const allowed = new Set(input.probeRowIds);
  const failedExamples: LoopFailedExample[] = [];
  for (const example of input.probe.failed_examples) {
    const { [HOLDOUT_ROW_ID_KEY]: rawId, ...row } = example.row;
    const rowId = typeof rawId === "number" ? rawId : Number.NaN;
    if (!allowed.has(rowId)) {
      continue;
    }
    failedExamples.push({
      prompt: example.prompt,
      reference: example.reference,
      output: example.output,
      row,
      rowId,
    });
  }
  return {
    ...test,
    baseline: input.test.baseline?.score ?? input.baseline,
    probe,
    failedExamples,
    failedTotal: input.probe.failed_total,
  };
}
