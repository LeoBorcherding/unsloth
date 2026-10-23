// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FeedbackSeedRow, LoopFailedExample } from "./types";

const BASE_COLUMNS = [
  "origin",
  "failed_prompt",
  "failed_reference",
  "failed_output",
] as const;

function cell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  return typeof value === "string" ? value : JSON.stringify(value);
}

/**
 * Columns the Benchmark failures seed hands the recipe: `origin` ("failure" or "fresh"), the failed
 * prompt, reference and answer, and `failed_<column>` for each of the recipe's own output columns.
 */
export function feedbackSeedColumns(recipeColumns: string[]): string[] {
  const own = new Set<string>(BASE_COLUMNS);
  const extra = recipeColumns
    .map((column) => `failed_${column}`)
    .filter((column) => !own.has(column));
  return [...BASE_COLUMNS, ...Array.from(new Set(extra))];
}

/**
 * One seed row per row the round generates: `failureShare` of them repeat the last round's failed
 * examples (cycling when there are fewer failures than slots), the rest are blank "fresh" rows.
 */
export function buildFeedbackSeedRows(input: {
  failures: LoopFailedExample[];
  rows: number;
  failureShare: number;
  recipeColumns: string[];
}): FeedbackSeedRow[] {
  const columns = feedbackSeedColumns(input.recipeColumns);
  const blank = (): FeedbackSeedRow => {
    const row = { origin: "fresh" } as FeedbackSeedRow;
    for (const column of columns) {
      if (column !== "origin") {
        row[column] = "";
      }
    }
    return row;
  };
  const total = Math.max(0, Math.floor(input.rows));
  const share = Math.min(1, Math.max(0, input.failureShare));
  const fromFailures =
    input.failures.length > 0 ? Math.min(total, Math.round(total * share)) : 0;
  const rows: FeedbackSeedRow[] = [];
  for (let i = 0; i < fromFailures; i += 1) {
    const failure = input.failures[i % input.failures.length] as LoopFailedExample;
    const row = blank();
    row.origin = "failure";
    row.failed_prompt = failure.prompt;
    row.failed_reference = failure.reference ?? "";
    row.failed_output = failure.output ?? "";
    for (const column of input.recipeColumns) {
      const key = `failed_${column}`;
      if (key in row && !(BASE_COLUMNS as readonly string[]).includes(key)) {
        row[key] = cell(failure.row[column]);
      }
    }
    rows.push(row);
  }
  while (rows.length < total) {
    rows.push(blank());
  }
  return rows;
}

export function feedbackSeedToJsonl(rows: FeedbackSeedRow[]): string {
  return rows.map((row) => JSON.stringify(row)).join("\n");
}
