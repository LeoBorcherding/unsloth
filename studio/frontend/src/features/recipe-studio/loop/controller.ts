// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LoopConfig,
  LoopDataset,
  LoopIteration,
  LoopMetric,
  LoopRecord,
  LoopStatus,
  LoopSteps,
} from "./types";

export const EVAL_LOSS_METRIC: LoopMetric = {
  id: "eval_loss",
  label: "Eval loss",
  goal: "min",
};

export const DEFAULT_LOOP_CONFIG: LoopConfig = {
  target: 0.5,
  maxIterations: 3,
  plateauPatience: 2,
  minImprovement: 0.01,
  baseRows: 200,
  rowsPerIteration: 200,
  holdoutRows: 64,
  modelName: "unsloth/SmolLM2-135M-Instruct",
  epochs: 1,
  learningRate: "2e-4",
};

export type LoopStopReason = Exclude<LoopStatus, "running" | "stopped" | "error">;

function isBetter(metric: LoopMetric, a: number, b: number, by = 0): boolean {
  return metric.goal === "min" ? a < b - by : a > b + by;
}

function best(metric: LoopMetric, scores: number[]): number | null {
  if (scores.length === 0) return null;
  return scores.reduce((acc, s) => (isBetter(metric, s, acc) ? s : acc));
}

export function bestIteration(record: LoopRecord): LoopIteration | null {
  let winner: LoopIteration | null = null;
  for (const it of record.iterations) {
    if (it.status !== "done" || it.score === null) continue;
    if (!winner || isBetter(record.metric, it.score, winner.score as number)) {
      winner = it;
    }
  }
  return winner;
}

function scoredIterations(iterations: LoopIteration[]): number[] {
  return iterations
    .filter((it) => it.status === "done" && it.score !== null)
    .map((it) => it.score as number);
}

/** Why the loop should stop before starting another iteration, or null to keep going. */
export function loopStopReason(
  iterations: LoopIteration[],
  config: LoopConfig,
  metric: LoopMetric,
): LoopStopReason | null {
  const scores = scoredIterations(iterations);
  const top = best(metric, scores);
  if (top !== null && !isBetter(metric, config.target, top)) {
    return "target_reached";
  }
  const patience = config.plateauPatience;
  if (patience > 0 && scores.length > patience) {
    const before = best(metric, scores.slice(0, -patience)) as number;
    const recent = best(metric, scores.slice(-patience)) as number;
    if (!isBetter(metric, recent, before, config.minImprovement)) {
      return "plateau";
    }
  }
  if (scores.length >= config.maxIterations) {
    return "max_iterations";
  }
  return null;
}

/** Iteration 1 draws the base rows; every later one adds a fresh draw of rowsPerIteration. */
export function rowsForIteration(config: LoopConfig, index: number): number {
  return index <= 1 ? config.baseRows : config.rowsPerIteration;
}

export function createLoopRecord(input: {
  id: string;
  recipeId: string;
  config: LoopConfig;
  metric: LoopMetric;
  now: number;
}): LoopRecord {
  return {
    id: input.id,
    recipeId: input.recipeId,
    createdAt: input.now,
    updatedAt: input.now,
    status: "running",
    metric: input.metric,
    config: input.config,
    holdout: null,
    iterations: [],
    error: null,
  };
}

/** Continue a saved loop: finished iterations stay, an unfinished one is redone. */
export function resumeLoopRecord(record: LoopRecord, now: number): LoopRecord {
  return {
    ...record,
    status: "running",
    error: null,
    updatedAt: now,
    iterations: record.iterations.filter((it) => it.status === "done"),
  };
}

function isAbort(error: unknown, signal: AbortSignal): boolean {
  return (
    signal.aborted ||
    (error instanceof Error && error.name === "AbortError")
  );
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Drives iterations until a stop condition, saving after every step so the history survives a
 * reload. Each iteration trains on every dataset version generated so far.
 */
export async function runLoop(input: {
  record: LoopRecord;
  steps: LoopSteps;
  signal: AbortSignal;
  save: (record: LoopRecord) => void;
  now?: () => number;
}): Promise<LoopRecord> {
  const { steps, signal, save } = input;
  const now = input.now ?? Date.now;
  let record = input.record;
  const commit = (patch: Partial<LoopRecord>): void => {
    record = { ...record, ...patch, updatedAt: now() };
    save(record);
  };
  const setIteration = (patch: Partial<LoopIteration>): void => {
    const iterations = record.iterations.slice();
    iterations[iterations.length - 1] = {
      ...(iterations[iterations.length - 1] as LoopIteration),
      ...patch,
    };
    commit({ iterations });
  };

  try {
    if (!record.holdout && record.config.holdoutRows > 0) {
      const holdout = await steps.generate({
        iteration: 0,
        rows: record.config.holdoutRows,
        purpose: "holdout",
        signal,
      });
      commit({ holdout });
    }
    while (true) {
      if (signal.aborted) {
        commit({ status: "stopped" });
        return record;
      }
      const reason = loopStopReason(
        record.iterations,
        record.config,
        record.metric,
      );
      if (reason) {
        commit({ status: reason });
        return record;
      }
      const index = record.iterations.length + 1;
      commit({
        iterations: [
          ...record.iterations,
          {
            index,
            status: "generating",
            dataset: null,
            run: null,
            score: null,
            error: null,
            startedAt: now(),
            finishedAt: null,
          },
        ],
      });
      const dataset = await steps.generate({
        iteration: index,
        rows: rowsForIteration(record.config, index),
        purpose: "train",
        signal,
      });
      setIteration({ dataset, status: "training" });
      const datasets = record.iterations
        .map((it) => it.dataset)
        .filter((d): d is LoopDataset => d !== null);
      const run = await steps.train({
        iteration: index,
        datasets,
        holdout: record.holdout,
        config: record.config,
        signal,
      });
      setIteration({ run, status: "scoring" });
      const score = await steps.score({
        iteration: index,
        run,
        holdout: record.holdout,
        signal,
      });
      setIteration({ score, status: "done", finishedAt: now() });
    }
  } catch (error) {
    const aborted = isAbort(error, signal);
    const message = aborted ? "Stopped." : errorText(error);
    const last = record.iterations[record.iterations.length - 1];
    if (last && last.status !== "done") {
      setIteration({ status: "error", error: message, finishedAt: now() });
    }
    commit(
      aborted
        ? { status: "stopped" }
        : { status: "error", error: message },
    );
    return record;
  }
}
