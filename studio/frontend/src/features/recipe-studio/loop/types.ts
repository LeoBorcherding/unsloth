// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LoopMetric = {
  id: string;
  label: string;
  goal: "min" | "max";
};

export type LoopConfig = {
  target: number;
  maxIterations: number;
  // 0 turns plateau detection off.
  plateauPatience: number;
  minImprovement: number;
  baseRows: number;
  rowsPerIteration: number;
  // Generated once and scored against every iteration; 0 lets training split its own eval rows.
  holdoutRows: number;
  modelName: string;
  // Epochs over every row so far, so a bigger dataset also trains longer.
  epochs: number;
  learningRate: string;
};

export type LoopDataset = {
  // Recipe dataset folder name (recipe_<run>); the version an iteration trained on.
  version: string;
  rows: number | null;
  executionId: string | null;
};

export type LoopTrainResult = {
  runId: string;
  adapter: string | null;
};

export type LoopIterationStatus =
  | "generating"
  | "training"
  | "scoring"
  | "done"
  | "error";

export type LoopIteration = {
  index: number;
  status: LoopIterationStatus;
  dataset: LoopDataset | null;
  run: LoopTrainResult | null;
  score: number | null;
  error: string | null;
  startedAt: number;
  finishedAt: number | null;
};

export type LoopStatus =
  | "running"
  | "target_reached"
  | "plateau"
  | "max_iterations"
  | "stopped"
  | "error";

export type LoopRecord = {
  id: string;
  recipeId: string;
  createdAt: number;
  updatedAt: number;
  status: LoopStatus;
  metric: LoopMetric;
  config: LoopConfig;
  holdout: LoopDataset | null;
  iterations: LoopIteration[];
  error: string | null;
};

type StepInput = { iteration: number; signal: AbortSignal };

/**
 * The three steps one iteration chains. The Train and Benchmark cards plug in by supplying
 * `train`, or `score` with its `metric`; the controller only sees this interface.
 */
export type LoopSteps = {
  metric: LoopMetric;
  generate: (
    input: StepInput & { rows: number; purpose: "train" | "holdout" },
  ) => Promise<LoopDataset>;
  train: (
    input: StepInput & {
      datasets: LoopDataset[];
      holdout: LoopDataset | null;
      config: LoopConfig;
    },
  ) => Promise<LoopTrainResult>;
  score: (
    input: StepInput & { run: LoopTrainResult; holdout: LoopDataset | null },
  ) => Promise<number>;
};
