// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the Loop card's settings parse to. Scores are pass rates from 0 to 1, higher is better. */
export type LoopSettings = {
  targetScore: number;
  maxRounds: number;
  // 0 turns plateau detection off.
  plateauRounds: number;
  rowsPerRound: number;
  // Share of each round after the first written from the last round's failed examples.
  failureShare: number;
};

/** One row of the feedback seed: a failed held-out example, or a blank row for fresh data. */
export type FeedbackSeedRow = Record<string, string> & {
  origin: "failure" | "fresh";
};

export type LoopFailedExample = {
  prompt: string;
  reference: string | null;
  output: string | null;
  row: Record<string, unknown>;
};

/** The Train card's held-out rows from round 1: every round is scored on the same rows. */
export type LoopHoldout = {
  sourceJobId: string;
  artifactPath: string;
  holdoutPath: string;
  rows: number;
};

export type LoopDataset = {
  executionId: string;
  jobId: string;
  artifactPath: string;
  runName: string;
  rows: number;
  fromFailures: number;
  // A few failure-derived rows: the failed prompt and the new row written from it.
  examples: { failed: string; written: string }[];
};

export type LoopAdapter = {
  jobId: string;
  outputDir: string;
  // The previous round's adapter this run continued from, null when it started from the base model.
  continuedFrom: string | null;
  trainRows: number;
};

export type LoopScore = {
  score: number;
  passed: number;
  scored: number;
  baseline: number | null;
  failedExamples: LoopFailedExample[];
  failedTotal: number;
};

export type LoopRoundStatus =
  | "generating"
  | "training"
  | "scoring"
  | "done"
  | "error";

export type LoopRound = {
  index: number;
  status: LoopRoundStatus;
  dataset: LoopDataset | null;
  adapter: LoopAdapter | null;
  result: LoopScore | null;
  error: string | null;
  startedAt: number;
  finishedAt: number | null;
};

export type LoopStatus =
  | "running"
  | "target_reached"
  | "plateau"
  | "max_rounds"
  | "stopped"
  | "error";

export type LoopRecord = {
  id: string;
  recipeId: string;
  createdAt: number;
  updatedAt: number;
  status: LoopStatus;
  settings: LoopSettings;
  holdout: LoopHoldout | null;
  rounds: LoopRound[];
  error: string | null;
};

type StepInput = { round: number; signal: AbortSignal };

/** The three steps a round chains; the controller only sees this interface. */
export type LoopSteps = {
  generate: (
    input: StepInput & { seedRows: FeedbackSeedRow[] },
  ) => Promise<LoopDataset>;
  train: (
    input: StepInput & {
      dataset: LoopDataset;
      // Null on round 1: the Train card carves the held-out split from this round's rows.
      holdout: LoopHoldout | null;
      continueFrom: string | null;
    },
  ) => Promise<{ adapter: LoopAdapter; holdout: LoopHoldout }>;
  score: (
    input: StepInput & { adapter: LoopAdapter; holdout: LoopHoldout },
  ) => Promise<LoopScore>;
};
