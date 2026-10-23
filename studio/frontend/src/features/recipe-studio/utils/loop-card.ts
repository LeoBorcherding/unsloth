// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoopSettings } from "../loop/types";
import type {
  BenchmarkConfig,
  LoopCardConfig,
  NodeConfig,
  SeedConfig,
  TrainCardConfig,
} from "../types";
import { isRecipeTask } from "./benchmark";
import { makeLoopConfig, makeSeedConfig } from "./config-factories";
import { listTrainableColumns, parseTrainCardSettings } from "./train-card";

const STRING_FIELDS = [
  "targetScore",
  "maxRounds",
  "plateauRounds",
  "rowsPerRound",
  "failureShare",
] as const satisfies readonly (keyof LoopCardConfig)[];

export function parseLoopCardConfig(
  id: string,
  name: string,
  raw: unknown,
): LoopCardConfig {
  const input = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const config: LoopCardConfig = { ...makeLoopConfig(id, []), name };
  for (const key of STRING_FIELDS) {
    if (typeof input[key] === "string") {
      config[key] = input[key];
    }
  }
  if (typeof input.stopOnPlateau === "boolean") {
    config.stopOnPlateau = input.stopOnPlateau;
  }
  return config;
}

function wholeNumber(
  label: string,
  value: string,
  min: number,
  max: number,
  errors: string[],
): number {
  const trimmed = value.trim();
  const parsed = trimmed ? Number(trimmed) : Number.NaN;
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    errors.push(`${label} must be a whole number from ${min} to ${max}.`);
  }
  return parsed;
}

export function parseLoopSettings(config: LoopCardConfig): {
  settings: LoopSettings;
  errors: string[];
} {
  const errors: string[] = [];
  const target = wholeNumber("Target score", config.targetScore, 1, 100, errors);
  const share = wholeNumber("Rows from failures", config.failureShare, 0, 100, errors);
  const settings: LoopSettings = {
    targetScore: target / 100,
    maxRounds: wholeNumber("Max rounds", config.maxRounds, 1, 20, errors),
    plateauRounds: config.stopOnPlateau
      ? wholeNumber("Plateau rounds", config.plateauRounds, 1, 10, errors)
      : 0,
    rowsPerRound: wholeNumber("Rows per round", config.rowsPerRound, 4, 5000, errors),
    failureShare: share / 100,
  };
  return { settings, errors };
}

export type LoopParts = {
  train: TrainCardConfig | null;
  benchmark: BenchmarkConfig | null;
  feedback: SeedConfig | null;
  // A recipe takes one seed, so another seed leaves no room for the failures seed.
  otherSeed: SeedConfig | null;
};

export function findLoopParts(configs: Record<string, NodeConfig>): LoopParts {
  const parts: LoopParts = { train: null, benchmark: null, feedback: null, otherSeed: null };
  for (const config of Object.values(configs)) {
    if (config.kind === "train") {
      parts.train ??= config;
    } else if (config.kind === "benchmark") {
      parts.benchmark ??= config;
    } else if (config.kind === "seed") {
      if (config.loop_feedback) {
        parts.feedback ??= config;
      } else {
        parts.otherSeed ??= config;
      }
    }
  }
  return parts;
}

/** The recipe's own output columns, without the ones the failures seed adds. */
export function loopRecipeColumns(configs: Record<string, NodeConfig>): string[] {
  const own = Object.fromEntries(
    Object.entries(configs).filter(
      ([, config]) => !(config.kind === "seed" && config.loop_feedback),
    ),
  );
  return listTrainableColumns(own);
}

/** What still blocks a loop start, in the order a user would fix it. */
export function loopSetupIssues(
  parts: LoopParts,
  configs: Record<string, NodeConfig>,
): string[] {
  const issues: string[] = [];
  if (!parts.train) {
    issues.push("Add a Train card.");
  } else {
    const { errors } = parseTrainCardSettings(parts.train, listTrainableColumns(configs));
    issues.push(...errors.map((error) => `Train card: ${error}`));
    if (Number(parts.train.holdoutPercent) <= 0) {
      issues.push("Train card: hold out some rows so every round is scored on the same ones.");
    }
  }
  if (!parts.benchmark) {
    issues.push("Add a Benchmark card.");
  } else {
    const benchmark = parts.benchmark;
    if (!isRecipeTask(benchmark.task)) {
      // lm-eval tasks need a GGUF, and each round produces a LoRA adapter.
      issues.push("Benchmark card: use the Recipe held-out task to score each round's adapter.");
    }
    if (!benchmark.prompt_column.trim()) {
      issues.push("Benchmark card: pick the prompt column.");
    }
    if (benchmark.scorer === "reference" && !benchmark.reference_column?.trim()) {
      issues.push("Benchmark card: pick the reference column.");
    }
    if (benchmark.scorer === "judge" && !benchmark.judge_alias?.trim()) {
      issues.push("Benchmark card: pick a judge model.");
    }
  }
  if (parts.otherSeed) {
    issues.push(
      `Remove seed ${parts.otherSeed.name}: the Benchmark failures seed is this recipe's only seed.`,
    );
  } else if (!parts.feedback) {
    issues.push("Add the Benchmark failures seed.");
  }
  return issues;
}

export function isFeedbackSeed(config: NodeConfig | undefined): boolean {
  return config?.kind === "seed" && config.loop_feedback === true;
}

/** Named like any seed: an imported recipe's one seed always comes back as "seed". */
export function makeFeedbackSeedConfig(id: string, existing: NodeConfig[]): SeedConfig {
  return {
    ...makeSeedConfig(id, existing, "local"),
    name: "seed",
    // biome-ignore lint/style/useNamingConvention: ui schema
    loop_feedback: true,
  };
}
