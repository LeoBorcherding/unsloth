// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeExecutionRecord } from "../execution-types";
import type { NodeConfig, TrainCardConfig, TrainCardRun } from "../types";

/** Node kinds that take a trained adapter off a Train card's output handle. */
export const TRAINED_MODEL_CONSUMER_KINDS: ReadonlySet<string> = new Set([
  "benchmark",
]);

const ACTIVE_PHASES = new Set([
  "starting",
  "downloading_model",
  "downloading_dataset",
  "loading_model",
  "loading_dataset",
  "configuring",
  "training",
  "finalizing",
  "training_drafter",
]);

export type TrainCardSettings = {
  baseModel: string;
  method: TrainCardConfig["method"];
  promptColumn: string;
  responseColumn: string;
  maxSteps: number;
  learningRate: number;
  loraRank: number;
  loraAlpha: number;
  batchSize: number;
  contextLength: number;
  holdoutPercent: number;
  seed: number;
  outputName: string;
};

/** Newest completed full run with saved files: previews keep their rows in memory only. */
export function pickTrainSourceExecution(
  executions: RecipeExecutionRecord[],
): RecipeExecutionRecord | null {
  let best: RecipeExecutionRecord | null = null;
  for (const execution of executions) {
    if (
      execution.kind !== "full" ||
      execution.status !== "completed" ||
      !execution.jobId ||
      !execution.artifact_path?.trim()
    ) {
      continue;
    }
    if (!best || execution.createdAt > best.createdAt) {
      best = execution;
    }
  }
  return best;
}

/** Columns the recipe writes to its dataset, for the prompt and response pickers. */
export function listTrainableColumns(
  configs: Record<string, NodeConfig>,
): string[] {
  const names: string[] = [];
  for (const config of Object.values(configs)) {
    if (
      config.kind === "sampler" ||
      config.kind === "llm" ||
      config.kind === "expression"
    ) {
      if (!config.drop && config.name.trim()) {
        names.push(config.name.trim());
      }
      continue;
    }
    if (config.kind === "seed" && !config.drop) {
      const dropped = new Set(config.seed_drop_columns ?? []);
      for (const column of config.seed_columns ?? []) {
        if (column.trim() && !dropped.has(column)) {
          names.push(column.trim());
        }
      }
    }
  }
  return Array.from(new Set(names));
}

function parseNumber(value: string): number {
  const trimmed = value.trim();
  return trimmed ? Number(trimmed) : Number.NaN;
}

function wholeNumber(
  label: string,
  value: string,
  min: number,
  max: number,
  errors: string[],
): number {
  const parsed = parseNumber(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    errors.push(`${label} must be a whole number from ${min} to ${max}.`);
  }
  return parsed;
}

export function parseTrainCardSettings(
  config: TrainCardConfig,
  columns: string[],
): { settings: TrainCardSettings; errors: string[] } {
  const errors: string[] = [];
  const baseModel = config.baseModel.trim();
  if (!baseModel) {
    errors.push("Pick a base model.");
  }
  const promptColumn = config.promptColumn.trim();
  const responseColumn = config.responseColumn.trim();
  if (!promptColumn || !responseColumn) {
    errors.push("Pick the prompt and response columns.");
  } else if (promptColumn === responseColumn) {
    errors.push("Prompt and response must be different columns.");
  } else {
    for (const column of [promptColumn, responseColumn]) {
      if (!columns.includes(column)) {
        errors.push(`Column ${column} is not in this recipe's output.`);
      }
    }
  }
  const learningRate = parseNumber(config.learningRate);
  if (!(Number.isFinite(learningRate) && learningRate > 0 && learningRate < 1)) {
    errors.push("Learning rate must be a number between 0 and 1.");
  }
  const settings: TrainCardSettings = {
    baseModel,
    method: config.method,
    promptColumn,
    responseColumn,
    maxSteps: wholeNumber("Steps", config.maxSteps, 1, 100_000, errors),
    learningRate,
    loraRank: wholeNumber("LoRA rank", config.loraRank, 1, 1024, errors),
    loraAlpha: wholeNumber("LoRA alpha", config.loraAlpha, 1, 4096, errors),
    batchSize: wholeNumber("Batch size", config.batchSize, 1, 1024, errors),
    contextLength: wholeNumber(
      "Max sequence length",
      config.contextLength,
      128,
      262_144,
      errors,
    ),
    holdoutPercent: wholeNumber(
      "Hold out",
      config.holdoutPercent,
      0,
      50,
      errors,
    ),
    seed: wholeNumber("Seed", config.seed, 0, 2 ** 31 - 1, errors),
    outputName: config.outputName.trim() || config.name,
  };
  return { settings, errors };
}

/** Column roles for the training store's manual mapping, so the run never stops to ask. */
export function buildTrainCardMapping(
  settings: Pick<TrainCardSettings, "promptColumn" | "responseColumn">,
): Record<string, string> {
  return {
    [settings.promptColumn]: "user",
    [settings.responseColumn]: "assistant",
  };
}

export function isTrainRunActive(run: TrainCardRun | null | undefined): boolean {
  return Boolean(run && ACTIVE_PHASES.has(run.phase));
}

/** What the output handle carries: only a finished run has an adapter to hand on. */
export function getTrainCardOutput(
  config: NodeConfig | undefined,
): TrainCardRun | null {
  if (config?.kind !== "train") {
    return null;
  }
  const run = config.lastRun;
  return run?.phase === "completed" && run.jobId ? run : null;
}

export function describeTrainPhase(phase: string): string {
  switch (phase) {
    case "starting":
      return "Starting";
    case "downloading_model":
      return "Downloading model";
    case "downloading_dataset":
    case "loading_dataset":
      return "Loading dataset";
    case "loading_model":
      return "Loading model";
    case "configuring":
      return "Configuring";
    case "training":
      return "Training";
    case "finalizing":
    case "training_drafter":
      return "Saving adapter";
    case "completed":
      return "Finished";
    case "stopped":
      return "Stopped";
    case "error":
      return "Failed";
    default:
      return phase;
  }
}

export function formatHoldoutSummary(run: TrainCardRun): string {
  return run.holdoutRows > 0
    ? `${run.trainRows} train / ${run.holdoutRows} held out`
    : `${run.trainRows} train rows`;
}
