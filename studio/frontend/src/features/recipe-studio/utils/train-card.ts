// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RewardRecord } from "@/features/training";
import type { RecipeExecutionRecord } from "../execution-types";
import type {
  NodeConfig,
  TrainCardConfig,
  TrainCardObjective,
  TrainCardPairRule,
  TrainCardReward,
  TrainCardRun,
} from "../types";
import { makeTrainConfig } from "./config-factories";

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

export const TRAIN_OBJECTIVE_LABELS: Record<TrainCardObjective, string> = {
  sft: "SFT",
  grpo: "GRPO",
  dpo: "DPO",
  orpo: "ORPO",
};

type ObjectiveDefaults = Pick<
  TrainCardConfig,
  "learningRate" | "maxSteps" | "contextLength"
>;

/** Per-objective starting points, from the Unsloth notebooks (GRPO and DPO at 5e-6, ORPO at 8e-6). */
export const TRAIN_OBJECTIVE_DEFAULTS: Record<TrainCardObjective, ObjectiveDefaults> = {
  sft: { learningRate: "2e-4", maxSteps: "60", contextLength: "2048" },
  grpo: { learningRate: "5e-6", maxSteps: "50", contextLength: "1024" },
  dpo: { learningRate: "5e-6", maxSteps: "60", contextLength: "2048" },
  orpo: { learningRate: "8e-6", maxSteps: "60", contextLength: "2048" },
};

/** Switching objective moves the fields still at the old objective's default to the new one's. */
export function switchTrainObjective(
  config: TrainCardConfig,
  objective: TrainCardObjective,
): Partial<TrainCardConfig> {
  const from = TRAIN_OBJECTIVE_DEFAULTS[config.objective];
  const to = TRAIN_OBJECTIVE_DEFAULTS[objective];
  const patch: Partial<TrainCardConfig> = { objective };
  for (const key of Object.keys(to) as (keyof ObjectiveDefaults)[]) {
    if (config[key].trim() === from[key] && to[key] !== from[key]) {
      patch[key] = to[key];
    }
  }
  return patch;
}

export function isPreferenceTrainObjective(objective: TrainCardObjective): boolean {
  return objective === "dpo" || objective === "orpo";
}

/** A reward from the library: just what the card needs to check its columns. */
export type TrainRewardInfo = { name: string; compareTo: string | null };

/** Library rewards to pick from: valid rule rewards, plus any picked one the library no longer has. */
export function rewardChoices(
  library: RewardRecord[],
  picked: TrainCardReward[],
): { name: string; description: string; missing: boolean }[] {
  const usable = library.filter(
    (reward) => reward.valid && !reward.shadowed && reward.kind === "rule",
  );
  const names = new Set(usable.map((reward) => reward.name));
  return [
    ...usable.map((reward) => ({
      name: reward.name,
      description: reward.description,
      missing: false,
    })),
    ...picked
      .filter((reward) => !names.has(reward.name))
      .map((reward) => ({
        name: reward.name,
        description: "Not in the reward library.",
        missing: true,
      })),
  ];
}

export type TrainCardPairs = {
  // biome-ignore lint/style/useNamingConvention: api schema
  answer_a: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  answer_b: string;
  rule: TrainCardPairRule;
  // biome-ignore lint/style/useNamingConvention: api schema
  winner_column: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  score_a_column: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  score_b_column: string | null;
};

export type TrainCardSettings = {
  baseModel: string;
  method: TrainCardConfig["method"];
  objective: TrainCardObjective;
  promptColumn: string;
  responseColumn: string;
  rewards: TrainCardReward[];
  pairs: TrainCardPairs | null;
  numGenerations: number;
  maxCompletionLength: number;
  systemPrompt: string;
  beta: number | null;
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

function checkColumns(
  picked: string[],
  columns: string[],
  errors: string[],
): void {
  for (const column of picked) {
    if (column && !columns.includes(column)) {
      errors.push(`Column ${column} is not in this recipe's output.`);
    }
  }
}

function parsePairs(
  config: TrainCardConfig,
  columns: string[],
  errors: string[],
): TrainCardPairs {
  const pairA = config.pairA.trim();
  const pairB = config.pairB.trim();
  const winner = config.winnerColumn.trim();
  const scoreA = config.scoreAColumn.trim();
  const scoreB = config.scoreBColumn.trim();
  const picked = [pairA, pairB];
  if (!(pairA && pairB)) {
    errors.push("Pick answer A and answer B.");
  } else if (pairA === pairB) {
    errors.push("Answer A and answer B must be different columns.");
  }
  if (config.pairRule === "winner") {
    picked.push(winner);
    if (!winner) {
      errors.push("Pick the column that names the better answer.");
    }
  } else if (config.pairRule === "scores") {
    picked.push(scoreA, scoreB);
    if (!(scoreA && scoreB)) {
      errors.push("Pick a score column for each answer.");
    }
  }
  checkColumns(picked, columns, errors);
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    answer_a: pairA,
    // biome-ignore lint/style/useNamingConvention: api schema
    answer_b: pairB,
    rule: config.pairRule,
    // biome-ignore lint/style/useNamingConvention: api schema
    winner_column: config.pairRule === "winner" ? winner : null,
    // biome-ignore lint/style/useNamingConvention: api schema
    score_a_column: config.pairRule === "scores" ? scoreA : null,
    // biome-ignore lint/style/useNamingConvention: api schema
    score_b_column: config.pairRule === "scores" ? scoreB : null,
  };
}

function checkRewards(
  rewards: TrainCardReward[],
  responseColumn: string,
  library: TrainRewardInfo[],
  errors: string[],
): void {
  if (rewards.length === 0) {
    errors.push("Pick at least one reward.");
    return;
  }
  const byName = new Map(library.map((reward) => [reward.name, reward]));
  for (const reward of rewards) {
    if (!Number.isFinite(reward.weight)) {
      errors.push(`Reward ${reward.name} needs a numeric weight.`);
    }
    const info = byName.get(reward.name);
    if (library.length > 0 && !info) {
      errors.push(`Reward ${reward.name} is not in the reward library.`);
    } else if (info?.compareTo === "answer" && !responseColumn) {
      errors.push(`Pick the answer column: ${reward.name} checks against it.`);
    }
  }
}

export function parseTrainCardSettings(
  config: TrainCardConfig,
  columns: string[],
  library: TrainRewardInfo[] = [],
): { settings: TrainCardSettings; errors: string[] } {
  const errors: string[] = [];
  const objective = config.objective;
  const baseModel = config.baseModel.trim();
  if (!baseModel) {
    errors.push("Pick a base model.");
  }
  const promptColumn = config.promptColumn.trim();
  const responseColumn = config.responseColumn.trim();
  const preference = isPreferenceTrainObjective(objective);
  if (objective === "sft") {
    if (!promptColumn || !responseColumn) {
      errors.push("Pick the prompt and response columns.");
    } else if (promptColumn === responseColumn) {
      errors.push("Prompt and response must be different columns.");
    } else {
      checkColumns([promptColumn, responseColumn], columns, errors);
    }
  } else if (!promptColumn) {
    errors.push("Pick the prompt column.");
  } else if (objective === "grpo" && promptColumn === responseColumn) {
    errors.push("Prompt and answer must be different columns.");
  } else {
    checkColumns(
      [promptColumn, objective === "grpo" ? responseColumn : ""],
      columns,
      errors,
    );
  }
  const pairs = preference ? parsePairs(config, columns, errors) : null;
  if (objective === "grpo") {
    checkRewards(config.rewards, responseColumn, library, errors);
  }
  const learningRate = parseNumber(config.learningRate);
  if (!(Number.isFinite(learningRate) && learningRate > 0 && learningRate < 1)) {
    errors.push("Learning rate must be a number between 0 and 1.");
  }
  const beta = config.beta.trim() ? parseNumber(config.beta) : null;
  if (
    beta !== null &&
    !(Number.isFinite(beta) && (objective === "grpo" ? beta >= 0 : beta > 0))
  ) {
    errors.push(
      objective === "grpo"
        ? "Beta must be 0 or more."
        : "Beta must be a number above 0.",
    );
  }
  const grpo = objective === "grpo";
  const settings: TrainCardSettings = {
    baseModel,
    method: config.method,
    objective,
    promptColumn,
    responseColumn: preference ? "" : responseColumn,
    rewards: grpo ? config.rewards : [],
    pairs,
    numGenerations: grpo
      ? wholeNumber("Generations", config.numGenerations, 2, 64, errors)
      : 4,
    maxCompletionLength: grpo
      ? wholeNumber(
          "Max answer tokens",
          config.maxCompletionLength,
          16,
          32_768,
          errors,
        )
      : 0,
    systemPrompt: grpo ? config.systemPrompt.trim() : "",
    beta: objective === "sft" ? null : beta,
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

/** RL roles for the Train page's Column roles. Preference splits add `chosen` and `rejected`. */
export function buildTrainCardRlMapping(
  settings: Pick<TrainCardSettings, "objective" | "promptColumn" | "responseColumn">,
): Record<string, string> {
  if (settings.objective === "grpo") {
    return settings.responseColumn
      ? { [settings.promptColumn]: "prompt", [settings.responseColumn]: "answer" }
      : { [settings.promptColumn]: "prompt" };
  }
  return {
    [settings.promptColumn]: "prompt",
    chosen: "chosen",
    rejected: "rejected",
  };
}

export function isTrainRunActive(run: TrainCardRun | null | undefined): boolean {
  return Boolean(run && ACTIVE_PHASES.has(run.phase));
}

/** The rows training never saw. `row_ids` are positions in the recipe run's parquet shards
 *  (sorted file order); the rows themselves come from `/jobs/{source_job_id}/training-split/holdout-rows`. */
export type TrainCardHoldout = {
  // biome-ignore lint/style/useNamingConvention: handle schema
  source_job_id: string;
  // biome-ignore lint/style/useNamingConvention: handle schema
  artifact_path: string;
  percent: number;
  seed: number;
  // biome-ignore lint/style/useNamingConvention: handle schema
  parquet_path: string;
  // biome-ignore lint/style/useNamingConvention: handle schema
  row_ids: number[];
  rows: number;
};

/** Output handle contract: what a Benchmark card reads off a finished Train card. */
export type TrainCardOutput = {
  // biome-ignore lint/style/useNamingConvention: handle schema
  adapter_path: string | null;
  // biome-ignore lint/style/useNamingConvention: handle schema
  base_model: string;
  // biome-ignore lint/style/useNamingConvention: handle schema
  training_job_id: string;
  objective: TrainCardObjective;
  // biome-ignore lint/style/useNamingConvention: handle schema
  prompt_column: string;
  // biome-ignore lint/style/useNamingConvention: handle schema
  response_column: string;
  holdout: TrainCardHoldout | null;
};

/** Only a finished run has an adapter to hand on. */
export function getTrainCardOutput(
  config: NodeConfig | undefined,
): TrainCardOutput | null {
  if (config?.kind !== "train") {
    return null;
  }
  const run = config.lastRun;
  if (!(run?.phase === "completed" && run.jobId)) {
    return null;
  }
  const holdout =
    run.holdoutPath && run.sourceJobId && run.artifactPath && run.holdoutRows > 0
      ? {
          // biome-ignore lint/style/useNamingConvention: handle schema
          source_job_id: run.sourceJobId,
          // biome-ignore lint/style/useNamingConvention: handle schema
          artifact_path: run.artifactPath,
          percent: run.holdoutPercent,
          seed: run.seed,
          // biome-ignore lint/style/useNamingConvention: handle schema
          parquet_path: run.holdoutPath,
          // biome-ignore lint/style/useNamingConvention: handle schema
          row_ids: run.holdoutRowIds,
          rows: run.holdoutRows,
        }
      : null;
  return {
    // biome-ignore lint/style/useNamingConvention: handle schema
    adapter_path: run.outputDir,
    // biome-ignore lint/style/useNamingConvention: handle schema
    base_model: run.baseModel,
    // biome-ignore lint/style/useNamingConvention: handle schema
    training_job_id: run.jobId,
    objective: run.objective ?? "sft",
    // biome-ignore lint/style/useNamingConvention: handle schema
    prompt_column: run.promptColumn,
    // biome-ignore lint/style/useNamingConvention: handle schema
    response_column: run.responseColumn,
    holdout,
  };
}

export type TrainCardState =
  | "needs_run"
  | "setup"
  | "ready"
  | "preparing"
  | "training"
  | "adapter_ready"
  | "stopped"
  | "failed";

/** One state per card, so the canvas shows a single status instead of stacked rows. */
export function getTrainCardState({
  hasSource,
  errors,
  lastRun,
  active,
  launching,
  launchFailed,
}: {
  hasSource: boolean;
  errors: string[];
  lastRun: TrainCardRun | null;
  active: boolean;
  launching: boolean;
  launchFailed: boolean;
}): TrainCardState {
  if (launching) {
    return "preparing";
  }
  if (active) {
    return "training";
  }
  if (launchFailed || lastRun?.phase === "error") {
    return "failed";
  }
  if (lastRun?.phase === "completed") {
    return "adapter_ready";
  }
  if (lastRun?.phase === "stopped") {
    return "stopped";
  }
  if (!hasSource) {
    return "needs_run";
  }
  return errors.length > 0 ? "setup" : "ready";
}

/** "unsloth/Qwen2.5-0.5B-Instruct" reads as "Qwen2.5-0.5B-Instruct" on a narrow card. */
export function shortModelName(model: string): string {
  const trimmed = model.trim().replace(/\/+$/, "");
  return trimmed.slice(trimmed.lastIndexOf("/") + 1) || trimmed;
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
  const rows =
    run.objective === "dpo" || run.objective === "orpo" ? "pairs" : "rows";
  const base =
    run.holdoutRows > 0
      ? `${run.trainRows} train ${rows} / ${run.holdoutRows} held out`
      : `${run.trainRows} train ${rows}`;
  return run.pairsDropped ? `${base} · ${run.pairsDropped} ties left out` : base;
}

/** The headline number while training: loss for SFT, mean reward for GRPO, accuracy for DPO/ORPO. */
export function pickTrainProgressMetric(
  objective: TrainCardObjective,
  loss: number | null,
  rl: Record<string, number> | undefined,
): { label: string; value: number } | null {
  if (objective === "grpo" && rl && Number.isFinite(rl.reward)) {
    return { label: "reward", value: rl.reward };
  }
  if (
    isPreferenceTrainObjective(objective) &&
    rl &&
    Number.isFinite(rl["rewards/accuracies"])
  ) {
    return { label: "accuracy", value: rl["rewards/accuracies"] };
  }
  return loss ? { label: "loss", value: loss } : null;
}

const PAIR_RULE_WORDS: Record<TrainCardPairRule, string> = {
  a_better: "A always wins",
  winner: "winner column",
  scores: "higher score",
};

/** Short chips for the card face: what it reads and how long it runs. */
export function trainCardChips(config: TrainCardConfig): string[] {
  const prompt = config.promptColumn.trim();
  const response = config.responseColumn.trim();
  const chips: string[] = [];
  if (config.objective === "grpo") {
    if (prompt) {
      chips.push(response ? `${prompt} → ${response}` : prompt);
    }
    const names = config.rewards.map((reward) => reward.name);
    chips.push(
      names.length === 0
        ? "No rewards"
        : names.length > 2
          ? `${names.length} rewards`
          : names.join(" + "),
    );
  } else if (isPreferenceTrainObjective(config.objective)) {
    const a = config.pairA.trim();
    const b = config.pairB.trim();
    if (a && b) {
      chips.push(`${a} vs ${b}`, PAIR_RULE_WORDS[config.pairRule]);
    }
  } else if (prompt && response) {
    chips.push(`${prompt} → ${response}`);
  }
  chips.push(
    `${config.maxSteps || "?"} steps`,
    `${config.holdoutPercent || 0}% held out`,
  );
  return chips;
}

const ADVANCED_LABELS = {
  learningRate: "learning rate",
  loraRank: "LoRA rank",
  loraAlpha: "LoRA alpha",
  batchSize: "batch size",
  contextLength: "max sequence length",
  seed: "seed",
} as const satisfies Partial<Record<keyof TrainCardConfig, string>>;

const RL_ADVANCED_LABELS = {
  numGenerations: "generations",
  maxCompletionLength: "max answer tokens",
  systemPrompt: "system prompt",
  beta: "beta",
} as const satisfies Partial<Record<keyof TrainCardConfig, string>>;

/** The collapsed Advanced row names what differs from a new card's defaults, so nothing changed hides. */
export function summarizeAdvancedTrainSettings(config: TrainCardConfig): string {
  const defaults = {
    ...makeTrainConfig(config.id, []),
    ...TRAIN_OBJECTIVE_DEFAULTS[config.objective],
  };
  const changed: string[] = [];
  if (config.method !== defaults.method) {
    changed.push("method");
  }
  for (const [key, label] of Object.entries(ADVANCED_LABELS)) {
    const field = key as keyof typeof ADVANCED_LABELS;
    if (config[field].trim() !== defaults[field]) {
      changed.push(label);
    }
  }
  const rlFields: (keyof typeof RL_ADVANCED_LABELS)[] =
    config.objective === "grpo"
      ? ["numGenerations", "maxCompletionLength", "systemPrompt", "beta"]
      : config.objective === "sft"
        ? []
        : ["beta"];
  for (const field of rlFields) {
    if (config[field].trim() !== defaults[field].trim()) {
      changed.push(RL_ADVANCED_LABELS[field]);
    }
  }
  const outputName = config.outputName.trim();
  if (outputName && outputName !== config.name.trim()) {
    changed.push("run name");
  }
  if (changed.length === 0) {
    return "Defaults";
  }
  return changed.length > 2
    ? `${changed.length} changed`
    : `Changed: ${changed.join(", ")}`;
}
