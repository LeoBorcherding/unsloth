// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  NodeConfig,
  TrainCardConfig,
  TrainCardObjective,
  TrainCardPairRule,
  TrainCardReward,
  TrainCardRun,
} from "../../types";
import { makeTrainConfig } from "../config-factories";
import { isRecord, readString } from "./helpers";

type UiTrainNode = { name: string; train: Record<string, unknown> };

const STRING_FIELDS = [
  "baseModel",
  "promptColumn",
  "responseColumn",
  "maxSteps",
  "learningRate",
  "loraRank",
  "loraAlpha",
  "batchSize",
  "contextLength",
  "holdoutPercent",
  "seed",
  "outputName",
  "pairA",
  "pairB",
  "winnerColumn",
  "scoreAColumn",
  "scoreBColumn",
  "numGenerations",
  "maxCompletionLength",
  "systemPrompt",
  "beta",
] as const satisfies readonly (keyof TrainCardConfig)[];

const OBJECTIVES: readonly TrainCardObjective[] = ["sft", "grpo", "dpo", "orpo"];
const PAIR_RULES: readonly TrainCardPairRule[] = ["a_better", "winner", "scores"];

function readObjective(value: unknown): TrainCardObjective | undefined {
  return OBJECTIVES.find((objective) => objective === value);
}

function parseRewards(value: unknown): TrainCardReward[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const rewards: TrainCardReward[] = [];
  for (const item of value) {
    const name = isRecord(item) ? readString(item.name)?.trim() : undefined;
    const weight = isRecord(item) ? Number(item.weight) : Number.NaN;
    if (name && Number.isFinite(weight)) {
      rewards.push({ name, weight });
    }
  }
  return rewards;
}

export function parseUiTrainNodes(input: unknown): UiTrainNode[] {
  if (!Array.isArray(input)) {
    return [];
  }
  const nodes: UiTrainNode[] = [];
  for (const node of input) {
    if (!isRecord(node) || readString(node.node_type) !== "train") {
      continue;
    }
    const name = (readString(node.name) ?? readString(node.id))?.trim();
    if (name) {
      nodes.push({ name, train: isRecord(node.train) ? node.train : {} });
    }
  }
  return nodes;
}

function parseLastRun(value: unknown): TrainCardRun | null {
  if (!isRecord(value)) {
    return null;
  }
  const trainPath = readString(value.trainPath);
  const executionId = readString(value.executionId);
  if (!(trainPath && executionId)) {
    return null;
  }
  const count = (key: string) => {
    const parsed = Number(value[key]);
    return Number.isFinite(parsed) ? parsed : 0;
  };
  const rowIds = Array.isArray(value.holdoutRowIds)
    ? value.holdoutRowIds.filter((id): id is number => Number.isInteger(id))
    : [];
  return {
    jobId: readString(value.jobId) ?? null,
    executionId,
    sourceJobId: readString(value.sourceJobId) ?? null,
    artifactPath: readString(value.artifactPath) ?? null,
    baseModel: readString(value.baseModel) ?? "",
    outputName: readString(value.outputName) ?? "",
    ...(readObjective(value.objective)
      ? { objective: readObjective(value.objective) }
      : {}),
    promptColumn: readString(value.promptColumn) ?? "",
    responseColumn: readString(value.responseColumn) ?? "",
    holdoutPercent: count("holdoutPercent"),
    seed: count("seed"),
    trainPath,
    holdoutPath: readString(value.holdoutPath) ?? null,
    holdoutRowIds: rowIds,
    trainRows: count("trainRows"),
    holdoutRows: count("holdoutRows"),
    ...(count("pairsDropped") > 0 ? { pairsDropped: count("pairsDropped") } : {}),
    phase: readString(value.phase) ?? "idle",
    outputDir: readString(value.outputDir) ?? null,
    startedAt: count("startedAt"),
  };
}

export function buildTrainConfigFromUi(
  node: UiTrainNode,
  id: string,
  existing: NodeConfig[],
): TrainCardConfig {
  const config: TrainCardConfig = { ...makeTrainConfig(id, existing), name: node.name };
  for (const key of STRING_FIELDS) {
    const value = node.train[key];
    if (typeof value === "string") {
      config[key] = value;
    }
  }
  if (node.train.method === "lora" || node.train.method === "qlora") {
    config.method = node.train.method;
  }
  config.objective = readObjective(node.train.objective) ?? "sft";
  const pairRule = PAIR_RULES.find((rule) => rule === node.train.pairRule);
  if (pairRule) {
    config.pairRule = pairRule;
  }
  config.rewards = parseRewards(node.train.rewards) ?? config.rewards;
  config.lastRun = parseLastRun(node.train.lastRun);
  return config;
}
