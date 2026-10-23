// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig, TrainCardConfig, TrainCardRun } from "../../types";
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
] as const satisfies readonly (keyof TrainCardConfig)[];

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
    promptColumn: readString(value.promptColumn) ?? "",
    responseColumn: readString(value.responseColumn) ?? "",
    holdoutPercent: count("holdoutPercent"),
    seed: count("seed"),
    trainPath,
    holdoutPath: readString(value.holdoutPath) ?? null,
    holdoutRowIds: rowIds,
    trainRows: count("trainRows"),
    holdoutRows: count("holdoutRows"),
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
  config.lastRun = parseLastRun(node.train.lastRun);
  return config;
}
