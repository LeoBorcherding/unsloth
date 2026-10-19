// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildModelConfig } = await import(
  "../src/features/recipe-studio/utils/payload/builders-model.ts"
);
const { parseModelConfig } = await import(
  "../src/features/recipe-studio/utils/import/parsers/model-parser.ts"
);
const { buildExecutionPayload } = await import(
  "../src/features/recipe-studio/executions/run-settings.ts"
);

type ModelConfig = Parameters<typeof buildModelConfig>[0];

function model(name: string, provider: string, parallel: string): ModelConfig {
  return {
    id: `${name}-id`,
    kind: "model_config",
    name,
    model: name,
    provider,
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_max_parallel_requests: parallel,
  } as ModelConfig;
}

function parallelOf(config: Record<string, unknown>): unknown {
  const inference = config.inference_parameters as
    | Record<string, unknown>
    | undefined;
  return inference?.max_parallel_requests;
}

const settings = {
  batchSize: 1000,
  batchEnabled: false,
  mergeBatches: false,
  llmParallelRequests: null,
  nonInferenceWorkers: 4,
  maxConversationRestarts: 5,
  maxConversationCorrectionSteps: 0,
  disableEarlyShutdown: true,
  shutdownErrorRate: 0.5,
  shutdownErrorWindow: 10,
};

test("a model's own requests-at-once is sent and survives a save and reopen", () => {
  const errors: string[] = [];
  const built = buildModelConfig(model("local", "local", "1"), errors);
  assert.deepEqual(errors, []);
  assert.equal(parallelOf(built), 1);

  const parsed = parseModelConfig(built, "local", "local-id");
  assert.equal(parsed.inference_max_parallel_requests, "1");
});

test("an empty requests-at-once leaves the default to the backend", () => {
  const built = buildModelConfig(model("remote", "remote", ""), []);
  assert.equal(parallelOf(built), undefined);
});

test("a requests-at-once below 1 or with a fraction is refused, not sent", () => {
  for (const value of ["0", "-1", "0.5", "two"]) {
    const errors: string[] = [];
    const built = buildModelConfig(model("local", "local", value), errors);
    assert.equal(parallelOf(built), undefined, value);
    assert.equal(errors.length, 1, value);
  }
});

test("with no run-wide override a local and a remote model keep different limits", () => {
  const modelConfigs = [
    buildModelConfig(model("local", "local", "1"), []),
    buildModelConfig(model("remote", "remote", "16"), []),
  ];
  const payload = buildExecutionPayload({
    payload: {
      recipe: {
        // biome-ignore lint/style/useNamingConvention: api schema
        model_configs: modelConfigs,
      },
      run: {},
    } as never,
    kind: "preview",
    rows: 3,
    settings,
  });
  const limits = payload.recipe.model_configs.map((config) => [
    config.alias,
    parallelOf(config as Record<string, unknown>),
  ]);
  assert.deepEqual(limits, [
    ["local", 1],
    ["remote", 16],
  ]);
});
