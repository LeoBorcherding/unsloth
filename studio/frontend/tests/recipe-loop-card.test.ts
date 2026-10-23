// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);
const { makeBenchmarkConfig, makeLoopConfig, makeSamplerConfig, makeSeedConfig, makeTrainConfig } =
  await import("../src/features/recipe-studio/utils/config-factories.ts");
const { nodeDataFromConfig } = await import("../src/features/recipe-studio/utils/node-data.ts");
const { isValidRecipeConnection } = await import(
  "../src/features/recipe-studio/utils/graph/recipe-graph-connection.ts"
);
const { HANDLE_IDS } = await import("../src/features/recipe-studio/utils/handles.ts");
const {
  findLoopParts,
  loopRecipeColumns,
  loopSetupIssues,
  makeFeedbackSeedConfig,
  parseLoopSettings,
} = await import("../src/features/recipe-studio/utils/loop-card.ts");

type Configs = Parameters<typeof buildRecipePayload>[0];

function loopRecipe() {
  const topic = {
    ...makeSamplerConfig("n1", "category", []),
    name: "topic",
    values: ["billing", "shipping"],
  };
  const feedback = {
    ...makeFeedbackSeedConfig("n2", [topic]),
    hf_path: "/root/.unsloth/studio/seed-uploads/abc_benchmark-failures-round-1.jsonl",
    local_file_name: "benchmark-failures-round-1.jsonl",
    seed_columns: ["origin", "failed_prompt", "failed_reference", "failed_output", "failed_topic"],
  };
  const train = {
    ...makeTrainConfig("n3", [topic, feedback]),
    baseModel: "unsloth/Qwen2.5-0.5B-Instruct",
    promptColumn: "topic",
    responseColumn: "origin",
  };
  const benchmark = {
    ...makeBenchmarkConfig("n4", [topic, feedback, train]),
    prompt_column: "topic",
    reference_column: "origin",
  };
  const loop = { ...makeLoopConfig("n5", []), targetScore: "75", stopOnPlateau: false };
  const configs: Configs = {
    [topic.id]: topic,
    [feedback.id]: feedback,
    [train.id]: train,
    [benchmark.id]: benchmark,
    [loop.id]: loop,
  };
  const nodes = Object.values(configs).map((config, i) => ({
    id: config.id,
    type: "builder" as const,
    position: { x: i * 400, y: 100 },
    data: nodeDataFromConfig(config),
  }));
  const edges = [
    {
      id: "feedback",
      source: benchmark.id,
      target: feedback.id,
      sourceHandle: HANDLE_IDS.benchmarkOut,
      targetHandle: HANDLE_IDS.dataInBottom,
      type: "canvas",
    },
    {
      id: "score",
      source: benchmark.id,
      target: loop.id,
      sourceHandle: HANDLE_IDS.benchmarkOut,
      targetHandle: HANDLE_IDS.dataIn,
      type: "canvas",
    },
  ];
  return { configs, nodes, edges, topic, feedback, train, benchmark, loop };
}

test("the loop card, the failures seed and the feedback edge survive save and reopen", () => {
  const { configs, nodes, edges } = loopRecipe();
  const built = buildRecipePayload(configs, nodes, edges);
  assert.deepEqual(built.errors, []);
  assert.equal(built.payload.ui.seed_loop_feedback, true);
  assert.ok(
    !(built.payload.recipe.columns as { name: string }[]).some((column) => column.name === "loop_1"),
    "the loop card is not a dataset column",
  );
  const seedSource = (built.payload.recipe.seed_config as { source: { seed_type: string } }).source;
  assert.equal(seedSource.seed_type, "local");

  const reopened = importRecipePayload(JSON.stringify(built.payload));
  assert.ok(reopened.snapshot);
  const restored = Object.values(reopened.snapshot.configs);
  const loop = restored.find((config) => config.kind === "loop");
  assert.ok(loop && loop.kind === "loop");
  assert.equal(loop.targetScore, "75");
  assert.equal(loop.stopOnPlateau, false);
  const seed = restored.find((config) => config.kind === "seed");
  assert.ok(seed && seed.kind === "seed");
  assert.equal(seed.loop_feedback, true);

  const benchmarkId = findLoopParts(reopened.snapshot.configs).benchmark?.id;
  const back = reopened.snapshot.edges.find(
    (edge) => edge.source === benchmarkId && edge.target === seed.id,
  );
  assert.ok(back, "the feedback edge is restored");
  assert.equal(back.sourceHandle, HANDLE_IDS.benchmarkOut);
});

test("the Benchmark card feeds only the Loop card and the failures seed", () => {
  const { configs, topic, feedback, benchmark, loop } = loopRecipe();
  const from = (target: string, targetHandle: string = HANDLE_IDS.dataIn) => ({
    source: benchmark.id,
    sourceHandle: HANDLE_IDS.benchmarkOut,
    target,
    targetHandle,
  });
  assert.equal(isValidRecipeConnection(from(feedback.id, HANDLE_IDS.dataInBottom), configs), true);
  assert.equal(isValidRecipeConnection(from(loop.id), configs), true);
  assert.equal(isValidRecipeConnection(from(topic.id), configs), false);
  const plainSeed = { ...makeSeedConfig("n9", [], "local"), name: "plain" };
  assert.equal(
    isValidRecipeConnection(from(plainSeed.id), { ...configs, [plainSeed.id]: plainSeed }),
    false,
  );
});

test("setup issues name what still blocks a start", () => {
  const { configs, feedback, train } = loopRecipe();
  assert.deepEqual(loopSetupIssues(findLoopParts(configs), configs), []);

  const withoutSeed = { ...configs };
  delete withoutSeed[feedback.id];
  assert.deepEqual(
    loopSetupIssues(findLoopParts(withoutSeed), withoutSeed).filter((issue) => !issue.startsWith("Train card")),
    ["Add the Benchmark failures seed."],
  );

  const otherSeed = { ...makeSeedConfig("n8", []), name: "hf_seed" };
  const twoSeeds = { ...withoutSeed, [otherSeed.id]: otherSeed };
  assert.match(loopSetupIssues(findLoopParts(twoSeeds), twoSeeds).join(" "), /Remove seed hf_seed/);

  const noHoldout = { ...configs, [train.id]: { ...train, holdoutPercent: "0" } };
  assert.match(loopSetupIssues(findLoopParts(noHoldout), noHoldout).join(" "), /hold out/);
});

test("the failures seed's own columns are not fed back into it", () => {
  const { configs } = loopRecipe();
  assert.deepEqual(loopRecipeColumns(configs), ["topic"]);
});

test("loop settings parse percentages and turn plateau off with the switch", () => {
  const { settings, errors } = parseLoopSettings(makeLoopConfig("l", []));
  assert.deepEqual(errors, []);
  assert.deepEqual(settings, {
    targetScore: 0.8,
    maxRounds: 3,
    plateauRounds: 2,
    rowsPerRound: 80,
    failureShare: 0.5,
  });
  const off = parseLoopSettings({ ...makeLoopConfig("l", []), stopOnPlateau: false, plateauRounds: "x" });
  assert.deepEqual(off.errors, []);
  assert.equal(off.settings.plateauRounds, 0);
  const bad = parseLoopSettings({ ...makeLoopConfig("l", []), targetScore: "120", maxRounds: "0" });
  assert.equal(bad.errors.length, 2);
});
