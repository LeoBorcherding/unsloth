// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);
const { makeTrainConfig } = await import(
  "../src/features/recipe-studio/utils/config-factories.ts"
);
const { nodeDataFromConfig } = await import(
  "../src/features/recipe-studio/utils/node-data.ts"
);
const { isValidRecipeConnection } = await import(
  "../src/features/recipe-studio/utils/graph/recipe-graph-connection.ts"
);
const {
  buildTrainCardMapping,
  getTrainCardOutput,
  getTrainCardState,
  listTrainableColumns,
  parseTrainCardSettings,
  pickTrainSourceExecution,
  shortModelName,
  summarizeAdvancedTrainSettings,
} = await import("../src/features/recipe-studio/utils/train-card.ts");

type Snapshot = NonNullable<
  ReturnType<typeof importRecipePayload>["snapshot"]
>;
type TrainCardConfig = ReturnType<typeof makeTrainConfig>;

function loadTemplate(): Snapshot {
  const template = readFileSync(
    new URL(
      "../src/features/data-recipes/learning-recipes/text-to-sql.json",
      import.meta.url,
    ),
    "utf8",
  );
  const imported = importRecipePayload(template);
  assert.ok(imported.snapshot);
  return imported.snapshot;
}

function withTrainCard(snapshot: Snapshot, patch: Partial<TrainCardConfig>) {
  const train: TrainCardConfig = {
    ...makeTrainConfig("n999", Object.values(snapshot.configs)),
    ...patch,
  };
  const configs = { ...snapshot.configs, [train.id]: train };
  const nodes = [
    ...snapshot.nodes,
    {
      id: train.id,
      type: "builder" as const,
      position: { x: 900, y: 120 },
      data: nodeDataFromConfig(train),
    },
  ];
  return { configs, nodes, train };
}

function build(snapshot: Snapshot, configs: Snapshot["configs"], nodes: Snapshot["nodes"]) {
  return buildRecipePayload(
    configs,
    nodes,
    snapshot.edges,
    snapshot.processors,
    snapshot.layoutDirection,
    snapshot.auxNodePositions,
  );
}

const finishedRun = {
  jobId: "job-7",
  executionId: "exec-2",
  sourceJobId: "recipe-job-3",
  artifactPath: "/data/recipe_x",
  baseModel: "unsloth/Qwen2.5-0.5B-Instruct",
  outputName: "train_1",
  promptColumn: "question",
  responseColumn: "answer",
  holdoutPercent: 10,
  seed: 3407,
  trainPath: "/data/recipe_x/splits/holdout-10-seed3407/train.parquet",
  holdoutPath: "/data/recipe_x/splits/holdout-10-seed3407/holdout.parquet",
  holdoutRowIds: [3, 17, 21, 40, 52, 66, 71, 80, 88, 95],
  trainRows: 90,
  holdoutRows: 10,
  phase: "completed",
  outputDir: "/outputs/train_1",
  startedAt: 1_700_000_000_000,
};

test("a train card stays out of the dataset columns and survives save and reopen", () => {
  const snapshot = loadTemplate();
  const { configs, nodes } = withTrainCard(snapshot, {
    baseModel: "unsloth/Qwen2.5-0.5B-Instruct",
    method: "lora",
    holdoutPercent: "15",
    promptColumn: "sql_prompt",
    responseColumn: "sql",
    lastRun: finishedRun,
  });
  const baseline = build(snapshot, snapshot.configs, snapshot.nodes);
  const built = build(snapshot, configs, nodes);
  assert.deepEqual(built.errors, baseline.errors);
  assert.deepEqual(built.payload.recipe, baseline.payload.recipe);
  const uiNode = built.payload.ui.nodes.find((node) => node.node_type === "train");
  assert.ok(uiNode, "the card is stored in ui.nodes");

  const reopened = importRecipePayload(JSON.stringify(built.payload));
  assert.ok(reopened.snapshot);
  const restored = Object.values(reopened.snapshot.configs).find(
    (config) => config.kind === "train",
  );
  assert.ok(restored && restored.kind === "train");
  assert.equal(restored.baseModel, "unsloth/Qwen2.5-0.5B-Instruct");
  assert.equal(restored.method, "lora");
  assert.equal(restored.holdoutPercent, "15");
  assert.equal(restored.promptColumn, "sql_prompt");
  assert.deepEqual(restored.lastRun, finishedRun);
  const restoredNode = reopened.snapshot.nodes.find((node) => node.id === restored.id);
  assert.deepEqual(restoredNode?.position, { x: 900, y: 120 });
});

test("settings parse into numbers and name every problem", () => {
  const base = makeTrainConfig("t1", []);
  const columns = ["question", "answer"];
  const ready = parseTrainCardSettings(
    {
      ...base,
      baseModel: " unsloth/Qwen2.5-0.5B-Instruct ",
      promptColumn: "question",
      responseColumn: "answer",
    },
    columns,
  );
  assert.deepEqual(ready.errors, []);
  assert.equal(ready.settings.baseModel, "unsloth/Qwen2.5-0.5B-Instruct");
  assert.equal(ready.settings.learningRate, 2e-4);
  assert.equal(ready.settings.holdoutPercent, 10);
  assert.equal(ready.settings.maxSteps, 60);
  assert.deepEqual(buildTrainCardMapping(ready.settings), {
    question: "user",
    answer: "assistant",
  });

  const broken = parseTrainCardSettings(
    {
      ...base,
      promptColumn: "question",
      responseColumn: "gone",
      holdoutPercent: "60",
      maxSteps: "1.5",
      learningRate: "fast",
    },
    columns,
  );
  assert.deepEqual(broken.errors, [
    "Pick a base model.",
    "Column gone is not in this recipe's output.",
    "Learning rate must be a number between 0 and 1.",
    "Steps must be a whole number from 1 to 100000.",
    "Hold out must be a whole number from 0 to 50.",
  ]);
});

test("the card trains on the newest completed full run with saved files", () => {
  const run = (patch: Record<string, unknown>) =>
    ({
      id: "x",
      kind: "full",
      status: "completed",
      jobId: "job",
      artifact_path: "/data/recipe_x",
      createdAt: 1,
      ...patch,
    }) as never;
  const picked = pickTrainSourceExecution([
    run({ id: "old", createdAt: 1 }),
    run({ id: "preview", kind: "preview", createdAt: 9 }),
    run({ id: "failed", status: "error", createdAt: 8 }),
    run({ id: "nofiles", artifact_path: null, createdAt: 7 }),
    run({ id: "new", createdAt: 5 }),
  ]);
  assert.equal(picked?.id, "new");
  assert.equal(pickTrainSourceExecution([run({ status: "running" })]), null);
});

test("trainable columns skip dropped fields and model setup", () => {
  const snapshot = loadTemplate();
  const columns = listTrainableColumns(snapshot.configs);
  assert.ok(columns.includes("sql_prompt"));
  assert.ok(columns.includes("sql"));
  for (const config of Object.values(snapshot.configs)) {
    if (config.kind === "model_config" || config.kind === "model_provider") {
      assert.ok(!columns.includes(config.name));
    }
    if ("drop" in config && config.drop) {
      assert.ok(!columns.includes(config.name));
    }
  }
});

test("only a finished run is offered on the output handle", () => {
  const card = makeTrainConfig("t1", []);
  assert.equal(
    getTrainCardOutput({ ...card, lastRun: { ...finishedRun, phase: "training" } }),
    null,
  );
  assert.equal(getTrainCardOutput({ ...card, lastRun: null }), null);
});

test("the output handle names the adapter and the exact held-out rows", () => {
  const card = makeTrainConfig("t1", []);
  assert.deepEqual(getTrainCardOutput({ ...card, lastRun: finishedRun }), {
    adapter_path: "/outputs/train_1",
    base_model: "unsloth/Qwen2.5-0.5B-Instruct",
    training_job_id: "job-7",
    prompt_column: "question",
    response_column: "answer",
    holdout: {
      source_job_id: "recipe-job-3",
      artifact_path: "/data/recipe_x",
      percent: 10,
      seed: 3407,
      parquet_path: "/data/recipe_x/splits/holdout-10-seed3407/holdout.parquet",
      row_ids: [3, 17, 21, 40, 52, 66, 71, 80, 88, 95],
      rows: 10,
    },
  });
  const noHoldout = getTrainCardOutput({
    ...card,
    lastRun: { ...finishedRun, holdoutPath: null, holdoutRowIds: [], holdoutRows: 0 },
  });
  assert.equal(noHoldout?.holdout, null);
});

test("a saved run keeps its held-out row ids through save and reopen", () => {
  const snapshot = loadTemplate();
  const { configs, nodes, train } = withTrainCard(snapshot, { lastRun: finishedRun });
  const built = build(snapshot, configs, nodes);
  const reopened = importRecipePayload(JSON.stringify(built.payload));
  const card = Object.values(reopened.snapshot?.configs ?? {}).find(
    (config) => config.kind === "train" && config.name === train.name,
  );
  assert.ok(card && card.kind === "train");
  assert.deepEqual(card.lastRun, finishedRun);
});

test("the card shows one state, most urgent first", () => {
  const base = {
    hasSource: true,
    errors: [] as string[],
    lastRun: null,
    active: false,
    launching: false,
    launchFailed: false,
  };
  assert.equal(getTrainCardState({ ...base, hasSource: false }), "needs_run");
  assert.equal(
    getTrainCardState({ ...base, hasSource: false, errors: ["Pick a base model."] }),
    "needs_run",
  );
  assert.equal(getTrainCardState({ ...base, errors: ["Pick a base model."] }), "setup");
  assert.equal(getTrainCardState(base), "ready");
  assert.equal(getTrainCardState({ ...base, launching: true, active: true }), "preparing");
  assert.equal(
    getTrainCardState({ ...base, active: true, lastRun: { ...finishedRun, phase: "training" } }),
    "training",
  );
  assert.equal(getTrainCardState({ ...base, lastRun: finishedRun }), "adapter_ready");
  assert.equal(
    getTrainCardState({ ...base, hasSource: false, lastRun: finishedRun }),
    "adapter_ready",
  );
  assert.equal(
    getTrainCardState({ ...base, lastRun: { ...finishedRun, phase: "stopped" } }),
    "stopped",
  );
  assert.equal(
    getTrainCardState({ ...base, lastRun: { ...finishedRun, phase: "error" } }),
    "failed",
  );
  assert.equal(getTrainCardState({ ...base, launchFailed: true }), "failed");
});

test("the collapsed Advanced row says what changed from the defaults", () => {
  const card = makeTrainConfig("t1", []);
  assert.equal(summarizeAdvancedTrainSettings(card), "Defaults");
  assert.equal(
    summarizeAdvancedTrainSettings({ ...card, maxSteps: "200", holdoutPercent: "20" }),
    "Defaults",
  );
  assert.equal(
    summarizeAdvancedTrainSettings({ ...card, learningRate: "1e-4" }),
    "Changed: learning rate",
  );
  assert.equal(
    summarizeAdvancedTrainSettings({ ...card, method: "lora", outputName: "support-bot" }),
    "Changed: method, run name",
  );
  assert.equal(
    summarizeAdvancedTrainSettings({ ...card, method: "lora", seed: "1", loraRank: "8" }),
    "3 changed",
  );
});

test("model ids shorten to the name after the org", () => {
  assert.equal(shortModelName("unsloth/Qwen2.5-0.5B-Instruct"), "Qwen2.5-0.5B-Instruct");
  assert.equal(shortModelName("local-model"), "local-model");
  assert.equal(shortModelName("/models/llama/"), "llama");
});

test("columns can feed a train card, but its adapter cannot feed a column", () => {
  const snapshot = loadTemplate();
  const { configs, train } = withTrainCard(snapshot, {});
  const column = Object.values(snapshot.configs).find(
    (config) => config.kind === "llm",
  );
  assert.ok(column);
  assert.equal(
    isValidRecipeConnection(
      {
        source: column.id,
        target: train.id,
        sourceHandle: "data-out",
        targetHandle: "data-in",
      },
      configs,
    ),
    true,
  );
  assert.equal(
    isValidRecipeConnection(
      {
        source: train.id,
        target: column.id,
        sourceHandle: "data-out",
        targetHandle: "data-in",
      },
      configs,
    ),
    false,
  );
});
