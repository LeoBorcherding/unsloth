// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const benchmark = await import(
  "../src/features/recipe-studio/utils/benchmark.ts"
);
const { makeBenchmarkConfig } = await import(
  "../src/features/recipe-studio/utils/config-factories.ts"
);
const { applyRecipeConnection, isValidRecipeConnection } = await import(
  "../src/features/recipe-studio/utils/graph/recipe-graph-connection.ts"
);
const { HANDLE_IDS } = await import(
  "../src/features/recipe-studio/utils/handles.ts"
);
const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);
const { nodeDataFromConfig } = await import(
  "../src/features/recipe-studio/utils/node-data.ts"
);

type Snapshot = NonNullable<
  ReturnType<typeof importRecipePayload>["snapshot"]
>;
type BenchmarkConfig = ReturnType<typeof makeBenchmarkConfig>;

function loadTemplate(name: string): Snapshot {
  const raw = readFileSync(
    new URL(
      `../src/features/data-recipes/learning-recipes/${name}.json`,
      import.meta.url,
    ),
    "utf8",
  );
  const imported = importRecipePayload(raw);
  assert.ok(imported.snapshot, imported.errors.join(" "));
  return imported.snapshot;
}

function build(snapshot: Snapshot) {
  return buildRecipePayload(
    snapshot.configs,
    snapshot.nodes,
    snapshot.edges,
    snapshot.processors,
    snapshot.layoutDirection,
    snapshot.auxNodePositions,
  );
}

function addBenchmark(snapshot: Snapshot): {
  snapshot: Snapshot;
  benchmarkId: string;
} {
  const benchmarkId = `n${snapshot.nextId}`;
  const config = makeBenchmarkConfig(benchmarkId, Object.values(snapshot.configs));
  const configs = { ...snapshot.configs, [benchmarkId]: config };
  const node = {
    id: benchmarkId,
    type: "builder" as const,
    position: { x: 900, y: 40 },
    data: nodeDataFromConfig(config, snapshot.layoutDirection),
  };
  const modelId = Object.values(configs).find(
    (item) => item.kind === "model_config",
  )?.id;
  assert.ok(modelId);
  const connection = {
    source: modelId,
    sourceHandle: HANDLE_IDS.semanticOut,
    target: benchmarkId,
    targetHandle: HANDLE_IDS.semanticIn,
  };
  assert.equal(isValidRecipeConnection(connection, configs), true);
  const connected = applyRecipeConnection(connection, configs, snapshot.edges);
  return {
    benchmarkId,
    snapshot: {
      ...snapshot,
      configs: connected.configs ?? configs,
      edges: connected.edges,
      nodes: [...snapshot.nodes, node],
      nextId: snapshot.nextId + 1,
    },
  };
}

test("the held-out slice is the tail of the run, capped by max rows", () => {
  assert.deepEqual(benchmark.getBenchmarkHoldout(100, "20", "10"), {
    offset: 80,
    limit: 10,
    holdout: 20,
  });
  assert.deepEqual(benchmark.getBenchmarkHoldout(7, "10", "50"), {
    offset: 6,
    limit: 1,
    holdout: 1,
  });
  assert.deepEqual(benchmark.getBenchmarkHoldout(0, "20", "20").limit, 0);
  assert.equal(benchmark.getBenchmarkHoldout(1000, "100", "9999").limit, 200);
});

test("a model block wired into the card sets the model under test and survives save and reload", () => {
  const template = loadTemplate("instruction-from-answer");
  const { snapshot, benchmarkId } = addBenchmark(template);
  const wired = snapshot.configs[benchmarkId] as BenchmarkConfig;
  assert.equal(wired.model_alias, "ministral");

  const configs = {
    ...snapshot.configs,
    [benchmarkId]: {
      ...wired,
      prompt_column: "generated_instruction",
      reference_column: "output",
      scorer: "judge",
      judge_alias: "ministral",
      baseline_mode: "none",
    } satisfies BenchmarkConfig,
  };
  const built = build({ ...snapshot, configs });
  assert.deepEqual(built.errors, build(template).errors, "the card adds no build errors");
  assert.equal(
    built.payload.recipe.columns.some((column) => column.name === wired.name),
    false,
    "the card is not a dataset column",
  );
  const uiNode = built.payload.ui.nodes.find((node) => node.id === wired.name);
  assert.equal(uiNode?.node_type, "benchmark");
  assert.ok(
    built.payload.ui.edges.some(
      (edge) => edge.from === "ministral" && edge.to === wired.name,
    ),
  );

  const reloaded = importRecipePayload(JSON.stringify(built.payload));
  assert.ok(reloaded.snapshot, reloaded.errors.join(" "));
  const restored = Object.values(reloaded.snapshot.configs).find(
    (item) => item.kind === "benchmark",
  ) as BenchmarkConfig | undefined;
  assert.ok(restored);
  assert.equal(restored.model_alias, "ministral");
  assert.equal(restored.scorer, "judge");
  assert.equal(restored.prompt_column, "generated_instruction");
  assert.equal(restored.baseline_mode, "none");
  assert.ok(
    reloaded.snapshot.edges.some((edge) => edge.target === restored.id),
    "the model edge is restored",
  );
});

test("the request carries only the models the card uses", () => {
  const { snapshot, benchmarkId } = addBenchmark(loadTemplate("text-to-python"));
  const config = {
    ...(snapshot.configs[benchmarkId] as BenchmarkConfig),
    prompt_column: "instruction",
    scorer: "python",
    baseline_mode: "adapter_off",
  } satisfies BenchmarkConfig;
  const rows = [{ instruction: "Write add(a, b)." }];
  const { request, errors } = benchmark.buildBenchmarkRequest(
    config,
    Object.values(snapshot.configs),
    rows,
  );
  assert.deepEqual(errors, []);
  assert.ok(request);
  assert.equal(request.model_alias, "coding-model");
  assert.deepEqual(request.baseline, { mode: "adapter_off", alias: null });
  assert.equal(request.judge_alias, null);
  assert.deepEqual(
    request.model_configs.map((item) => item.alias),
    ["coding-model"],
  );
  assert.deepEqual(
    request.model_providers.map((item) => item.name),
    ["openai-compatible"],
  );
});

test("an incomplete card explains what is missing instead of running", () => {
  const config = {
    ...makeBenchmarkConfig("n1", []),
    scorer: "judge",
  } satisfies BenchmarkConfig;
  const { request, errors } = benchmark.buildBenchmarkRequest(config, [], []);
  assert.equal(request, null);
  assert.deepEqual(errors, [
    "Connect or pick the model to test.",
    "Pick the column with the prompt.",
    "Pick a judge model.",
    "No held-out rows. Run the recipe first.",
  ]);
});

test("the output handle exposes score, delta and failed examples", () => {
  const config = {
    ...makeBenchmarkConfig("n1", []),
    last_result: {
      candidate: { total: 4, scored: 4, passed: 3, errors: 0, score: 0.75 },
      baseline: { total: 4, scored: 4, passed: 1, errors: 0, score: 0.25 },
      delta: 0.5,
      cancelled: false,
      failed_examples: [
        { prompt: "3*3?", reference: "9", output: "6", reason: "no", row: {} },
      ],
      failed_total: 1,
      finished_at: 1,
      execution_id: "run-1",
      rows: 4,
    },
  } satisfies BenchmarkConfig;
  const output = benchmark.getBenchmarkOutput(config);
  assert.deepEqual(
    { ...output, failed_examples: output?.failed_examples.length },
    { score: 0.75, baseline_score: 0.25, delta: 0.5, failed_examples: 1 },
  );
  assert.equal(benchmark.formatBenchmarkScore(0.75), "75%");
  assert.equal(benchmark.formatBenchmarkDelta(0.5), "+50 pts");
  assert.equal(benchmark.formatBenchmarkDelta(-0.1), "-10 pts");
  assert.equal(
    benchmark.failedExamplesToJsonl(config.last_result),
    '{"prompt":"3*3?","reference":"9","output":"6","reason":"no"}',
  );
});

test("the newest finished full run is the benchmark source", () => {
  const run = (id: string, kind: string, status: string, finishedAt: number) =>
    ({ id, kind, status, finishedAt, createdAt: 0, jobId: id, dataset: [] }) as never;
  const picked = benchmark.pickBenchmarkExecution([
    run("old-full", "full", "completed", 1),
    run("new-preview", "preview", "completed", 3),
    run("new-full", "full", "completed", 2),
    run("failed", "full", "error", 4),
  ]);
  assert.equal(picked?.id, "new-full");
});

test("a wired-in Train card's held-out split replaces the last-N% slice", () => {
  const train = {
    id: "t1",
    kind: "train",
    name: "train_1",
    lastRun: {
      phase: "completed",
      jobId: "job-1",
      outputName: "support-bot",
      outputDir: "/outputs/support-bot",
      holdoutPath: "/recipes/run-1/splits/holdout-20-seed3407/holdout.parquet",
      holdoutRows: 20,
    },
  };
  const edges = [{ source: "t1", target: "b1" }];
  const trained = benchmark.findTrainedModelInput("b1", edges, { t1: train });
  assert.deepEqual(trained, {
    trainName: "train_1",
    outputName: "support-bot",
    outputDir: "/outputs/support-bot",
    holdoutPath: "/recipes/run-1/splits/holdout-20-seed3407/holdout.parquet",
    holdoutRows: 20,
  });

  const unfinished = { ...train, lastRun: { ...train.lastRun, phase: "training" } };
  assert.equal(benchmark.findTrainedModelInput("b1", edges, { t1: unfinished }), null);
  assert.equal(benchmark.findTrainedModelInput("other", edges, { t1: train }), null);

  const { snapshot, benchmarkId } = addBenchmark(loadTemplate("text-to-python"));
  const config = {
    ...(snapshot.configs[benchmarkId] as BenchmarkConfig),
    prompt_column: "instruction",
    scorer: "python",
    max_rows: "15",
  } satisfies BenchmarkConfig;
  const { request, errors } = benchmark.buildBenchmarkRequest(
    config,
    Object.values(snapshot.configs),
    [],
    trained,
  );
  assert.deepEqual(errors, []);
  assert.equal(request?.holdout_path, trained?.holdoutPath);
  assert.equal(request?.max_rows, 15);
  assert.deepEqual(request?.rows, []);
});
