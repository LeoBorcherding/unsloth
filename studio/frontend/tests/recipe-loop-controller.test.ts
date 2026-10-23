// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  DEFAULT_LOOP_CONFIG,
  EVAL_LOSS_METRIC,
  createLoopRecord,
  loopStopReason,
  resumeLoopRecord,
  rowsForIteration,
  runLoop,
} = await import("../src/features/recipe-studio/loop/controller.ts");
const { buildLoopTrainingRequest, datasetVersionFromArtifact } = await import(
  "../src/features/recipe-studio/loop/training-request.ts"
);

type Steps = Parameters<typeof runLoop>[0]["steps"];
type Iteration = Parameters<typeof loopStopReason>[0][number];

function done(index: number, score: number): Iteration {
  return {
    index,
    status: "done",
    dataset: null,
    run: null,
    score,
    error: null,
    startedAt: 0,
    finishedAt: 0,
  };
}

const config = {
  ...DEFAULT_LOOP_CONFIG,
  target: 0.5,
  maxIterations: 5,
  plateauPatience: 2,
  minImprovement: 0.01,
  baseRows: 100,
  rowsPerIteration: 50,
  holdoutRows: 32,
};

function fakeSteps(scores: number[], calls: string[]): Steps {
  let version = 0;
  return {
    metric: EVAL_LOSS_METRIC,
    async generate({ rows, purpose }) {
      version += 1;
      calls.push(`generate:${purpose}:${rows}`);
      return { version: `recipe_v${version}`, rows, executionId: `e${version}` };
    },
    async train({ iteration, datasets, holdout }) {
      calls.push(
        `train:${iteration}:${datasets.map((d) => d.version).join(",")}:${holdout?.version}`,
      );
      return { runId: `run${iteration}`, adapter: `adapter${iteration}` };
    },
    async score({ iteration }) {
      calls.push(`score:${iteration}`);
      const score = scores[iteration - 1];
      if (score === undefined) throw new Error("no score scripted");
      return score;
    },
  };
}

test("stop rules: target, plateau, iteration cap, and a higher-is-better metric", () => {
  assert.equal(loopStopReason([], config, EVAL_LOSS_METRIC), null);
  assert.equal(
    loopStopReason([done(1, 0.9), done(2, 0.5)], config, EVAL_LOSS_METRIC),
    "target_reached",
  );
  assert.equal(
    loopStopReason([done(1, 0.9), done(2, 0.7)], config, EVAL_LOSS_METRIC),
    null,
  );
  // Best of the last two (0.795) is not 0.01 better than the best before them (0.8).
  assert.equal(
    loopStopReason(
      [done(1, 0.8), done(2, 0.795), done(3, 0.81)],
      config,
      EVAL_LOSS_METRIC,
    ),
    "plateau",
  );
  assert.equal(
    loopStopReason(
      [done(1, 0.8), done(2, 0.795), done(3, 0.81)],
      { ...config, plateauPatience: 0, maxIterations: 3 },
      EVAL_LOSS_METRIC,
    ),
    "max_iterations",
  );
  const accuracy = { id: "accuracy", label: "Accuracy", goal: "max" as const };
  assert.equal(
    loopStopReason([done(1, 0.4), done(2, 0.92)], { ...config, target: 0.9 }, accuracy),
    "target_reached",
  );
  assert.equal(rowsForIteration(config, 1), 100);
  assert.equal(rowsForIteration(config, 3), 50);
});

test("a loop makes the held-out rows once, trains on every version so far, and stops at the target", async () => {
  const calls: string[] = [];
  const saved: Array<{ status: string; iterations: number }> = [];
  const record = createLoopRecord({
    id: "l1",
    recipeId: "r1",
    config,
    metric: EVAL_LOSS_METRIC,
    now: 1,
  });
  const final = await runLoop({
    record,
    steps: fakeSteps([1.2, 0.8, 0.45], calls),
    signal: new AbortController().signal,
    save: (r) => saved.push({ status: r.status, iterations: r.iterations.length }),
    now: () => 2,
  });

  assert.equal(final.status, "target_reached");
  assert.deepEqual(calls, [
    "generate:holdout:32",
    "generate:train:100",
    "train:1:recipe_v2:recipe_v1",
    "score:1",
    "generate:train:50",
    "train:2:recipe_v2,recipe_v3:recipe_v1",
    "score:2",
    "generate:train:50",
    "train:3:recipe_v2,recipe_v3,recipe_v4:recipe_v1",
    "score:3",
  ]);
  assert.deepEqual(
    final.iterations.map((it) => [it.index, it.dataset?.version, it.run?.runId, it.score]),
    [
      [1, "recipe_v2", "run1", 1.2],
      [2, "recipe_v3", "run2", 0.8],
      [3, "recipe_v4", "run3", 0.45],
    ],
  );
  // Saved after every step, so a reload mid-iteration still has the partial history.
  assert.ok(saved.length >= 1 + 3 * 4);
  assert.deepEqual(saved[saved.length - 1], { status: "target_reached", iterations: 3 });
});

test("resume keeps finished iterations and the held-out rows, and redoes the unfinished one", async () => {
  const calls: string[] = [];
  const interrupted = {
    ...createLoopRecord({
      id: "l1",
      recipeId: "r1",
      config,
      metric: EVAL_LOSS_METRIC,
      now: 1,
    }),
    holdout: { version: "recipe_hold", rows: 32, executionId: "eh" },
    iterations: [
      { ...done(1, 1.0), dataset: { version: "recipe_a", rows: 100, executionId: "e1" } },
      {
        ...done(2, 0),
        status: "training" as const,
        score: null,
        dataset: { version: "recipe_b", rows: 50, executionId: "e2" },
      },
    ],
  };
  const final = await runLoop({
    record: resumeLoopRecord(interrupted, 5),
    steps: fakeSteps([1.0, 0.7, 0.69, 0.695], calls),
    signal: new AbortController().signal,
    save: () => undefined,
  });

  assert.equal(calls[0], "generate:train:50");
  assert.ok(!calls.some((c) => c.startsWith("generate:holdout")));
  assert.equal(calls[1], "train:2:recipe_a,recipe_v1:recipe_hold");
  assert.equal(final.status, "plateau");
  assert.deepEqual(
    final.iterations.map((it) => it.score),
    [1.0, 0.7, 0.69, 0.695],
  );
});

test("a failing step ends the loop with the error on that iteration; Stop ends it as stopped", async () => {
  const failing = fakeSteps([], []);
  failing.train = async () => {
    throw new Error("CUDA out of memory");
  };
  const failed = await runLoop({
    record: createLoopRecord({ id: "l1", recipeId: "r1", config, metric: EVAL_LOSS_METRIC, now: 1 }),
    steps: failing,
    signal: new AbortController().signal,
    save: () => undefined,
  });
  assert.equal(failed.status, "error");
  assert.equal(failed.error, "CUDA out of memory");
  assert.equal(failed.iterations[0]?.status, "error");

  const controller = new AbortController();
  const stopping = fakeSteps([1.0, 0.9], []);
  const score = stopping.score;
  stopping.score = async (input) => {
    const value = await score(input);
    controller.abort();
    return value;
  };
  const stopped = await runLoop({
    record: createLoopRecord({ id: "l2", recipeId: "r1", config, metric: EVAL_LOSS_METRIC, now: 1 }),
    steps: stopping,
    signal: controller.signal,
    save: () => undefined,
  });
  assert.equal(stopped.status, "stopped");
  assert.deepEqual(stopped.iterations.map((it) => it.status), ["done"]);
});

test("training reads the recipe datasets through the recipes/ dataset root", () => {
  assert.equal(
    datasetVersionFromArtifact("/home/u/.unsloth/studio/assets/datasets/recipes/recipe_loop-v1"),
    "recipe_loop-v1",
  );
  assert.equal(datasetVersionFromArtifact(null), null);
  const request = buildLoopTrainingRequest({
    datasets: [
      { version: "recipe_a", rows: 100, executionId: null },
      { version: "recipe_b", rows: 50, executionId: null },
    ],
    holdout: { version: "recipe_hold", rows: 32, executionId: null },
    config: { modelName: "unsloth/SmolLM2-135M-Instruct", maxSteps: 30, learningRate: "2e-4" },
    iteration: 2,
  });
  assert.deepEqual(request.local_datasets, [
    "recipes/recipe_a/parquet-files",
    "recipes/recipe_b/parquet-files",
  ]);
  assert.deepEqual(request.local_eval_datasets, ["recipes/recipe_hold/parquet-files"]);
  assert.ok((request.eval_steps as number) > 0);
  const noHoldout = buildLoopTrainingRequest({
    datasets: [],
    holdout: null,
    config: { modelName: "m", maxSteps: 1, learningRate: "1e-4" },
    iteration: 1,
  });
  assert.deepEqual(noHoldout.local_eval_datasets, []);
});
