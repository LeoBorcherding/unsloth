// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createLoopRecord, loopStopReason, resumeLoopRecord, runLoop } = await import(
  "../src/features/recipe-studio/loop/controller.ts"
);
const { buildFeedbackSeedRows, feedbackSeedColumns } = await import(
  "../src/features/recipe-studio/loop/feedback-seed.ts"
);
const { buildLoopTrainingRequest } = await import(
  "../src/features/recipe-studio/loop/training-request.ts"
);
const { buildLoopBenchmarkRequest, loopScoreFromResult } = await import(
  "../src/features/recipe-studio/loop/benchmark-request.ts"
);

type Steps = Parameters<typeof runLoop>[0]["steps"];
type Round = Parameters<typeof loopStopReason>[0][number];
type Settings = Parameters<typeof loopStopReason>[1];
type Failure = Parameters<typeof buildFeedbackSeedRows>[0]["failures"][number];

const settings: Settings = {
  targetScore: 0.8,
  maxRounds: 4,
  plateauRounds: 2,
  rowsPerRound: 10,
  failureShare: 0.5,
};

function done(index: number, score: number): Round {
  return {
    index,
    status: "done",
    dataset: null,
    adapter: null,
    result: {
      score,
      passed: 0,
      scored: 10,
      baseline: null,
      failedExamples: [],
      failedTotal: 0,
    },
    error: null,
    startedAt: 0,
    finishedAt: 0,
  };
}

function failure(question: string, topic: string): Failure {
  return {
    prompt: question,
    reference: `Q-${topic.toUpperCase()}`,
    output: "Q-BIL",
    row: { ticket: question, topic, queue: `Q-${topic.toUpperCase()}` },
  };
}

test("stop rules read the benchmark score as higher is better", () => {
  assert.equal(loopStopReason([], settings), null);
  assert.equal(loopStopReason([done(1, 0.5)], settings), null);
  assert.equal(loopStopReason([done(1, 0.5), done(2, 0.8)], settings), "target_reached");
  assert.equal(loopStopReason([done(1, 0.9)], settings), "target_reached");
  // Two rounds that never beat round 1 by a full point is a plateau.
  assert.equal(
    loopStopReason([done(1, 0.6), done(2, 0.605), done(3, 0.55)], settings),
    "plateau",
  );
  assert.equal(
    loopStopReason([done(1, 0.6), done(2, 0.55), done(3, 0.62)], settings),
    null,
  );
  assert.equal(
    loopStopReason([done(1, 0.6), done(2, 0.6), done(3, 0.6)], {
      ...settings,
      plateauRounds: 0,
    }),
    null,
  );
  assert.equal(
    loopStopReason([done(1, 0.4), done(2, 0.5), done(3, 0.6), done(4, 0.7)], settings),
    "max_rounds",
  );
  const errored: Round = { ...done(2, 0.95), status: "error" };
  assert.equal(loopStopReason([done(1, 0.5), errored], settings), null);
});

test("failed examples become seed rows, mixed with blank fresh rows", () => {
  const failures = [failure("Refund my order", "returns"), failure("Card declined", "billing")];
  const rows = buildFeedbackSeedRows({
    failures,
    rows: 10,
    failureShare: 0.5,
    recipeColumns: ["ticket", "topic", "queue"],
  });
  assert.equal(rows.length, 10);
  const fromFailures = rows.filter((row) => row.origin === "failure");
  assert.equal(fromFailures.length, 5);
  // Cycles through the failures when there are fewer than slots.
  assert.deepEqual(
    fromFailures.map((row) => row.failed_topic),
    ["returns", "billing", "returns", "billing", "returns"],
  );
  assert.equal(fromFailures[0]?.failed_prompt, "Refund my order");
  assert.equal(fromFailures[0]?.failed_reference, "Q-RETURNS");
  assert.equal(fromFailures[0]?.failed_output, "Q-BIL");
  const fresh = rows.filter((row) => row.origin === "fresh");
  assert.equal(fresh.length, 5);
  for (const row of fresh) {
    assert.equal(row.failed_prompt, "");
    assert.equal(row.failed_topic, "");
  }
  // Every row has the same columns, so the seed reads as one table.
  const columns = feedbackSeedColumns(["ticket", "topic", "queue"]);
  for (const row of rows) {
    assert.deepEqual(Object.keys(row).sort(), [...columns].sort());
  }
});

test("round 1, with nothing failed yet, is all fresh rows", () => {
  const rows = buildFeedbackSeedRows({
    failures: [],
    rows: 4,
    failureShare: 0.5,
    recipeColumns: ["ticket"],
  });
  assert.deepEqual(
    rows.map((row) => row.origin),
    ["fresh", "fresh", "fresh", "fresh"],
  );
  const all = buildFeedbackSeedRows({
    failures: [failure("a", "tech")],
    rows: 3,
    failureShare: 1,
    recipeColumns: ["ticket"],
  });
  assert.deepEqual(
    all.map((row) => row.origin),
    ["failure", "failure", "failure"],
  );
});

const trainSettings = {
  baseModel: "unsloth/Qwen2.5-0.5B-Instruct",
  method: "qlora" as const,
  promptColumn: "ticket",
  responseColumn: "queue",
  maxSteps: 30,
  learningRate: 0.0002,
  loraRank: 16,
  loraAlpha: 16,
  batchSize: 4,
  contextLength: 1024,
  holdoutPercent: 25,
  seed: 3407,
  outputName: "router",
};

test("round 1 trains from the base model, later rounds continue the last adapter", () => {
  const first = buildLoopTrainingRequest({
    settings: trainSettings,
    trainPath: "/data/r1/splits/holdout-25-seed3407/train.parquet",
    continueFrom: null,
    round: 1,
  });
  assert.equal(first.model_name, "unsloth/Qwen2.5-0.5B-Instruct");
  assert.equal(first.model_format, null);
  assert.deepEqual(first.local_datasets, [
    "/data/r1/splits/holdout-25-seed3407/train.parquet",
  ]);
  assert.deepEqual(first.custom_format_mapping, { ticket: "user", queue: "assistant" });
  assert.equal(first.max_steps, 30);
  assert.equal(first.load_in_4bit, true);

  const second = buildLoopTrainingRequest({
    settings: trainSettings,
    trainPath: "/data/r2/parquet-files",
    continueFrom: "/outputs/router_round_1",
    round: 2,
  });
  assert.equal(second.model_name, "/outputs/router_round_1");
  assert.equal(second.model_format, "adapter");
  assert.equal(second.training_type, "LoRA/QLoRA");
  assert.deepEqual(second.local_datasets, ["/data/r2/parquet-files"]);
  assert.notEqual(second.random_seed, first.random_seed);
});

function fakeSteps(scores: number[], calls: Record<string, unknown[]>): Steps {
  return {
    generate: async ({ round, seedRows }) => {
      calls.generate.push({ round, seedRows });
      return {
        executionId: `exec-${round}`,
        jobId: `job-${round}`,
        artifactPath: `/data/r${round}`,
        runName: `round ${round}`,
        rows: seedRows.length,
        fromFailures: seedRows.filter((row) => row.origin === "failure").length,
        examples: [],
      };
    },
    train: async ({ round, holdout, continueFrom }) => {
      calls.train.push({ round, holdout, continueFrom });
      return {
        adapter: {
          jobId: `train-${round}`,
          outputDir: `/outputs/round_${round}`,
          continuedFrom: continueFrom,
          trainRows: 8,
        },
        holdout: holdout ?? {
          sourceJobId: `job-${round}`,
          artifactPath: `/data/r${round}`,
          holdoutPath: `/data/r${round}/splits/holdout.parquet`,
          rows: 2,
        },
      };
    },
    score: async ({ round, holdout }) => {
      calls.score.push({ round, holdout: holdout.holdoutPath });
      const score = scores[round - 1] ?? 0;
      return {
        score,
        passed: Math.round(score * 10),
        scored: 10,
        baseline: null,
        failedExamples: [failure(`missed in round ${round}`, "returns")],
        failedTotal: 1,
      };
    },
  };
}

test("each round feeds its failures forward and continues from the last adapter", async () => {
  const calls: Record<string, unknown[]> = { generate: [], train: [], score: [] };
  const saved: unknown[] = [];
  const record = await runLoop({
    record: createLoopRecord({ id: "l1", recipeId: "r1", settings, now: 0 }),
    steps: fakeSteps([0.5, 0.7, 0.85], calls),
    recipeColumns: ["ticket", "topic", "queue"],
    signal: new AbortController().signal,
    save: (r) => saved.push(r),
    now: () => 1,
  });
  assert.equal(record.status, "target_reached");
  assert.deepEqual(
    record.rounds.map((round) => round.result?.score),
    [0.5, 0.7, 0.85],
  );

  const generated = calls.generate as { round: number; seedRows: { origin: string; failed_prompt: string }[] }[];
  assert.equal(generated[0]?.seedRows.filter((row) => row.origin === "failure").length, 0);
  const round2 = generated[1]?.seedRows.filter((row) => row.origin === "failure") ?? [];
  assert.equal(round2.length, 5);
  assert.equal(round2[0]?.failed_prompt, "missed in round 1");
  assert.equal(record.rounds[1]?.dataset?.fromFailures, 5);

  const trained = calls.train as { continueFrom: string | null; holdout: unknown }[];
  assert.deepEqual(
    trained.map((call) => call.continueFrom),
    [null, "/outputs/round_1", "/outputs/round_2"],
  );
  assert.equal(trained[0]?.holdout, null);
  // Every round is scored on round 1's held-out rows.
  assert.deepEqual(
    (calls.score as { holdout: string }[]).map((call) => call.holdout),
    ["/data/r1/splits/holdout.parquet", "/data/r1/splits/holdout.parquet", "/data/r1/splits/holdout.parquet"],
  );
  assert.ok(saved.length >= 9, "saved after every step");
});

test("a failed round can be resumed from the last finished adapter", async () => {
  const calls: Record<string, unknown[]> = { generate: [], train: [], score: [] };
  const steps = fakeSteps([0.5, 0.6, 0.9], calls);
  const failing: Steps = {
    ...steps,
    train: async (input) => {
      if (input.round === 2) {
        throw new Error("CUDA out of memory");
      }
      return steps.train(input);
    },
  };
  const broken = await runLoop({
    record: createLoopRecord({ id: "l2", recipeId: "r1", settings, now: 0 }),
    steps: failing,
    recipeColumns: ["ticket"],
    signal: new AbortController().signal,
    save: () => undefined,
  });
  assert.equal(broken.status, "error");
  assert.equal(broken.error, "CUDA out of memory");
  assert.deepEqual(
    broken.rounds.map((round) => round.status),
    ["done", "error"],
  );

  const resumed = resumeLoopRecord(broken, settings, 5);
  assert.equal(resumed.status, "running");
  assert.equal(resumed.rounds.length, 1);
  assert.equal(resumed.holdout?.holdoutPath, "/data/r1/splits/holdout.parquet");
  calls.train = [];
  const finished = await runLoop({
    record: resumed,
    steps,
    recipeColumns: ["ticket"],
    signal: new AbortController().signal,
    save: () => undefined,
  });
  assert.equal(finished.status, "target_reached");
  assert.deepEqual(
    finished.rounds.map((round) => round.index),
    [1, 2, 3],
  );
  assert.equal(
    (calls.train[0] as { continueFrom: string | null }).continueFrom,
    "/outputs/round_1",
  );
});

test("stopping mid-round marks the loop stopped and the round unfinished", async () => {
  const controller = new AbortController();
  const calls: Record<string, unknown[]> = { generate: [], train: [], score: [] };
  const steps = fakeSteps([0.5], calls);
  const record = await runLoop({
    record: createLoopRecord({ id: "l3", recipeId: "r1", settings, now: 0 }),
    steps: {
      ...steps,
      train: async () => {
        controller.abort();
        const error = new Error("Stopped.");
        error.name = "AbortError";
        throw error;
      },
    },
    recipeColumns: ["ticket"],
    signal: controller.signal,
    save: () => undefined,
  });
  assert.equal(record.status, "stopped");
  assert.equal(record.rounds[0]?.status, "error");
  assert.equal(record.rounds[0]?.error, "Stopped.");
});

test("a loop record survives a JSON round trip, the shape the server stores", () => {
  const record = resumeLoopRecord(
    {
      ...createLoopRecord({ id: "l4", recipeId: "r1", settings, now: 0 }),
      rounds: [done(1, 0.5), done(2, 0.7)],
      holdout: { sourceJobId: "j", artifactPath: "/a", holdoutPath: "/a/h.parquet", rows: 20 },
      status: "stopped",
    },
    settings,
    9,
  );
  assert.deepEqual(JSON.parse(JSON.stringify(record)), record);
});

const benchmark = {
  id: "b1",
  kind: "benchmark" as const,
  name: "benchmark",
  model_alias: "",
  baseline_mode: "adapter_off" as const,
  prompt_column: "ticket",
  reference_column: "queue",
  scorer: "reference" as const,
  holdout_percent: "20",
  max_rows: "30",
};

test("the loop benchmarks the loaded adapter on the fixed held-out split", () => {
  const { request, errors } = buildLoopBenchmarkRequest({
    benchmark,
    configs: [],
    holdoutPath: "/data/r1/splits/holdout-25-seed3407/holdout.parquet",
  });
  assert.deepEqual(errors, []);
  assert.ok(request);
  assert.equal(request.holdout_path, "/data/r1/splits/holdout-25-seed3407/holdout.parquet");
  assert.deepEqual(request.rows, []);
  assert.equal(request.max_rows, 30);
  assert.equal(request.reference_column, "queue");
  assert.deepEqual(request.baseline, { mode: "adapter_off", alias: null });
  const candidate = request.model_configs.find((config) => config.alias === request.model_alias);
  assert.equal(candidate?.model, "local");
  const provider = request.model_providers.find((p) => p.name === candidate?.provider);
  assert.equal(provider?.is_local, true);

  const missing = buildLoopBenchmarkRequest({
    benchmark: { ...benchmark, reference_column: "" },
    configs: [],
    holdoutPath: "/x",
  });
  assert.equal(missing.request, null);
  assert.match(missing.errors.join(" "), /reference column/);
});

test("a benchmark that scored nothing is an error, not a 0% round", () => {
  const result = {
    candidate: { total: 3, scored: 0, passed: 0, errors: 3, score: null },
    cancelled: false,
    failed_examples: [],
    failed_total: 3,
  };
  assert.throws(() => loopScoreFromResult(result), /errored/);
  const scored = loopScoreFromResult({
    ...result,
    candidate: { total: 3, scored: 3, passed: 2, errors: 0, score: 2 / 3 },
    baseline: { total: 3, scored: 3, passed: 1, errors: 0, score: 1 / 3 },
    failed_examples: [
      { prompt: "p", reference: "r", output: "o", reason: "Answer does not contain the reference.", row: { a: 1 } },
    ],
    failed_total: 1,
  });
  assert.equal(scored.score, 2 / 3);
  assert.equal(scored.baseline, 1 / 3);
  assert.deepEqual(scored.failedExamples, [
    { prompt: "p", reference: "r", output: "o", row: { a: 1 } },
  ]);
});
