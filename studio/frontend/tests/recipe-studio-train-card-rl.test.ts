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
const {
  buildTrainCardRlMapping,
  formatHoldoutSummary,
  getTrainCardOutput,
  parseTrainCardSettings,
  pickTrainProgressMetric,
  rewardBreakdown,
  rewardChoices,
  summarizeAdvancedTrainSettings,
  switchTrainObjective,
  trainCardChips,
} = await import("../src/features/recipe-studio/utils/train-card.ts");

type TrainCardConfig = ReturnType<typeof makeTrainConfig>;

const COLUMNS = ["question", "answer", "careful", "quick", "judge", "score_a", "score_b"];
const LIBRARY = [
  { name: "exact-answer", compareTo: "answer" },
  { name: "strict-xml-format", compareTo: null },
  { name: "numeric-close", compareTo: "answer" },
];

function card(patch: Partial<TrainCardConfig>): TrainCardConfig {
  return {
    ...makeTrainConfig("t1", []),
    baseModel: "unsloth/Qwen3-0.6B",
    ...patch,
  };
}

test("a new card trains SFT and starts GRPO with the notebook rewards", () => {
  const fresh = makeTrainConfig("t1", []);
  assert.equal(fresh.objective, "sft");
  assert.deepEqual(fresh.rewards, [
    { name: "strict-xml-format", weight: 1 },
    { name: "exact-answer", weight: 1 },
  ]);
  assert.match(fresh.systemPrompt, /<reasoning>[\s\S]*<answer>/);
  assert.equal(fresh.numGenerations, "4");
});

test("switching objective moves only the fields still at their defaults", () => {
  const fresh = makeTrainConfig("t1", []);
  assert.deepEqual(switchTrainObjective(fresh, "grpo"), {
    objective: "grpo",
    learningRate: "5e-6",
    maxSteps: "50",
    contextLength: "1024",
  });
  const tuned = { ...fresh, learningRate: "1e-4", maxSteps: "30" };
  assert.deepEqual(switchTrainObjective(tuned, "orpo"), {
    objective: "orpo",
  });
  const grpo = { ...fresh, ...switchTrainObjective(fresh, "grpo") };
  assert.deepEqual(switchTrainObjective(grpo, "dpo"), {
    objective: "dpo",
    maxSteps: "60",
    contextLength: "2048",
  });
});

test("GRPO needs a prompt, rewards, and the answer column its rewards compare to", () => {
  const ready = parseTrainCardSettings(
    card({ objective: "grpo", promptColumn: "question", responseColumn: "answer" }),
    COLUMNS,
    LIBRARY,
  );
  assert.deepEqual(ready.errors, []);
  assert.equal(ready.settings.numGenerations, 4);
  assert.equal(ready.settings.maxCompletionLength, 200);
  assert.equal(ready.settings.beta, null);
  assert.equal(ready.settings.pairs, null);
  assert.deepEqual(buildTrainCardRlMapping(ready.settings), {
    question: "prompt",
    answer: "answer",
  });

  const noAnswer = parseTrainCardSettings(
    card({ objective: "grpo", promptColumn: "question" }),
    COLUMNS,
    LIBRARY,
  );
  assert.deepEqual(noAnswer.errors, [
    "Pick the answer column: exact-answer checks against it.",
  ]);

  const formatOnly = parseTrainCardSettings(
    card({
      objective: "grpo",
      promptColumn: "question",
      rewards: [{ name: "strict-xml-format", weight: 1 }],
    }),
    COLUMNS,
    LIBRARY,
  );
  assert.deepEqual(formatOnly.errors, []);
  assert.deepEqual(buildTrainCardRlMapping(formatOnly.settings), {
    question: "prompt",
  });

  const broken = parseTrainCardSettings(
    card({
      objective: "grpo",
      promptColumn: "question",
      responseColumn: "answer",
      rewards: [
        { name: "gone", weight: 1 },
        { name: "strict-xml-format", weight: Number.NaN },
      ],
      numGenerations: "1",
      beta: "-1",
    }),
    COLUMNS,
    LIBRARY,
  );
  assert.deepEqual(broken.errors, [
    "Reward gone is not in the reward library.",
    "Reward strict-xml-format needs a numeric weight.",
    "Beta must be 0 or more.",
    "Generations must be a whole number from 2 to 64.",
  ]);
  assert.deepEqual(
    parseTrainCardSettings(
      card({ objective: "grpo", promptColumn: "question", rewards: [] }),
      COLUMNS,
      LIBRARY,
    ).errors,
    ["Pick at least one reward."],
  );
});

test("DPO and ORPO build preference pairs from two answer columns", () => {
  const base = card({
    objective: "dpo",
    promptColumn: "question",
    pairA: "careful",
    pairB: "quick",
  });
  const always = parseTrainCardSettings(base, COLUMNS);
  assert.deepEqual(always.errors, []);
  assert.deepEqual(always.settings.pairs, {
    answer_a: "careful",
    answer_b: "quick",
    rule: "a_better",
    winner_column: null,
    score_a_column: null,
    score_b_column: null,
  });
  assert.equal(always.settings.responseColumn, "");
  assert.deepEqual(buildTrainCardRlMapping(always.settings), {
    question: "prompt",
    chosen: "chosen",
    rejected: "rejected",
  });

  const judged = parseTrainCardSettings(
    { ...base, objective: "orpo", pairRule: "winner", winnerColumn: "judge" },
    COLUMNS,
  );
  assert.deepEqual(judged.errors, []);
  assert.equal(judged.settings.pairs?.winner_column, "judge");

  assert.deepEqual(
    parseTrainCardSettings({ ...base, pairRule: "winner" }, COLUMNS).errors,
    ["Pick the column that names the better answer."],
  );
  assert.deepEqual(
    parseTrainCardSettings(
      { ...base, pairRule: "scores", scoreAColumn: "score_a", scoreBColumn: "gone" },
      COLUMNS,
    ).errors,
    ["Column gone is not in this recipe's output."],
  );
  assert.deepEqual(
    parseTrainCardSettings({ ...base, pairB: "careful", beta: "0" }, COLUMNS).errors,
    ["Answer A and answer B must be different columns.", "Beta must be a number above 0."],
  );
});

test("RL settings survive save and reopen", () => {
  const template = readFileSync(
    new URL(
      "../src/features/data-recipes/learning-recipes/text-to-sql.json",
      import.meta.url,
    ),
    "utf8",
  );
  const snapshot = importRecipePayload(template).snapshot;
  assert.ok(snapshot);
  const train: TrainCardConfig = {
    ...makeTrainConfig("n999", Object.values(snapshot.configs)),
    objective: "grpo",
    promptColumn: "sql_prompt",
    responseColumn: "sql",
    rewards: [{ name: "exact-answer", weight: 2 }],
    numGenerations: "6",
    systemPrompt: "Answer in <answer> tags.",
    pairRule: "scores",
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
  const built = buildRecipePayload(
    configs,
    nodes,
    snapshot.edges,
    snapshot.processors,
    snapshot.layoutDirection,
    snapshot.auxNodePositions,
  );
  const reopened = importRecipePayload(JSON.stringify(built.payload));
  const restored = Object.values(reopened.snapshot?.configs ?? {}).find(
    (config) => config.kind === "train",
  );
  assert.ok(restored && restored.kind === "train");
  assert.equal(restored.objective, "grpo");
  assert.deepEqual(restored.rewards, [{ name: "exact-answer", weight: 2 }]);
  assert.equal(restored.numGenerations, "6");
  assert.equal(restored.systemPrompt, "Answer in <answer> tags.");
  assert.equal(restored.pairRule, "scores");
});

test("cards saved before RL open as SFT", () => {
  const legacy = JSON.stringify({
    recipe: { model_providers: [], model_configs: [], columns: [], processors: [] },
    run: { rows: 10, preview: true },
    ui: {
      layout_direction: "LR",
      nodes: [
        {
          id: "train_1",
          x: 0,
          y: 0,
          node_type: "train",
          name: "train_1",
          train: { baseModel: "unsloth/Qwen2.5-0.5B-Instruct", promptColumn: "q" },
        },
      ],
      edges: [],
    },
  });
  const restored = Object.values(importRecipePayload(legacy).snapshot?.configs ?? {}).find(
    (config) => config.kind === "train",
  );
  assert.ok(restored && restored.kind === "train");
  assert.equal(restored.objective, "sft");
  assert.equal(restored.rewards.length, 2);
  assert.equal(restored.pairRule, "a_better");
});

test("the card face names the objective's inputs", () => {
  assert.deepEqual(
    trainCardChips(
      card({ objective: "grpo", promptColumn: "question", responseColumn: "answer", maxSteps: "40" }),
    ),
    ["question → answer", "strict-xml-format + exact-answer", "40 steps", "10% held out"],
  );
  assert.deepEqual(
    trainCardChips(
      card({ objective: "dpo", pairA: "careful", pairB: "quick", pairRule: "winner" }),
    ),
    ["careful vs quick", "winner column", "60 steps", "10% held out"],
  );
  assert.deepEqual(
    trainCardChips(
      card({
        objective: "grpo",
        rewards: [
          { name: "a", weight: 1 },
          { name: "b", weight: 1 },
          { name: "c", weight: 1 },
        ],
      }),
    ),
    ["3 rewards", "60 steps", "10% held out"],
  );
});

test("progress shows reward for GRPO, accuracy for DPO, loss otherwise", () => {
  assert.deepEqual(pickTrainProgressMetric("grpo", 0.01, { reward: 1.25 }), {
    label: "mean reward",
    value: 1.25,
  });
  assert.deepEqual(
    pickTrainProgressMetric("dpo", 0.69, { "rewards/accuracies": 0.75 }),
    { label: "accuracy", value: 0.75 },
  );
  assert.deepEqual(pickTrainProgressMetric("grpo", 0.02, undefined), {
    label: "loss",
    value: 0.02,
  });
  assert.equal(pickTrainProgressMetric("sft", null, undefined), null);
});

test("a finished preference run says how many rows had no winner", () => {
  const run = {
    jobId: "job-1",
    executionId: "exec-1",
    sourceJobId: "recipe-1",
    artifactPath: "/data/r",
    baseModel: "unsloth/Qwen3-0.6B",
    outputName: "dpo",
    objective: "dpo" as const,
    promptColumn: "question",
    responseColumn: "",
    holdoutPercent: 10,
    seed: 3407,
    trainPath: "/data/r/splits/holdout-10-seed3407/pairs-abc.parquet",
    holdoutPath: "/data/r/splits/holdout-10-seed3407/holdout.parquet",
    holdoutRowIds: [1],
    trainRows: 40,
    holdoutRows: 5,
    pairsDropped: 3,
    phase: "completed",
    outputDir: "/outputs/dpo",
    startedAt: 1,
  };
  assert.equal(
    formatHoldoutSummary(run),
    "40 train pairs / 5 held out · 3 ties left out",
  );
  assert.equal(getTrainCardOutput(card({ lastRun: run }))?.objective, "dpo");
});

test("Advanced names RL settings changed from the objective's defaults", () => {
  const grpo = card({ objective: "grpo", learningRate: "5e-6", maxSteps: "50", contextLength: "1024" });
  assert.equal(summarizeAdvancedTrainSettings(grpo), "Defaults");
  assert.equal(
    summarizeAdvancedTrainSettings({ ...grpo, numGenerations: "8" }),
    "Changed: generations",
  );
  assert.equal(
    summarizeAdvancedTrainSettings({ ...grpo, learningRate: "2e-4" }),
    "Changed: learning rate",
  );
  assert.equal(
    summarizeAdvancedTrainSettings({ ...card({ objective: "dpo", learningRate: "5e-6" }), beta: "0.2" }),
    "Changed: beta",
  );
});

test("the reward list shows usable library rewards and keeps a picked one that went missing", () => {
  const record = (name: string, patch: Record<string, unknown> = {}) => ({
    name,
    kind: "rule" as const,
    description: `${name} description`,
    source: "bundled" as const,
    valid: true,
    shadowed: false,
    error: null,
    rule: null,
    ...patch,
  });
  const choices = rewardChoices(
    [
      record("exact-answer"),
      record("broken", { valid: false }),
      record("old", { shadowed: true }),
      record("code", { kind: "python" }),
    ],
    [
      { name: "exact-answer", weight: 1 },
      { name: "deleted", weight: 1 },
    ],
  );
  assert.deepEqual(
    choices.map((choice) => [choice.name, choice.missing]),
    [
      ["exact-answer", false],
      ["deleted", true],
    ],
  );
});

test("the mean reward is the weighted sum of each reward's own mean", () => {
  // exact-answer scores 2.0 on a match, so 5 of 8 right is 1.25 on its own.
  const rl = {
    reward: 1.25 + 0.5 * 0.5,
    "rewards/exact_answer/mean": 1.25,
    "rewards/soft_xml_format/mean": 0.5,
  };
  const rewards = [
    { name: "soft-xml-format", weight: 0.5 },
    { name: "exact-answer", weight: 1 },
    { name: "numeric-close", weight: 1 },
  ];
  assert.deepEqual(rewardBreakdown(rewards, rl), [
    { name: "soft-xml-format", value: 0.5 },
    { name: "exact-answer", value: 1.25 },
  ]);
  const weighted = rewardBreakdown(rewards, rl).reduce(
    (sum, item) =>
      sum + item.value * (rewards.find((r) => r.name === item.name)?.weight ?? 0),
    0,
  );
  assert.equal(weighted, pickTrainProgressMetric("grpo", null, rl)?.value);
  assert.deepEqual(rewardBreakdown(rewards, undefined), []);
});
