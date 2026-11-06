// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_MERGE_SETTINGS,
  type MergeSettings,
  mergeBlockReason,
  mergeRequest,
  rowsForMethod,
} from "../src/features/export/lib/merge.ts";

function settings(patch: Partial<MergeSettings>): MergeSettings {
  return { ...DEFAULT_MERGE_SETTINGS, ...patch };
}

const TWO = [
  { model: " org/a ", weight: 0.6, density: 0.5 },
  { model: "org/b", weight: 0.4, density: 0.7 },
];

test("TIES sends weights, densities and the base", () => {
  const s = settings({ method: "ties", rows: TWO, baseModel: " org/base " });
  assert.equal(mergeBlockReason(s), null);
  assert.deepEqual(mergeRequest(s), {
    method: "ties",
    models: [
      { model: "org/a", weight: 0.6, density: 0.5 },
      { model: "org/b", weight: 0.4, density: 0.7 },
    ],
    base_model: "org/base",
    t: null,
    dtype: "bfloat16",
    tokenizer_source: "base",
    output_name: "merged-model",
    cuda: false,
  });
});

test("linear and task arithmetic send no density, linear no base", () => {
  const linear = mergeRequest(
    settings({ method: "linear", rows: TWO, baseModel: "org/base" }),
  );
  assert.equal(linear.base_model, null);
  assert.deepEqual(linear.models[0], { model: "org/a", weight: 0.6 });
  const task = mergeRequest(
    settings({ method: "task_arithmetic", rows: TWO, baseModel: "org/base" }),
  );
  assert.equal(task.base_model, "org/base");
  assert.deepEqual(task.models[1], { model: "org/b", weight: 0.4 });
});

test("SLERP sends t, the first model as base and no weights", () => {
  const s = settings({ method: "slerp", rows: TWO, t: 0.3 });
  assert.equal(mergeBlockReason(s), null);
  const request = mergeRequest(s);
  assert.equal(request.base_model, "org/a");
  assert.equal(request.t, 0.3);
  assert.deepEqual(request.models, [{ model: "org/a" }, { model: "org/b" }]);
});

test("the merge is blocked where the backend would refuse it", () => {
  const block = (patch: Partial<MergeSettings>) =>
    mergeBlockReason(settings({ rows: TWO, baseModel: "org/base", ...patch }));
  assert.match(block({ baseModel: "" }) ?? "", /needs a base model/);
  assert.match(block({ baseModel: "org/a" }) ?? "", /base field only/);
  assert.match(
    block({ method: "linear", rows: [TWO[0]] }) ?? "",
    /at least two/,
  );
  assert.match(
    block({ method: "slerp", rows: [...TWO, { ...TWO[0], model: "c" }] }) ?? "",
    /exactly two/,
  );
  assert.match(block({ method: "slerp", t: 1.5 }) ?? "", /between 0 and 1/);
  assert.match(
    block({ rows: [TWO[0], { ...TWO[1], density: 0 }] }) ?? "",
    /Density/,
  );
  assert.match(
    block({ rows: [TWO[0], { ...TWO[1], weight: Number.NaN }] }) ?? "",
    /weight/,
  );
  assert.match(block({ rows: [TWO[0], TWO[0]] }) ?? "", /only once/);
  assert.match(
    block({ rows: [TWO[0], { ...TWO[1], model: " " }] }) ?? "",
    /every model/,
  );
  assert.match(block({ outputName: "../x" }) ?? "", /plain folder name/);
  assert.match(block({ outputName: ".hidden" }) ?? "", /plain folder name/);
});

test("tokenizer source follows the setting, and linear only sends the union", () => {
  const ties = mergeRequest({
    ...DEFAULT_MERGE_SETTINGS,
    rows: [
      { model: "a", weight: 0.5, density: 0.5 },
      { model: "b", weight: 0.5, density: 0.5 },
    ],
    baseModel: "base",
  });
  assert.equal(ties.tokenizer_source, "base");
  const linear = { ...DEFAULT_MERGE_SETTINGS, method: "linear" as const };
  assert.equal(mergeRequest(linear).tokenizer_source, null);
  assert.equal(
    mergeRequest({ ...linear, tokenizerSource: "union" }).tokenizer_source,
    "union",
  );
});

test("switching to SLERP always leaves exactly two rows", () => {
  const one = [{ model: "org/a", weight: 1, density: 0.5 }];
  assert.deepEqual(
    rowsForMethod("slerp", one).map((r) => r.model),
    ["org/a", ""],
  );
  const three = [...one, ...one, ...one].map((r, i) => ({ ...r, model: `m${i}` }));
  assert.equal(rowsForMethod("slerp", three).length, 2);
  assert.equal(rowsForMethod("ties", one), one);
});
