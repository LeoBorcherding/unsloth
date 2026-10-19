// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { loadWithStubs, type StubElement, stubJsxRuntime } from "./helpers/module-stubs.ts";

// The API already takes subset and split and the dialog already sends them, but nothing in the
// Hugging Face tab could set them, so a multi-config dataset could only ever load its default.

const DIALOG = new URL(
  "../src/features/recipe-studio/dialogs/seed/seed-dialog.tsx",
  import.meta.url,
);
const FIELDS = new URL(
  "../src/features/recipe-studio/dialogs/seed/hf-subset-split-fields.tsx",
  import.meta.url,
);

function Input(): null {
  return null;
}
function FieldLabel(): null {
  return null;
}

function inputs(node: unknown, out: StubElement[] = []): StubElement[] {
  if (!node || typeof node !== "object") return out;
  if (Array.isArray(node)) {
    for (const child of node) inputs(child, out);
    return out;
  }
  const element = node as StubElement;
  if (element.type === Input) out.push(element);
  inputs(element.props?.children, out);
  return out;
}

test("the Hugging Face tab renders Subset and Split fields, locked while a load runs", () => {
  const dialog = readFileSync(DIALOG, "utf8");
  // An edit during a load would be overwritten by the older response's subset and split.
  assert.match(
    dialog,
    /<HfSubsetSplitFields\s+config=\{config\}\s+onUpdate=\{onUpdate\}\s+disabled=\{isInspecting\}\s*\/>/,
  );
});

test("the fields are disabled while inspecting", () => {
  const { HfSubsetSplitFields } = loadWithStubs<{
    HfSubsetSplitFields: (props: Record<string, unknown>) => StubElement;
  }>(FIELDS, {
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/ui/input": { Input },
    "../shared/field-label": { FieldLabel },
  });
  const tree = HfSubsetSplitFields({
    config: { id: "seed", hf_subset: "", hf_split: "" },
    onUpdate: () => undefined,
    disabled: true,
  });
  const fields = inputs(tree);
  assert.equal(fields.length, 2);
  for (const field of fields) assert.equal(field.props.disabled, true);
});

test("editing subset or split sets it and drops the stale load", () => {
  const { HfSubsetSplitFields } = loadWithStubs<{
    HfSubsetSplitFields: (props: Record<string, unknown>) => StubElement;
  }>(FIELDS, {
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/ui/input": { Input },
    "../shared/field-label": { FieldLabel },
  });
  const patches: Record<string, unknown>[] = [];
  const tree = HfSubsetSplitFields({
    config: { id: "seed", hf_subset: "", hf_split: "train", hf_path: "datasets/openai/gsm8k/main/train-*.parquet" },
    onUpdate: (patch: Record<string, unknown>) => patches.push(patch),
  });
  const [subset, split] = inputs(tree);
  assert.equal(split?.props.value, "train");

  (subset?.props.onChange as (e: unknown) => void)({ target: { value: "socratic" } });
  (split?.props.onChange as (e: unknown) => void)({ target: { value: "test" } });

  assert.equal(patches[0]?.hf_subset, "socratic");
  assert.equal(patches[1]?.hf_split, "test");
  for (const patch of patches) {
    assert.equal(patch.hf_path, "");
    assert.deepEqual(patch.seed_columns, []);
  }
});

test("a changed subset or split counts as a new load", () => {
  const dialog = readFileSync(DIALOG, "utf8");
  assert.match(dialog, /return hfLoadKey\(dataset, config\.hf_token, config\.hf_subset, config\.hf_split\)/);
});
