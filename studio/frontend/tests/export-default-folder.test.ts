// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// buildRelativeSaveDirectory lives in export-page.tsx, which this runner cannot import, so the
// two helpers are sliced out of the shipped source and transpiled on their own.

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";

type Build = (
  exportMethod: string | null,
  sourceMode: "checkpoint" | "model",
  sourceBaseModelName: string,
  selectedModelIdx: string | null,
  checkpoint: string | null,
) => string;

function loadBuild(): Build {
  const source = ts.createSourceFile(
    "export-page.tsx",
    readSrc("features/export/export-page.tsx"),
    ts.ScriptTarget.ES2022,
    true,
    ts.ScriptKind.TSX,
  );
  const wanted = new Set(["safePathSegment", "buildRelativeSaveDirectory"]);
  const pieces = source.statements
    .filter(
      (node): node is ts.FunctionDeclaration =>
        ts.isFunctionDeclaration(node) && wanted.has(node.name?.text ?? ""),
    )
    .map((node) => node.getText(source));
  assert.equal(pieces.length, wanted.size);
  const { outputText } = ts.transpileModule(
    `${pieces.join("\n")}\nreturn buildRelativeSaveDirectory;`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  return new Function(outputText)() as Build;
}

const build = loadBuild();
const BASE = "unsloth/Llama-3.2-1B-Instruct";
const RUN_A = "unsloth_Llama-3.2-1B-Instruct_1771227800";
const RUN_B = "unsloth_Llama-3.2-1B-Instruct_1771999999";

test("a GGUF export of the same step from two runs defaults to two folders", () => {
  const a = build("gguf", "checkpoint", BASE, RUN_A, "checkpoint-60");
  const b = build("gguf", "checkpoint", BASE, RUN_B, "checkpoint-60");
  assert.notEqual(a, b);
  assert.equal(a, `${RUN_A}/checkpoint-60-GGUF`);
});

test("merged and LoRA exports of one checkpoint default to different folders", () => {
  const merged = build("merged", "checkpoint", BASE, RUN_A, "checkpoint-60");
  const lora = build("lora", "checkpoint", BASE, RUN_A, "checkpoint-60");
  assert.notEqual(merged, lora);
  assert.equal(merged, `${RUN_A}/checkpoint-60-merged`);
  assert.equal(lora, `${RUN_A}/checkpoint-60-adapter`);
});

test("a run's final model keeps its flat GGUF folder", () => {
  assert.equal(
    build("gguf", "checkpoint", BASE, RUN_A, RUN_A),
    `${RUN_A}-GGUF`,
  );
});

test("a model source is unchanged", () => {
  assert.equal(
    build("gguf", "model", BASE, null, null),
    "unsloth-Llama-3.2-1B-Instruct-GGUF",
  );
  assert.equal(
    build("merged", "model", BASE, null, null),
    "unsloth-Llama-3.2-1B-Instruct-merged",
  );
  assert.equal(
    build("lora", "model", BASE, null, null),
    "unsloth-Llama-3.2-1B-Instruct-adapter",
  );
});
