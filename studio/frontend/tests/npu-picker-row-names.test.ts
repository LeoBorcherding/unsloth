// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const src = readFileSync(
  new URL(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
    import.meta.url,
  ),
  "utf8",
);

function body(name: string): string {
  const start = src.indexOf(`const ${name} = (`);
  assert.ok(start > 0, `${name} not found`);
  const end = src.indexOf("\n  const render", start + 1);
  return src.slice(start, end > 0 ? end : undefined);
}

test("a loaded NPU model's row is named by its Hugging Face repo, not the Lemonade id", () => {
  const row = body("renderLoadedRow");
  assert.match(row, /const label = npuName\(checkpoint\) \?\? npuName\(entry\.id\) \?\? entry\.id;/);
  assert.match(row, /label=\{label\}/);
  assert.doesNotMatch(row, /label=\{entry\.id\}/);
  assert.match(row, /Inference settings for \$\{label\}/);
  assert.match(row, /More options for \$\{label\}/);
});

test("every NPU row label in an aria-label is declared in that row, so it never falls back to window.name", () => {
  const row = body("renderNpuRow");
  for (const [, ident] of row.matchAll(/(?:Inference settings|More options|Delete) for \$\{(\w+)\}/g)) {
    assert.match(row, new RegExp(`const ${ident}\\b`), `${ident} is not declared in renderNpuRow`);
  }
});
