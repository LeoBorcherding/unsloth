// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the Code font row's (Default) label names the font chat code falls back to", () => {
  const controls = readSrc("features/settings/components/appearance-custom-controls.tsx");
  const label = controls.match(/const DEFAULT_FONT_NAMES = \{[^}]*\bcode: "([^"]+)"/)?.[1];
  assert.ok(label, "DEFAULT_FONT_NAMES.code not found");

  const fallbacks = [...readSrc("index.css").matchAll(/var\(--custom-code-font, "([^"]+)"/g)].map((m) => m[1]);
  assert.ok(fallbacks.length > 0, "no --custom-code-font fallbacks in index.css");
  for (const font of fallbacks) assert.equal(label, font);

  const store = readSrc("features/settings/stores/appearance-custom-store.ts");
  const stack = store.match(/"--custom-code-font",\s*c\.codeFont \? `"\$\{c\.codeFont\}", "([^"]+)"/)?.[1];
  assert.equal(stack, label);
});
