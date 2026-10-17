// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// Read as source for the same reason as spec-fallback-partial-offload-copy.test.ts.
const settings = readSrc("features/chat/chat-settings-sheet.tsx");

test("a drafter that ran out of VRAM does not blame the llama.cpp build", () => {
  const branch = settings.match(/case "drafter_oom":[\s\S]*?return `([^`]+)`;/);
  assert.ok(branch, "specFallbackMessage has no drafter_oom case");
  const copy = branch[1];
  assert.doesNotMatch(copy, /llama\.cpp|build|update/i);
  assert.match(copy, /memory/i);
  assert.match(copy, /Manual/);
});
