// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A scheduled task sends a text prompt, so its Model list must leave out cached image and speech repos.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const IMPORTS_PREDICATE =
  /import \{ isChattableCachedRepo \} from "@\/features\/chat\/api\/chat-adapter";/;
const FILTERED_LOOP = /if \(isChattableCachedRepo\(repo\) &&/;
const EXPORTED_PREDICATE = /^export function isChattableCachedRepo\(/m;

function read(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

test("the task dialog runs cached GGUF repos through the chat picker's predicate", () => {
  const dialog = read("../src/features/scheduled-tasks/task-dialog.tsx");
  assert.match(dialog, IMPORTS_PREDICATE);
  const loop = dialog.slice(
    dialog.indexOf("for (const repo of inventory.cachedGguf.rows)"),
  );
  assert.match(loop.slice(0, 200), FILTERED_LOOP);
});

test("the predicate is the exported one that drops image and speech tasks", () => {
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, EXPORTED_PREDICATE);
  const start = adapter.indexOf("const NON_CHAT_TASKS");
  const tasks = adapter.slice(start, adapter.indexOf("]);", start));
  for (const task of [
    "text-to-image",
    "text-to-speech",
    "automatic-speech-recognition",
  ]) {
    assert.ok(tasks.includes(`"${task}"`), task);
  }
});
