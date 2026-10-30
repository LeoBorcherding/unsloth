// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isDiffusionTask } from "../src/features/model-picker/components/model-selector/model-capabilities.ts";
import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/scheduled-tasks/task-dialog.tsx");

test("the task model picker leaves out image and video models", () => {
  // An image GGUF listed under "On this device" failed every run with model_load_failed.
  assert.match(dialog, /inventory\.cachedGguf\.rows\) \{\s*if \(isDiffusionTask\(repo\.task\)\) continue;/);
  assert.match(dialog, /localModels\.rows\.filter\(\(row\) => !isDiffusionTask\(row\.task\)\)/);
  assert.match(dialog, /chatLocalModelOptions\(chatRows\)/);
});

test("the tasks it drops are the generation ones", () => {
  assert.ok(isDiffusionTask("text-to-image"));
  assert.ok(isDiffusionTask("text-to-video"));
  assert.ok(!isDiffusionTask("text-generation"));
  assert.ok(!isDiffusionTask(null));
});
