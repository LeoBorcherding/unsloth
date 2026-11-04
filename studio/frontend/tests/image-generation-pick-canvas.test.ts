// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as defaults from "../src/features/images/image-generation-defaults.ts";

import { readSrc } from "./helpers/kit.ts";

const { DEFAULT_RESOLUTION, resolutionFor } = defaults;
const canvasForPick = (defaults as Record<string, unknown>).canvasForPick as
  | ((
      current: { width: number; height: number },
      previousDefault: { width: number; height: number },
      nextDefault: { width: number; height: number },
    ) => { width: number; height: number })
  | undefined;
const nativeCanvasFor = (defaults as Record<string, unknown>).nativeCanvasFor as
  | ((repoId: string) => { width: number; height: number } | null)
  | undefined;
const canvasRevertFor = (defaults as Record<string, unknown>).canvasRevertFor as
  | ((
      owned: { prev: { width: number; height: number }; applied: { width: number; height: number } } | undefined,
      current: { width: number; height: number },
      picked: { width: number; height: number },
    ) => { prev: { width: number; height: number }; applied: { width: number; height: number } } | undefined)
  | undefined;
const SMALL = { width: 512, height: 512 };

test("picking SDXL-Turbo moves an untouched 1024 canvas to 512, and back on the next pick", () => {
  assert.ok(canvasForPick && nativeCanvasFor, "canvasForPick / nativeCanvasFor are exported");
  const turbo = nativeCanvasFor("unsloth/sdxl-turbo") ?? DEFAULT_RESOLUTION;
  assert.deepEqual(canvasForPick(DEFAULT_RESOLUTION, DEFAULT_RESOLUTION, turbo), SMALL);
  const flux = nativeCanvasFor("black-forest-labs/FLUX.1-dev") ?? DEFAULT_RESOLUTION;
  assert.deepEqual(canvasForPick(SMALL, turbo, flux), DEFAULT_RESOLUTION);
});

test("a size the user chose survives a pick", () => {
  assert.ok(canvasForPick);
  const chosen = { width: 768, height: 1344 };
  assert.deepEqual(canvasForPick(chosen, DEFAULT_RESOLUTION, SMALL), chosen);
  assert.deepEqual(canvasForPick(chosen, SMALL, DEFAULT_RESOLUTION), chosen);
});

test("only the native rule is known at pick time; the quant canvas comes from an engaged build", () => {
  assert.ok(canvasForPick && nativeCanvasFor);
  assert.equal(nativeCanvasFor("Qwen/Qwen-Image-2.1"), null);
  assert.equal(nativeCanvasFor("stabilityai/stable-diffusion-xl-base-1.0"), null);
  assert.deepEqual(nativeCanvasFor("local/SDXL_Turbo"), SMALL);
  // A resident explicit Qwen-Image-2.1 quant defaults to 512, so picking FLUX from an untouched form goes to 1024.
  const resident = resolutionFor("Qwen/Qwen-Image-2.1", { modelKind: "pipeline", transformerQuant: "fp8" });
  assert.deepEqual(resident, SMALL);
  assert.deepEqual(canvasForPick(SMALL, resident, nativeCanvasFor("black-forest-labs/FLUX.1-dev") ?? DEFAULT_RESOLUTION), DEFAULT_RESOLUTION);
});

test("the Images page pick path applies the picked model's native canvas", () => {
  const src = readSrc("features/images/images-page.tsx");
  const start = src.indexOf("const applyImageModelDefaults = useCallback");
  const pick = src.slice(start, src.indexOf("const dismissLoadToast", start));
  assert.match(pick, /canvasForPick\(/);
  assert.match(pick, /nativeCanvasFor\(key\)/);
  assert.match(pick, /pendingModelDefaults\?\.canvas \?\? residentCanvas/);
  assert.match(pick, /setCanvas\(canvas\)/);
  // Only a canvas the pick moved is the load's to replace; a kept 1024 must not become a quant's 512.
  assert.match(pick, /pickCanvas\.current = canvas === current \? null : canvas/);
  const revert = src.slice(src.indexOf("const revertPick = useCallback"), src.indexOf("}, [setCanvas]);"));
  assert.match(revert, /setCanvas\(r\.canvas\.prev\)/);
});

test("a finished pick load moves an untouched canvas to the engaged build's", () => {
  // A picked Qwen-Image-2.1 quant only learns its 512 from the engaged build, after the load.
  const src = readSrc("features/images/images-page.tsx");
  const ready = src.slice(src.indexOf('if (p.phase === "ready")'), src.indexOf('if (p.phase === "error")'));
  assert.match(ready, /canvasForPick\(current, pickedCanvas, engaged\)/);
  assert.match(ready, /transformerQuantSource: loaded\.resolved\?\.transformer_quant\?\.source/);
  assert.match(ready, /!pickRecipeSuperseded\.current\?\.\(\)/);
});

test("a failed pick puts back only a canvas a pick moved", () => {
  assert.ok(canvasForPick && canvasRevertFor);
  // First staged pick moves 1024 to 512: rollback owns it.
  const start = { ...DEFAULT_RESOLUTION };
  const first = canvasForPick(start, DEFAULT_RESOLUTION, SMALL);
  const owned = canvasRevertFor(undefined, start, first);
  assert.deepEqual(owned, { prev: DEFAULT_RESOLUTION, applied: SMALL });
  // The user then picks 768x768 and a second pick keeps it: a failed second load must not undo that edit.
  const chosen = { width: 768, height: 768 };
  const second = canvasForPick(chosen, SMALL, DEFAULT_RESOLUTION);
  assert.equal(second, chosen);
  assert.equal(canvasRevertFor(owned, chosen, second), undefined);
  // An untouched chain keeps the size from before the first pick.
  const back = canvasForPick(first, SMALL, DEFAULT_RESOLUTION);
  assert.deepEqual(canvasRevertFor(owned, first, back), { prev: DEFAULT_RESOLUTION, applied: DEFAULT_RESOLUTION });
});
