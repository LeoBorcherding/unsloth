// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as defaults from "../src/features/images/image-generation-defaults.ts";

import { readSrc } from "./helpers/kit.ts";

const { DEFAULT_RESOLUTION } = defaults;
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

test("only the native rule is known at pick time, so a quantised Qwen-Image-2.1 canvas is left alone", () => {
  assert.ok(nativeCanvasFor);
  // The quant canvas needs the engaged build; a resident Qwen quant at 512 has no native canvas, so 512 reads as chosen.
  assert.equal(nativeCanvasFor("Qwen/Qwen-Image-2.1"), null);
  assert.equal(nativeCanvasFor("stabilityai/stable-diffusion-xl-base-1.0"), null);
  assert.deepEqual(nativeCanvasFor("local/SDXL_Turbo"), SMALL);
});

test("the Images page pick path applies the picked model's native canvas", () => {
  const src = readSrc("features/images/images-page.tsx");
  const start = src.indexOf("const applyImageModelDefaults = useCallback");
  const pick = src.slice(start, src.indexOf("const dismissLoadToast", start));
  assert.match(pick, /canvasForPick\(/);
  assert.match(pick, /nativeCanvasFor\(key\)/);
  assert.match(pick, /setCanvas\(canvas\)/);
  const revert = src.slice(src.indexOf("const revertPick = useCallback"), src.indexOf("}, [setCanvas]);"));
  assert.match(revert, /setCanvas\(r\.canvas\.prev\)/);
});
