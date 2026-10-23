// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import {
  parseRemoveBackgroundResult,
  REMOVE_BACKGROUND_ROUTE,
  segmenterMissing,
} from "../src/features/images/lib/remove-background.ts";
import { recipeWorkflowLabel } from "../src/features/images/workflows.ts";

test("the route sits beside the segment route", () => {
  assert.equal(REMOVE_BACKGROUND_ROUTE, "/api/inference/images/remove-background");
});

test("the parse returns the saved record and refuses an answer without one", () => {
  const image = { id: "abc", url: "/api/inference/images/gallery/abc/file", prompt: "a car" };
  assert.equal(parseRemoveBackgroundResult({ image, mask: "data:image/png;base64,x" }).id, "abc");
  for (const raw of [null, {}, { image: null }, { image: { id: 3, url: "/x" } }, { b64_png: "data:" }]) {
    assert.throws(() => parseRemoveBackgroundResult(raw), /no image/);
  }
});

test("only a missing segmenter offers the download", () => {
  assert.equal(segmenterMissing("The SAM 3.1 segmenter is not downloaded. Download unsloth/sam3.1 first."), true);
  assert.equal(segmenterMissing('Found nothing matching "elephant".'), false);
  assert.equal(segmenterMissing("The image model is generating. Try again when it finishes."), false);
});

test("a cut-out's recipe reads as Remove background", () => {
  assert.equal(recipeWorkflowLabel("remove_background"), "Remove background");
});

test("every gallery image menu offers Remove background and saves through the API", () => {
  const page = readSrc("features/images/images-page.tsx");
  const items = page.slice(page.indexOf("const editImageItem"));
  assert.match(items.slice(0, items.indexOf(");\n")), /removeGalleryBackground\(image\)/);
  assert.match(page, /removeBackground\(\{ image_id: image\.id \}\)/);
  // Strip tiles and the main toolbar both pass the same leading items.
  assert.ok((page.match(/leadingItems=\{editImageItem\(/g) ?? []).length >= 2);
  const api = readSrc("features/images/mask-segment-api.ts");
  assert.match(api, /JSON\.stringify\(\{ \.\.\.body, save: true \}\)/);
});
