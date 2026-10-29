// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { INFO_RECIPE_KEYS, unknownRecipeKeys } from "../src/features/images/lib/image-metadata.ts";
import {
  DEFAULT_IP_ADAPTER_SCALE,
  ipAdapterCapable,
  ipAdapterRecipeLabel,
  ipAdapterRequest,
  ipAdapterScaleFromRecipe,
  withIpAdapterInput,
} from "../src/features/images/lib/ip-adapter.ts";

const ADAPTER = "h94/IP-Adapter:sdxl_vit-h";

test("the image prompt slot shows only for a loaded model that takes one, never for a Connection", () => {
  assert.equal(ipAdapterCapable({ loaded: true, supports_ip_adapter: true }, false), true);
  assert.equal(ipAdapterCapable({ loaded: true, supports_ip_adapter: true }, true), false);
  assert.equal(ipAdapterCapable({ loaded: true, supports_ip_adapter: false }, false), false);
  assert.equal(ipAdapterCapable({ loaded: true }, false), false);
  assert.equal(ipAdapterCapable({ loaded: false, supports_ip_adapter: true }, false), false);
  assert.equal(ipAdapterCapable(null, false), false);
});

test("the request carries an image prompt only on Create with an image and a non-zero strength", () => {
  const image = "data:image/png;base64,AAAA";
  assert.deepEqual(ipAdapterRequest(true, "create", image, 0.6), { image, scale: 0.6 });
  assert.equal(ipAdapterRequest(true, "create", image, 0), undefined);
  assert.equal(ipAdapterRequest(true, "create", null, 0.6), undefined);
  assert.equal(ipAdapterRequest(true, "transform", image, 0.6), undefined);
  assert.equal(ipAdapterRequest(false, "create", image, 0.6), undefined);
  assert.deepEqual(ipAdapterRequest(true, "create", image, 1.4), { image, scale: 1 });
  assert.equal(DEFAULT_IP_ADAPTER_SCALE, 0.6);
});

test("restore brings back the recorded strength and asks for the image again", () => {
  const record = { ip_adapter: ADAPTER, ip_adapter_scale: 0.6 };
  assert.equal(ipAdapterScaleFromRecipe(record), 0.6);
  assert.equal(ipAdapterScaleFromRecipe({}), null);
  assert.equal(ipAdapterScaleFromRecipe({ ip_adapter: ADAPTER, ip_adapter_scale: 7 }), null);
  assert.equal(ipAdapterScaleFromRecipe({ ip_adapter: null, ip_adapter_scale: 0.6 }), null);
  assert.equal(withIpAdapterInput(undefined, record), "the image prompt");
  assert.equal(withIpAdapterInput("the control image", record), "the control image and the image prompt");
  assert.equal(withIpAdapterInput(undefined, {}), null);
  assert.equal(withIpAdapterInput("the source image", {}), "the source image");
  assert.equal(ipAdapterRecipeLabel(record), `${ADAPTER} at 0.6`);
  assert.equal(ipAdapterRecipeLabel({}), null);
});

test("an imported recipe with an image prompt is known, not flagged as from a newer build", () => {
  assert.ok(INFO_RECIPE_KEYS.includes("ip_adapter"));
  assert.ok(INFO_RECIPE_KEYS.includes("ip_adapter_scale"));
  assert.deepEqual(unknownRecipeKeys({ prompt: "p", ip_adapter: ADAPTER, ip_adapter_scale: 0.6 }), []);
});

test("the Images page sends, restores and shows the image prompt", () => {
  const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  assert.match(page, /ip_adapter: ipAdapterRequest\(ipCapable, workflow, ipImage, ipScale\)/);
  assert.match(page, /\{ipCapable && workflow === "create" && \(\s*<Field\s+label="Image prompt"/);
  assert.match(page, /setIpScale\(ipAdapterScaleFromRecipe\(image\) \?\? DEFAULT_IP_ADAPTER_SCALE\)/);
  assert.match(page, /<RecipeRow label="Image prompt"/);
});
