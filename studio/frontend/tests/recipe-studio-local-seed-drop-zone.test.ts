// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The CSV/JSON seed field was a bare <input type="file">: no drop target, and
// Unsloth Desktop's native picker only says "Custom files", so the zone has to
// list the types itself.

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const SEED_DIR = new URL(
  "../src/features/recipe-studio/dialogs/seed/",
  import.meta.url,
);
const dialog = readFileSync(new URL("seed-dialog.tsx", SEED_DIR), "utf8");
const zoneUrl = new URL("local-seed-drop-zone.tsx", SEED_DIR);

test("the structured seed file is picked through a drop zone, not a bare file input", () => {
  assert.ok(existsSync(zoneUrl), "local-seed-drop-zone.tsx is missing");
  assert.doesNotMatch(dialog, /type="file"/);
  assert.match(dialog, /<LocalSeedDropZone\b/);
});

test("the drop zone takes native drops and names every type it accepts", () => {
  const zone = readFileSync(zoneUrl, "utf8");
  assert.match(zone, /useNativeFileDrop\(/);
  const accept = zone.match(/LOCAL_SEED_ACCEPT = "([^"]+)"/)?.[1];
  assert.equal(accept, ".csv,.json,.jsonl,.parquet");
  for (const ext of accept.split(",")) {
    const label = ext.slice(1) === "parquet" ? "Parquet" : ext.slice(1).toUpperCase();
    assert.match(zone, new RegExp(`\\b${label}\\b`), `${label} is not listed`);
  }
});

test("in Unsloth Desktop the drop takes only what the attachment reader serves", () => {
  const zone = readFileSync(zoneUrl, "utf8");
  assert.match(zone, /NATIVE_DROP_ACCEPT = "\.csv,\.json,\.jsonl"/);
  assert.match(zone, /accept: isTauri \? NATIVE_DROP_ACCEPT : LOCAL_SEED_ACCEPT/);
  assert.match(zone, /click to browse for Parquet/);
});
