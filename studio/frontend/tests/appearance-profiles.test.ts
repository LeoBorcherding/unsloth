// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import type { AppearanceSnapshot as Snapshot } from "../src/features/settings/appearance-theme/theme-file.ts";

const KEY = "unsloth_appearance_profiles";
const stored = new Map<string, string>();
let quotaFull = false;
(globalThis as { document?: unknown }).document = {
  documentElement: {
    style: { setProperty() {}, removeProperty() {} },
    setAttribute() {},
    removeAttribute() {},
    toggleAttribute() {},
    classList: { toggle() {} },
  },
};
(globalThis as { window?: unknown }).window = {
  addEventListener() {},
  localStorage: {
    getItem: (k: string) => stored.get(k) ?? null,
    setItem: (k: string, v: string) => {
      if (quotaFull) throw new DOMException("full", "QuotaExceededError");
      stored.set(k, v);
    },
    removeItem: (k: string) => stored.delete(k),
  },
};
const { sanitizeCustomization } =
  await import("../src/features/settings/stores/appearance-custom-store.ts");
const { sanitizeAdjustments } =
  await import("../src/features/chat-background/model.ts");
const { parseThemeFile, planThemeApply, THEME_FILE_KIND } =
  await import("../src/features/settings/appearance-theme/theme-file.ts");
const {
  MAX_PROFILES,
  captureProfile,
  parseProfile,
  profileExportFile,
  profileName,
  readProfiles,
  useAppearanceProfilesStore: store,
} = await import("../src/features/settings/appearance-theme/profiles.ts");

const font = (name: string) => ({
  name,
  dataUrl: "data:font/woff2;base64,QUJD",
});
const snap = (look: Record<string, unknown>, palette = "standard"): Snapshot =>
  ({
    palette,
    customization: sanitizeCustomization(look),
    glass: {
      material: "frosted",
      style: "frosted",
      strength: { light: 40, dark: 15 },
      lens: "off",
      menus: true,
      scope: "all",
    },
    background: { adjustments: sanitizeAdjustments({}), blob: new Blob(["x"]) },
  }) as Snapshot;
const LOOK = snap(
  {
    contrast: 70,
    uiFont: "Brand",
    codeFont: "Georgia",
    chatWidth: "wide",
    sentAttachments: "chips",
    importedFonts: [font("Brand"), font("Unused")],
    reduceMotion: "off",
    pointerCursors: true,
  },
  "matcha",
);
const saveLook = (name: string, includeLayout = false) =>
  store.getState().save(captureProfile(LOOK, { name, includeLayout }));
const rehydrate = () => store.persist.rehydrate();

beforeEach(() => {
  quotaFull = false;
  stored.clear();
  store.setState({ profiles: [] });
});

test("a saved profile is a theme file without assets or personal fields", () => {
  const file = captureProfile(LOOK, { name: "Dusk", includeLayout: false });
  assert.equal(file.kind, THEME_FILE_KIND);
  const text = JSON.stringify(file);
  for (const key of ["fonts", "background", "layout"])
    assert.equal(key in file, false, key);
  for (const key of ["importedFonts", "reduceMotion", "pointerCursors"])
    assert.equal(text.includes(`"${key}"`), false, key);
  const look = file.look as Record<string, unknown>;
  // Imported fonts stay on this device, so the selection is kept by name.
  assert.deepEqual([look.uiFont, look.codeFont], ["Brand", "Georgia"]);
  assert.equal(parseThemeFile(text).ok, true);
  assert.equal(
    "layout" in captureProfile(LOOK, { name: "x", includeLayout: true }),
    true,
  );
});

test("save lists the profile and persists it across a reload", async () => {
  assert.equal(saveLook("Dusk"), "saved");
  assert.equal(saveLook("Dawn", true), "saved");
  assert.deepEqual(store.getState().profiles.map(profileName), [
    "Dusk",
    "Dawn",
  ]);
  const saved = stored.get(KEY);
  assert.ok(saved);
  store.setState({ profiles: [] });
  stored.set(KEY, saved);
  await rehydrate();
  assert.deepEqual(store.getState().profiles.map(profileName), [
    "Dusk",
    "Dawn",
  ]);
});

test("applying a profile restores the look and leaves personal settings alone", () => {
  saveLook("Dusk", true);
  const theme = parseProfile(store.getState().profiles[0]);
  assert.ok(theme);
  const current = snap({ reduceMotion: "on", pointerCursors: false });
  const plan = planThemeApply(current, theme, {
    applyLayout: theme.hasLayout,
    applyAssets: true,
  });
  assert.equal(plan.palette, "matcha");
  assert.equal(plan.customization.contrast, 70);
  assert.equal(plan.customization.chatWidth, "wide");
  assert.equal(plan.customization.sentAttachments, "chips");
  assert.equal(plan.customization.codeFont, "Georgia");
  assert.equal(plan.customization.reduceMotion, "on");
  assert.equal(plan.customization.pointerCursors, false);
  assert.equal(plan.background, "keep");
  // A layout left out of the profile is not applied.
  saveLook("Bare");
  const bare = parseProfile(store.getState().profiles[1]);
  assert.equal(bare?.hasLayout, false);
});

test("rename cleans the name and refuses an empty one", () => {
  saveLook("Dusk");
  const { id } = store.getState().profiles[0];
  assert.equal(store.getState().rename(id, "  Night\u202E Owl  "), true);
  assert.equal(profileName(store.getState().profiles[0]), "Night  Owl");
  assert.equal(store.getState().rename(id, " \u200B "), false);
  assert.equal(profileName(store.getState().profiles[0]), "Night  Owl");
});

test("delete removes one profile and restore puts it back in place", () => {
  saveLook("A");
  saveLook("B");
  saveLook("C");
  const b = store.getState().profiles[1];
  store.getState().remove(b.id);
  assert.deepEqual(store.getState().profiles.map(profileName), ["A", "C"]);
  assert.equal(store.getState().restore(b, 1), "saved");
  assert.deepEqual(store.getState().profiles.map(profileName), ["A", "B", "C"]);
});

test("undoing a delete never evicts a profile saved since", () => {
  for (let i = 0; i < MAX_PROFILES; i++) saveLook(`P${i}`);
  const gone = store.getState().profiles[0];
  store.getState().remove(gone.id);
  saveLook("New");
  assert.equal(store.getState().restore(gone, 0), "full");
  const names = store.getState().profiles.map(profileName);
  assert.equal(names.length, MAX_PROFILES);
  assert.ok(names.includes("New"));
  assert.ok(!names.includes("P0"));
});

test("export embeds only the imported fonts the profile still selects", () => {
  saveLook("Dusk");
  const theme = parseProfile(store.getState().profiles[0]);
  assert.ok(theme);
  const file = profileExportFile(theme, LOOK.customization.importedFonts);
  const parsed = parseThemeFile(JSON.stringify(file));
  assert.ok(parsed.ok);
  assert.equal(parsed.theme.name, "Dusk");
  assert.deepEqual(
    parsed.theme.fonts?.map((f) => f.name),
    ["Brand"],
  );
  assert.equal(parsed.theme.customization.uiFont, "Brand");
  // Once the font is gone from this device, the file carries no fonts.
  const noFonts = profileExportFile(theme, []);
  assert.equal("fonts" in noFonts, false);
});

test("corrupt stored entries are dropped without a crash", async () => {
  saveLook("Good");
  const good = store.getState().profiles[0];
  stored.set(
    KEY,
    JSON.stringify({
      state: {
        profiles: [
          good,
          { ...good }, // duplicate id
          null,
          "nope",
          { id: 7, theme: good.theme },
          { id: "no-theme" },
          { id: "wrong-kind", theme: { ...good.theme, kind: "other" } },
          { id: "bad-version", theme: { ...good.theme, schemaVersion: 0 } },
          { id: "array", theme: [] },
          {
            id: "assets",
            theme: { ...good.theme, background: { dataUrl: "x" }, fonts: [] },
          },
        ],
      },
      version: 1,
    }),
  );
  await rehydrate();
  const ids = store.getState().profiles.map((p) => p.id);
  assert.deepEqual(ids, [good.id, "assets"]);
  const assets = store.getState().profiles[1].theme;
  assert.equal("background" in assets || "fonts" in assets, false);

  // A fresh load starts empty; a payload that can't be parsed leaves it that way.
  for (const junk of ["{not json", '{"state":{"profiles":"x"},"version":1}']) {
    store.setState({ profiles: [] });
    stored.set(KEY, junk);
    await rehydrate();
    assert.deepEqual(store.getState().profiles, []);
  }
  assert.deepEqual(readProfiles(undefined), []);
});

test("a save that storage refuses is rolled back and reported", () => {
  saveLook("Kept");
  quotaFull = true;
  assert.equal(saveLook("Lost"), "storage");
  assert.deepEqual(store.getState().profiles.map(profileName), ["Kept"]);
  const { id } = store.getState().profiles[0];
  assert.equal(store.getState().rename(id, "Renamed"), false);
  assert.equal(profileName(store.getState().profiles[0]), "Kept");
  assert.equal(store.getState().remove(id), false);
  assert.deepEqual(store.getState().profiles.map(profileName), ["Kept"]);
});

test("the list is capped", () => {
  for (let i = 0; i < MAX_PROFILES; i++)
    assert.equal(saveLook(`P${i}`), "saved");
  assert.equal(saveLook("One more"), "full");
  assert.equal(store.getState().profiles.length, MAX_PROFILES);
});
