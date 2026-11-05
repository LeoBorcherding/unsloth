// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { fitPageWidth, MIN_PAGE_WIDTHS } from "../src/features/browser/page-width.ts";
import { createSettingsSearchIndex, renderedSearchEntries } from "../src/features/settings/settings-search.ts";
import { readSrc } from "./helpers/kit.ts";

test("off keeps the tab's zoom whatever the panel width", () => {
  assert.equal(MIN_PAGE_WIDTHS[0], 0);
  assert.equal(fitPageWidth(1, 400, 0), 1);
  assert.equal(fitPageWidth(1.25, 400, 0), 1.25);
});

test("a narrow panel zooms out until the page lays out at the chosen width", () => {
  // 640 points at 60% is a 1067 px layout, past a 1024 phone/tablet breakpoint.
  assert.equal(fitPageWidth(1, 640, 1024), 0.6);
  assert.ok(640 / fitPageWidth(1, 640, 1024) >= 1024);
  assert.equal(fitPageWidth(1, 900, 1280), 0.7);
});

test("a wide panel and a reader's smaller zoom are left alone", () => {
  assert.equal(fitPageWidth(1, 1400, 1280), 1);
  assert.equal(fitPageWidth(0.5, 640, 1024), 0.5);
});

test("a very narrow panel stops at half size and the page reflows instead", () => {
  assert.equal(fitPageWidth(1, 300, 1280), 0.5);
});

test("the desktop page view sends the fitted zoom, and re-syncs when the setting changes", () => {
  const view = readSrc("features/browser/native-view.ts");
  assert.match(view, /zoom: fitPageWidth\(\s*tab\.zoom,\s*rect\.width \* \(getAppliedInterfaceZoom\(\) \|\| 1\),\s*useBrowserPrefsStore\.getState\(\)\.minPageWidth,/);
  assert.match(view, /useBrowserPrefsStore\.subscribe\(schedule\)/);
});

test("settings search finds the row only where it renders", () => {
  const desktop = createSettingsSearchIndex({ desktop: true, closeToTray: false, menuBarIcon: false });
  const web = createSettingsSearchIndex({ desktop: false, closeToTray: false, menuBarIcon: false });
  assert.ok(desktop.browser.includes("browser.minPageWidthSetting"));
  assert.ok(!web.browser.includes("browser.minPageWidthSetting"));
  // A desktop app without native pages (macOS 13 and earlier) hides the row too.
  const rendered = (native: boolean) =>
    renderedSearchEntries(desktop, "browser", "huggingface", true, false, native);
  assert.ok(rendered(true).includes("browser.minPageWidthSetting"));
  assert.ok(!rendered(false).includes("browser.minPageWidthSetting"));
  assert.ok(rendered(false).includes("browser.downloadLocationSetting"));
  assert.match(readSrc("features/settings/settings-dialog.tsx"), /canUpgradeToFullInstall,\s*nativePages,\s*\)/);
  assert.match(readSrc("features/settings/tabs/browser-tab.tsx"), /\{native \? \(\s*<SettingsRow\s*label=\{t\("browser\.minPageWidthSetting"\)\}/);
});
