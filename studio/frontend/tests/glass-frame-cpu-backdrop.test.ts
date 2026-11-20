// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import postcss from "postcss";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const GATE = "html.tauri.render-linux.webkit-cpu-paint[data-glass]";
const SIDEBAR = `${GATE}[data-chat-wallpaper-workspace-ready] [data-windows-glass-frame] > [data-slot="sidebar-inner"]`;
const PANEL = `${GATE}[data-sidebar-glass-frame][data-chat-wallpaper-workspace-ready] aside[data-slot="chat-settings-panel"]`;
const FRAME_BEFORE =
  'html[data-glass][data-chat-wallpaper-workspace-ready] [data-windows-glass-frame]::before';

function decls(selector: string): Map<string, string> {
  const out = new Map<string, string>();
  postcss.parse(CSS).walkRules((rule) => {
    if (!rule.selectors.includes(selector)) return;
    rule.walkDecls((d) => {
      out.set(d.prop, d.value);
    });
  });
  return out;
}

test("under WebKitGTK CPU painting the sidebar and docked panel blur for themselves", () => {
  // Content painted over the strip's ::before glass is blurred into it there (logo and slider thumb halos).
  const frame = decls(FRAME_BEFORE);
  for (const selector of [SIDEBAR, PANEL]) {
    const own = decls(selector);
    assert.equal(own.get("backdrop-filter"), frame.get("backdrop-filter"), selector);
    assert.equal(own.get("-webkit-backdrop-filter"), frame.get("-webkit-backdrop-filter"), selector);
    assert.equal(own.get("background-color"), frame.get("background-color"), `${selector} keeps the strip's fill`);
  }
});

test("the self blur comes after the rules that clear it and stays off every other engine", () => {
  const at = CSS.indexOf(`${GATE}[data-chat-wallpaper-workspace-ready] [data-windows-glass-frame]`);
  const clearSidebar = CSS.indexOf('[data-windows-glass-frame] > [data-slot="sidebar-inner"] {\n  position: relative;');
  const clearPanel = CSS.indexOf("panel are one blurred sheet");
  assert.ok(at > clearSidebar && at > clearPanel, "equal or lower specificity would lose to the clears otherwise");
  postcss.parse(CSS).walkRules((rule) => {
    for (const selector of rule.selectors) {
      if (!/\[data-windows-glass-frame\] > \[data-slot="sidebar-inner"\]$|\[data-sidebar-glass-frame\][^,]*aside\[data-slot="chat-settings-panel"\]$/.test(selector)) continue;
      let blur = false;
      rule.walkDecls(/backdrop-filter$/, (d) => {
        if (d.value !== "none") blur = true;
      });
      if (blur) assert.ok(selector.startsWith(GATE), `${selector} blurs outside the CPU-paint gate`);
    }
  });
});
