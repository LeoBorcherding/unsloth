// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import postcss from "postcss";

import { applyWindowCorners } from "../src/lib/desktop-window-corners.ts";
import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const GATE = 'html.tauri.render-linux[data-window-corners="round"]:not(.webkit-cpu-paint)';

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

test("a round Linux window clips the page, fixed layers and portals included, to 8px corners", () => {
  assert.equal(decls(`${GATE} body`).get("clip-path"), "inset(0 round 8px)");
});

test("body's fill stays inside the clip instead of spreading to the square canvas", () => {
  // A root with no background of its own hands body's to the canvas, which no clip reaches.
  assert.equal(decls(GATE).get("background-image"), "linear-gradient(transparent, transparent)");
});

test("only the round gate clips the body, so maximized, tiled, fullscreen and CPU-painted windows stay square", () => {
  postcss.parse(CSS).walkRules((rule) => {
    rule.walkDecls("clip-path", (d) => {
      for (const selector of rule.selectors) {
        if (/\bbody$/.test(selector)) assert.equal(selector, `${GATE} body`, `${selector} sets ${d.value}`);
      }
    });
  });
});

test("the shell's answer toggles the gate attribute", () => {
  const root = { dataset: {} as DOMStringMap } as HTMLElement;
  applyWindowCorners(root, true);
  assert.equal(root.dataset.windowCorners, "round");
  applyWindowCorners(root, false);
  assert.equal("windowCorners" in root.dataset, false);
});

test("main.tsx asks the shell only in the Linux desktop shell", () => {
  const main = readSrc("main.tsx");
  const block = main.slice(main.indexOf('if (isTauri && document.documentElement.classList.contains("render-linux"))'));
  assert.ok(block.indexOf("watchWindowCorners(document.documentElement)") > 0);
  assert.ok(block.indexOf("watchWindowCorners(") < block.indexOf("\n}\n"));
});
