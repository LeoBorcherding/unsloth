// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, readText } from "./helpers/kit.ts";

const APP_WINDOWS = readText("../../src-tauri/src/app_windows.rs");
const MAIN_RS = readText("../../src-tauri/src/main.rs");

let load = 0;
async function detectWith(globals: Record<string, unknown>) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { location: { protocol: "http:", href: "http://127.0.0.1:8888/chat" }, ...globals },
  });
  try {
    load += 1;
    return await import(`../src/lib/api-base.ts?window=${load}`);
  } finally {
    if (previous) Object.defineProperty(globalThis, "window", previous);
    else delete (globalThis as { window?: unknown }).window;
  }
}

test("an extra window runs as the web UI although Tauri injected its globals", async () => {
  const main = await detectWith({ __TAURI_INTERNALS__: {} });
  assert.equal(main.isTauri, true);
  const extra = await detectWith({ __TAURI_INTERNALS__: {}, __UNSLOTH_WEB_WINDOW__: true });
  assert.equal(extra.isTauri, false);
  assert.equal(extra.getApiBase(), "");
  assert.match(APP_WINDOWS, /"__UNSLOTH_WEB_WINDOW__"/);
});

test("the window seeds the keys the web session reads", () => {
  const session = readSrc("features/auth/session.ts");
  for (const [name, key] of [
    ["AUTH_TOKEN_KEY", "unsloth_auth_token"],
    ["AUTH_REFRESH_TOKEN_KEY", "unsloth_auth_refresh_token"],
    ["AUTH_SESSION_MARK_KEY", "unsloth_auth_session_mark"],
  ]) {
    assert.ok(session.includes(`export const ${name} = "${key}";`), name);
    assert.ok(APP_WINDOWS.includes(`"${key}"`), key);
  }
});

test("only the main window hides or quits on close, and the command is registered", () => {
  const events = MAIN_RS.slice(MAIN_RS.indexOf(".on_window_event("));
  const guard = events.indexOf(`if window.label() != "main" {\n                return;`);
  assert.ok(guard > 0 && guard < events.indexOf("CloseRequested"));
  assert.match(MAIN_RS, /app_windows::open_app_window,/);
  assert.match(readSrc("app/open-app-window.ts"), /invoke\("open_app_window"\)/);
});
