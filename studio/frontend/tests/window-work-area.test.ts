// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { toggleWorkAreaFill } from "../src/components/tauri/window-work-area.ts";

type AppWindow = Parameters<typeof toggleWorkAreaFill>[0];

function fakeWindow(options: { resizable: boolean; maximized?: boolean }) {
  const state = {
    resizable: options.resizable,
    maximized: options.maximized ?? false,
  };
  const appWindow = {
    isMaximized: async () => state.maximized,
    isResizable: async () => state.resizable,
    setResizable: async (next: boolean) => {
      state.resizable = next;
    },
    // Windows only runs its normal maximize on a resizable window.
    toggleMaximize: async () => {
      if (!state.maximized && !state.resizable) {
        return;
      }
      state.maximized = !state.maximized;
    },
  } as unknown as AppWindow;
  return { state, appWindow };
}

test("a fixed-size window is made resizable so the maximize lands", async () => {
  const { state, appWindow } = fakeWindow({ resizable: false });
  await toggleWorkAreaFill(appWindow);
  assert.equal(state.maximized, true);
});

test("restoring leaves the resize handles working", async () => {
  const { state, appWindow } = fakeWindow({ resizable: true });
  await toggleWorkAreaFill(appWindow);
  await toggleWorkAreaFill(appWindow);
  assert.equal(state.maximized, false);
  // The handles in WindowTitlebar skip startResizeDragging when this is off.
  assert.equal(state.resizable, true);
});

test("a window that was fixed-size goes back to fixed-size", async () => {
  const { state, appWindow } = fakeWindow({ resizable: false });
  await toggleWorkAreaFill(appWindow);
  await toggleWorkAreaFill(appWindow);
  assert.equal(state.maximized, false);
  assert.equal(state.resizable, false);
});

test("a restore of someone else's maximize leaves the flag alone", async () => {
  const { state, appWindow } = fakeWindow({ resizable: true, maximized: true });
  await toggleWorkAreaFill(appWindow);
  assert.equal(state.maximized, false);
  assert.equal(state.resizable, true);
});
