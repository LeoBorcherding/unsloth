// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Window as TauriWindow } from "@tauri-apps/api/window";

// Set only while a maximize below has forced resizing on, so the matching restore
// can hand the flag back exactly as it found it. One main window, one value.
let resizableBeforeFill: boolean | null = null;

export async function isFillingWorkArea(appWindow: TauriWindow): Promise<boolean> {
  return appWindow.isMaximized();
}

/** Maximize, or restore, the way dragging to the top edge does.
 *
 * why: the window is configured `resizable: false` because the app draws its own
 * resize handles, and maximizing a non-resizable window is undefined on Windows,
 * where it hides the window instead. Turning resizing on for the duration lets the
 * OS run its normal maximize, which lands flush with the work area. Sizing to the
 * work area by hand does not: Windows reports an outer rect that includes the
 * invisible resize border, so the visible window ends up inset a few pixels.
 *
 * Restore puts back only a flag this forced on. The window otherwise owns it, and
 * clearing it would leave the titlebar's own resize handles inert: they bail out
 * when the window reports itself non-resizable. */
export async function toggleWorkAreaFill(appWindow: TauriWindow): Promise<void> {
  if (await appWindow.isMaximized()) {
    const restoreResizable = resizableBeforeFill;
    resizableBeforeFill = null;
    await appWindow.toggleMaximize();
    if (restoreResizable === false) {
      await appWindow.setResizable(false);
    }
    return;
  }
  resizableBeforeFill = await appWindow.isResizable();
  if (!resizableBeforeFill) {
    await appWindow.setResizable(true);
  }
  await appWindow.toggleMaximize();
}
