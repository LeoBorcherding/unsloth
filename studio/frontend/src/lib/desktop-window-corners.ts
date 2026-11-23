// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Linux shell can show through its window's corners only on a composited screen, and squares
// them while maximized, tiled or fullscreen; the page rounds itself only while it says so.
export function applyWindowCorners(root: HTMLElement, rounded: boolean): void {
  if (rounded) root.dataset.windowCorners = "round";
  else delete root.dataset.windowCorners;
}

/** Older shells have no such command and keep square corners. */
export async function watchWindowCorners(root: HTMLElement): Promise<void> {
  const [{ invoke }, { listen }] = await Promise.all([
    import("@tauri-apps/api/core"),
    import("@tauri-apps/api/event"),
  ]);
  await listen<boolean>("window-corners", (event) => applyWindowCorners(root, event.payload));
  applyWindowCorners(root, await invoke<boolean>("window_corners_rounded"));
}
