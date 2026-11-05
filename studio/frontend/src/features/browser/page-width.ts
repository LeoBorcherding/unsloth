// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Narrowest layout a desktop page gets in a narrow panel, in CSS pixels; 0 lays it out at the panel's width. */
export const MIN_PAGE_WIDTHS = [0, 1024, 1280];

// Below this the text is too small to read, so a narrower panel lets the page reflow.
const FLOOR = 0.5;

/** Zoom that lays the page out at least `minWidth` wide in `width` view points, never above the tab's own.
 *  Snapped down to 5%, so dragging the panel does not re-zoom the page every frame. */
export function fitPageWidth(zoom: number, width: number, minWidth: number): number {
  if (minWidth <= 0 || width <= 0) return zoom;
  const fit = Math.max(FLOOR, Math.floor((width / minWidth) * 20) / 20);
  return Math.min(zoom, fit);
}
