// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { flushSync } from "react-dom";
import { holdGlassMotion } from "./glass-motion.ts";

/** The strip's slide, front and Run settings corner all run on this one clock. */
export const GLASS_FRAME_ROLL_MS = 200;

/** "peek": pinned while held out, so the column is already on screen and only the band rolls. */
export type GlassFrameRoll = "in" | "peek" | "out" | null;

const FRAME = "[data-windows-glass-frame]";
const HELD_OUT = '[data-slot="sidebar-container"][data-held-out]';

/** Next roll for a frame change: only a toggle rolls, so mount, reload and route changes never animate. */
export function nextGlassFrameRoll(was: boolean, now: boolean, roll: GlassFrameRoll, heldOut = false): GlassFrameRoll {
  if (was === now) return roll;
  if (!now) return "out";
  return heldOut ? "peek" : "in";
}

/**
 * The pinned sidebar's glass frame (data-windows-glass-frame) and the way it is rolling (data-glass-frame-roll).
 * Unpinning keeps the frame for one roll so its glass can leave instead of vanishing.
 */
export function useGlassFrameRoll(frame: boolean): { frame: boolean; roll: GlassFrameRoll } {
  const [state, setState] = useState<{ frame: boolean; roll: GlassFrameRoll }>({ frame, roll: null });
  // Derived in render, not an effect: the frame must not drop for a single commit on the way out. The DOM still
  // shows the previous commit here, so it says whether the sidebar was held out when the pin landed.
  if (state.frame !== frame) {
    const heldOut = frame && typeof document !== "undefined" && document.querySelector(HELD_OUT) !== null;
    setState({ frame, roll: nextGlassFrameRoll(state.frame, frame, state.roll, heldOut) });
  }
  useEffect(() => {
    const { roll } = state;
    if (!roll) return;
    // The click's own hold can end mid-roll; software compositing would then blur every remaining frame.
    holdGlassMotion(document.querySelector(FRAME), GLASS_FRAME_ROLL_MS);
    // Synchronous, with its effects: the frame and the <html> flag must drop in the same frame, or the panel
    // paints once with neither the strip's glass nor its own.
    const end = () => flushSync(() => setState((s) => (s.roll === roll ? { ...s, roll: null } : s)));
    const timer = setTimeout(end, GLASS_FRAME_ROLL_MS);
    return () => clearTimeout(timer);
  }, [state]);
  return { frame: frame || state.roll === "out", roll: state.roll };
}
