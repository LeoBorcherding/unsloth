// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { holdGlassMotion } from "./glass-motion.ts";

/** The Run settings corner rounds over the same 200 ms, so the strip and the corner arrive together. */
export const GLASS_FRAME_ROLL_MS = 200;

export type GlassFrameRoll = "in" | "out" | null;

/** Next roll for a frame change: only a toggle rolls, so mount, reload and route changes never animate. */
export function nextGlassFrameRoll(was: boolean, now: boolean, roll: GlassFrameRoll): GlassFrameRoll {
  if (was === now) return roll;
  return now ? "in" : "out";
}

/**
 * The pinned sidebar's glass frame (data-windows-glass-frame) and the way it is rolling (data-glass-frame-roll).
 * Unpinning keeps the frame for one roll so its glass can leave instead of vanishing.
 */
export function useGlassFrameRoll(frame: boolean): { frame: boolean; roll: GlassFrameRoll } {
  const [state, setState] = useState<{ frame: boolean; roll: GlassFrameRoll }>({ frame, roll: null });
  // Derived in render, not an effect: the frame must not drop for a single commit on the way out.
  if (state.frame !== frame) setState({ frame, roll: nextGlassFrameRoll(state.frame, frame, state.roll) });
  useEffect(() => {
    const { roll } = state;
    if (!roll) return;
    // The click's own hold can end mid-roll; software compositing would then blur every remaining frame.
    holdGlassMotion(document.querySelector("[data-windows-glass-frame]"), GLASS_FRAME_ROLL_MS);
    const timer = setTimeout(() => setState((s) => (s.roll === roll ? { ...s, roll: null } : s)), GLASS_FRAME_ROLL_MS);
    return () => clearTimeout(timer);
  }, [state]);
  return { frame: frame || state.roll === "out", roll: state.roll };
}
