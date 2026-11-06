// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CONTINUATION_KEPT_FRAMES } from "./video-continuation.ts";

export type TimelineKind = "real" | "restyle" | "continue" | "bridge";

// Frames a bridge keeps from the start of the clip after it (two latent frames); mirrors COSMOS3_BRIDGE_FRAMES.
export const BRIDGE_TARGET_FRAMES = 8;
export const TIMELINE_FPS = 24;
export const TIMELINE_WIDTH = 832;
export const TIMELINE_HEIGHT = 480;

export const TIMELINE_KINDS: ReadonlyArray<{ value: TimelineKind; label: string; hint: string }> = [
  { value: "real", label: "Real clip", hint: "A clip you upload, trimmed to its in and out points" },
  {
    value: "restyle",
    label: "Restyle",
    hint: "A clip you upload, repainted from a prompt by Cosmos3-Nano-Sim-Transfer (its first 4 seconds)",
  },
  { value: "continue", label: "Continue", hint: "New footage that carries on from the end of the part before" },
  {
    value: "bridge",
    label: "Bridge",
    hint: "New footage that starts where the part before ends and lands on the start of the part after",
  },
];

// Generated lengths: total frames sent to Cosmos3-Nano (4k + 1 at 24 fps).
export const TIMELINE_LENGTHS: ReadonlyArray<number> = [61, 93, 121];
export const DEFAULT_TIMELINE_LENGTH = 93;

export interface TimelineSegmentShape {
  kind: TimelineKind;
  hasFile: boolean;
  prompt: string;
}

export function hasSource(kind: TimelineKind): boolean {
  return kind === "real" || kind === "restyle";
}

/** Frames a generated part shares with its neighbours, dropped when the timeline is joined. */
export function sharedFrames(kind: TimelineKind): { head: number; tail: number } {
  if (kind === "continue") return { head: CONTINUATION_KEPT_FRAMES, tail: 0 };
  if (kind === "bridge") return { head: CONTINUATION_KEPT_FRAMES, tail: BRIDGE_TARGET_FRAMES };
  return { head: 0, tail: 0 };
}

/** Seconds of new footage a generated part of ``frames`` adds once its shared frames are dropped. */
export function newSeconds(kind: TimelineKind, frames: number): number {
  const { head, tail } = sharedFrames(kind);
  return Math.max(0, frames - head - tail) / TIMELINE_FPS;
}

/** Why the part at ``index`` cannot render, or null. */
export function segmentProblem(
  segments: ReadonlyArray<TimelineSegmentShape>,
  index: number,
): string | null {
  const seg = segments[index];
  if (hasSource(seg.kind)) {
    if (!seg.hasFile) return "Choose a clip";
    if (seg.kind === "restyle" && !seg.prompt.trim()) return "Write what the restyle should look like";
    return null;
  }
  if (index === 0) return "Needs a part before it to carry on from";
  if (!seg.prompt.trim()) return "Write what happens in this part";
  if (seg.kind === "bridge") {
    const next = segments[index + 1];
    if (!next || !hasSource(next.kind)) return "Needs a real or restyled clip right after it to land on";
  }
  return null;
}

/** The order parts render in: every uploaded clip first (a bridge needs the part after it), then the generated
 *  parts front to back (a continuation needs the part before it). */
export function renderOrder(kinds: ReadonlyArray<TimelineKind>): number[] {
  const order = kinds.map((_, i) => i);
  return [...order.filter((i) => hasSource(kinds[i])), ...order.filter((i) => !hasSource(kinds[i]))];
}

/** The join request: each rendered part once, with the frames it shares with its neighbours dropped. */
export function composeSpec(kinds: ReadonlyArray<TimelineKind>): {
  segments: Array<{ clip: number; skip_head: number; skip_tail: number }>;
  width: number;
  height: number;
  fps: number;
} {
  return {
    segments: kinds.map((kind, clip) => {
      const { head, tail } = sharedFrames(kind);
      return { clip, skip_head: head, skip_tail: tail };
    }),
    width: TIMELINE_WIDTH,
    height: TIMELINE_HEIGHT,
    fps: TIMELINE_FPS,
  };
}

/** The trim request for one uploaded clip. */
export function trimSpec(inSeconds: number, outSeconds: number | null): {
  segments: Array<{ clip: number; start: number; end: number | null }>;
  width: number;
  height: number;
  fps: number;
} {
  return {
    segments: [{ clip: 0, start: Math.max(0, inSeconds), end: outSeconds }],
    width: TIMELINE_WIDTH,
    height: TIMELINE_HEIGHT,
    fps: TIMELINE_FPS,
  };
}

/** Why an in / out pair is unusable, or null. ``duration`` is the clip's length when known. */
export function trimProblem(inSeconds: number, outSeconds: number | null, duration?: number): string | null {
  if (!Number.isFinite(inSeconds) || inSeconds < 0) return "The in point must be 0 s or later";
  if (outSeconds !== null && (!Number.isFinite(outSeconds) || outSeconds <= inSeconds)) {
    return "The out point must come after the in point";
  }
  if (duration !== undefined && inSeconds >= duration) return `The clip is only ${duration.toFixed(1)} s long`;
  return null;
}
