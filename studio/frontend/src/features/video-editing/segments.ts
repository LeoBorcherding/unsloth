// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CONTINUATION_KEPT_FRAMES } from "../video/video-continuation.ts";

export type SegmentKind = "real" | "restyle" | "continue" | "bridge";

// What renders a segment. Only Cosmos is wired today; another family plugs in as a new engine here.
export type SegmentEngine = "upload" | "cosmos3-sim-transfer" | "cosmos3-nano";

export interface SegmentKindInfo {
  value: SegmentKind;
  label: string;
  hint: string;
  engine: SegmentEngine;
  // "upload": the segment brings its own clip. "previous": it is generated from the segment before it.
  source: "upload" | "previous";
  // It also lands on the start of the segment after it.
  needsNext: boolean;
  // Frames it shares with the segment before and after it, dropped when the movie is joined.
  shared: { head: number; tail: number };
  // Whether it takes a prompt, and what the prompt describes.
  prompt: { label: string; placeholder: string } | null;
}

// Frames a bridge keeps from the start of the clip after it (two latent frames); mirrors COSMOS3_BRIDGE_FRAMES.
export const BRIDGE_TARGET_FRAMES = 8;
export const EDIT_FPS = 24;
export const EDIT_WIDTH = 832;
export const EDIT_HEIGHT = 480;
export const COSMOS3_NANO_REPO = "nvidia/Cosmos3-Nano";

export const SEGMENT_KINDS: ReadonlyArray<SegmentKindInfo> = [
  {
    value: "real",
    label: "Real clip",
    hint: "A clip you upload, trimmed to its in and out points.",
    engine: "upload",
    source: "upload",
    needsNext: false,
    shared: { head: 0, tail: 0 },
    prompt: null,
  },
  {
    value: "restyle",
    label: "Restyle",
    hint: "A clip you upload, repainted from a prompt by Cosmos3-Nano-Sim-Transfer. Uses its first 4 seconds.",
    engine: "cosmos3-sim-transfer",
    source: "upload",
    needsNext: false,
    shared: { head: 0, tail: 0 },
    prompt: { label: "Look", placeholder: "A red rally car at golden hour" },
  },
  {
    value: "continue",
    label: "Continue",
    hint: "New footage from Cosmos3-Nano that carries on from the end of the segment before.",
    engine: "cosmos3-nano",
    source: "previous",
    needsNext: false,
    shared: { head: CONTINUATION_KEPT_FRAMES, tail: 0 },
    prompt: { label: "What happens", placeholder: "The car keeps going down the forest road" },
  },
  {
    value: "bridge",
    label: "Bridge",
    hint: "New footage from Cosmos3-Nano that starts where the segment before ends and lands on the start of the clip after it.",
    engine: "cosmos3-nano",
    source: "previous",
    needsNext: true,
    shared: { head: CONTINUATION_KEPT_FRAMES, tail: BRIDGE_TARGET_FRAMES },
    prompt: {
      label: "What happens",
      placeholder: "The car slides through the corner and drives off toward the barrier",
    },
  },
];

export function kindInfo(kind: SegmentKind): SegmentKindInfo {
  return SEGMENT_KINDS.find((k) => k.value === kind) as SegmentKindInfo;
}

// Generated lengths: total frames sent to Cosmos3-Nano (4k + 1 at 24 fps).
export const SEGMENT_LENGTHS: ReadonlyArray<number> = [61, 93, 121];
export const DEFAULT_SEGMENT_LENGTH = 93;

export interface SegmentShape {
  kind: SegmentKind;
  hasFile: boolean;
  prompt: string;
}

export function hasSource(kind: SegmentKind): boolean {
  return kindInfo(kind).source === "upload";
}

/** Seconds of new footage a generated segment of ``frames`` adds once its shared frames are dropped. */
export function newSeconds(kind: SegmentKind, frames: number): number {
  const { head, tail } = kindInfo(kind).shared;
  return Math.max(0, frames - head - tail) / EDIT_FPS;
}

/** Why the segment at ``index`` cannot render, or null. */
export function segmentProblem(segments: ReadonlyArray<SegmentShape>, index: number): string | null {
  const seg = segments[index];
  const info = kindInfo(seg.kind);
  if (info.source === "upload") {
    if (!seg.hasFile) return "Choose a clip";
  } else if (index === 0) {
    return "Needs a segment before it to carry on from";
  }
  if (info.prompt && !seg.prompt.trim()) {
    return seg.kind === "restyle" ? "Describe the new look" : "Describe what happens";
  }
  if (info.needsNext) {
    const next = segments[index + 1];
    if (!next || !hasSource(next.kind)) return "Needs a real or restyled clip right after it to land on";
  }
  return null;
}

/** Render order: uploaded clips first (a bridge needs the clip after it), then generated segments front to back
 *  (each needs the one before it). */
export function renderOrder(kinds: ReadonlyArray<SegmentKind>): number[] {
  const order = kinds.map((_, i) => i);
  return [...order.filter((i) => hasSource(kinds[i])), ...order.filter((i) => !hasSource(kinds[i]))];
}

/** The join request: each rendered segment once, minus the frames it shares with its neighbours. */
export function composeSpec(kinds: ReadonlyArray<SegmentKind>): {
  segments: Array<{ clip: number; skip_head: number; skip_tail: number }>;
  width: number;
  height: number;
  fps: number;
} {
  return {
    segments: kinds.map((kind, clip) => {
      const { head, tail } = kindInfo(kind).shared;
      return { clip, skip_head: head, skip_tail: tail };
    }),
    width: EDIT_WIDTH,
    height: EDIT_HEIGHT,
    fps: EDIT_FPS,
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
    width: EDIT_WIDTH,
    height: EDIT_HEIGHT,
    fps: EDIT_FPS,
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

/** Which engines a list of segments needs, so the page can say what to set up before rendering. */
export function enginesNeeded(kinds: ReadonlyArray<SegmentKind>): Set<SegmentEngine> {
  return new Set(kinds.map((k) => kindInfo(k).engine));
}
