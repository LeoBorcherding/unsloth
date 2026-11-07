// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  cancelVideoGeneration,
  cancelVideoRestyle,
  composeVideoTimeline,
  fetchGalleryVideoBlob,
  fetchVideoRestyleClip,
  fetchVideoRestyleProgress,
  generateVideo,
  getVideoGenerateProgress,
  startVideoRestyle,
} from "../video/api";
import {
  composeSpec,
  DEFAULT_SEGMENT_LENGTH,
  EDIT_FPS,
  EDIT_HEIGHT,
  EDIT_WIDTH,
  hasSource,
  kindInfo,
  renderOrder,
  type SegmentKind,
  trimSpec,
} from "./segments";

const POLL_MS = 1000;

export type SegmentState = "idle" | "waiting" | "running" | "done" | "error";

export interface Segment {
  id: number;
  kind: SegmentKind;
  file: File | null;
  duration?: number;
  inS: string;
  outS: string;
  prompt: string;
  frames: number;
  seed: string;
  state: SegmentState;
  note: string | null;
  output: Blob | null;
}

let nextId = 1;
export function newSegment(kind: SegmentKind): Segment {
  return {
    id: nextId++,
    kind,
    file: null,
    inS: "0",
    outS: "",
    prompt: "",
    frames: DEFAULT_SEGMENT_LENGTH,
    seed: "",
    state: "idle",
    note: null,
    output: null,
  };
}

export function parseSeconds(value: string): number | null {
  if (value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : Number.NaN;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the clip."));
    reader.readAsDataURL(blob);
  });
}

class Cancelled extends Error {}

// A generated segment renders from its neighbours, so any change to the list makes every generated render stale (an
// uploaded clip's render stays good unless that segment itself changed), and with them the movie.
function restage(list: Segment[], changed?: number): Segment[] {
  return list.map((s) =>
    s.id === changed || !hasSource(s.kind) ? { ...s, state: "idle", note: null, output: null } : s,
  );
}

interface EditingState {
  segments: Segment[];
  running: boolean;
  cancelling: boolean;
  error: string | null;
  movie: Blob | null;
  movieKinds: SegmentKind[];
  // Frames each segment contributed to the movie.
  movieFrames: number[];
  stage: { index: number; step: number; total: number } | null;
  // Changes that do not invalidate a render (a probed clip length).
  patch: (id: number, patch: Partial<Segment>) => void;
  edit: (id: number, patch: Partial<Segment>) => void;
  move: (index: number, by: number) => void;
  remove: (id: number) => void;
  add: (kind: SegmentKind) => void;
  setError: (error: string | null) => void;
  render: () => Promise<void>;
  cancel: () => void;
}

// Kept in a store, not the page: a render keeps going, and the list survives, while you visit another page.
let cancelActive: null | (() => Promise<unknown>) = null;
let cancelled = false;

export const useVideoEditingStore = create<EditingState>((set, get) => {
  const update = (id: number, patch: Partial<Segment>) =>
    set((s) => ({ segments: s.segments.map((seg) => (seg.id === id ? { ...seg, ...patch } : seg)) }));
  const checkCancel = () => {
    if (cancelled) throw new Cancelled();
  };
  // A Cancel clicked while the start request was in flight found nothing to stop; stop the job it started now.
  const startedJob = async (stop: () => Promise<unknown>) => {
    cancelActive = stop;
    if (cancelled) {
      await stop().catch(() => undefined);
      throw new Cancelled();
    }
  };

  const renderUpload = async (seg: Segment): Promise<Blob> => {
    const trimmed = (
      await composeVideoTimeline([seg.file as File], trimSpec(Number(seg.inS || 0), parseSeconds(seg.outS)))
    ).video;
    if (seg.kind === "real") return trimmed;
    const seed = seg.seed.trim() === "" ? null : Number.parseInt(seg.seed, 10);
    const started = await startVideoRestyle({
      prompt: seg.prompt,
      seed,
      video: new File([trimmed], "segment.mp4", { type: "video/mp4" }),
    });
    await startedJob(cancelVideoRestyle);
    for (;;) {
      await sleep(POLL_MS);
      const p = await fetchVideoRestyleProgress();
      if (p.id && started.id && p.id !== started.id) throw new Error("Another restyle took over the GPU.");
      if (p.active && p.total) set((s) => ({ stage: s.stage && { ...s.stage, step: p.step ?? 0, total: p.total ?? 0 } }));
      if (p.phase === "done" && p.id) return fetchVideoRestyleClip(p.id, "output");
      if (p.phase === "cancelled") throw new Cancelled();
      if (p.phase === "error") throw new Error(p.error ?? "Restyle failed.");
      checkCancel();
    }
  };

  const renderGenerated = async (index: number, list: Segment[], outputs: Array<Blob | null>): Promise<Blob> => {
    const seg = list[index];
    const before = outputs[index - 1];
    if (!before) throw new Error("The segment before it has not rendered.");
    const after = kindInfo(seg.kind).needsNext ? outputs[index + 1] : null;
    const seed = seg.seed.trim() === "" ? undefined : Number.parseInt(seg.seed, 10);
    const previous = await getVideoGenerateProgress().catch(() => null);
    await generateVideo({
      prompt: seg.prompt,
      continuation_video: await blobToDataUrl(before),
      continuation_keep: "last",
      ...(after ? { bridge_video: await blobToDataUrl(after) } : {}),
      num_frames: seg.frames,
      fps: EDIT_FPS,
      width: EDIT_WIDTH,
      height: EDIT_HEIGHT,
      ...(seed === undefined ? {} : { seed }),
      live_preview: false,
    });
    await startedJob(cancelVideoGeneration);
    const stale = previous?.video?.id ?? null;
    for (;;) {
      await sleep(POLL_MS);
      const p = await getVideoGenerateProgress();
      if (p.active && p.total > 0) set({ stage: { index, step: p.step, total: p.total } });
      if (p.phase === "completed" && p.video && p.video.id !== stale) return fetchGalleryVideoBlob(p.video.id);
      if (p.phase === "failed") {
        const msg = p.error || "Video generation failed.";
        if (msg.toLowerCase().includes("cancelled")) throw new Cancelled();
        throw new Error(msg);
      }
      checkCancel();
    }
  };

  return {
    segments: [newSegment("real"), newSegment("continue"), newSegment("bridge"), newSegment("real")],
    running: false,
    cancelling: false,
    error: null,
    movie: null,
    movieKinds: [],
    movieFrames: [],
    stage: null,
    patch: update,
    edit: (id, patch) =>
      set((s) => ({
        segments: restage(
          s.segments.map((seg) => (seg.id === id ? { ...seg, ...patch } : seg)),
          id,
        ),
        movie: null,
      })),
    move: (index, by) =>
      set((s) => {
        const next = [...s.segments];
        const [item] = next.splice(index, 1);
        next.splice(index + by, 0, item);
        return { segments: restage(next), movie: null };
      }),
    remove: (id) => set((s) => ({ segments: restage(s.segments.filter((seg) => seg.id !== id)), movie: null })),
    add: (kind) => set((s) => ({ segments: restage([...s.segments, newSegment(kind)]), movie: null })),
    setError: (error) => set({ error }),
    render: async () => {
      if (get().running) return;
      const list = get().segments;
      cancelled = false;
      const outputs: Array<Blob | null> = list.map((s) => (s.state === "done" ? s.output : null));
      set((s) => ({
        running: true,
        error: null,
        movie: null,
        segments: s.segments.map((seg) => (seg.state === "done" ? seg : { ...seg, state: "waiting", note: null })),
      }));
      try {
        for (const index of renderOrder(list.map((s) => s.kind))) {
          if (outputs[index]) continue;
          checkCancel();
          const seg = list[index];
          update(seg.id, { state: "running" });
          set({ stage: { index, step: 0, total: 0 } });
          try {
            const out = hasSource(seg.kind) ? await renderUpload(seg) : await renderGenerated(index, list, outputs);
            outputs[index] = out;
            update(seg.id, { state: "done", output: out });
          } catch (err) {
            const stopped = err instanceof Cancelled;
            update(seg.id, {
              state: stopped ? "idle" : "error",
              note: stopped ? null : err instanceof Error ? err.message : String(err),
            });
            throw err;
          } finally {
            cancelActive = null;
          }
        }
        set({ stage: null });
        const joined = await composeVideoTimeline(outputs as Blob[], composeSpec(list.map((s) => s.kind)));
        set({ movie: joined.video, movieKinds: list.map((s) => s.kind), movieFrames: joined.frames });
      } catch (err) {
        set((s) => ({
          error: err instanceof Cancelled ? null : err instanceof Error ? err.message : String(err),
          segments: s.segments.map((seg) => (seg.state === "waiting" ? { ...seg, state: "idle" } : seg)),
        }));
      } finally {
        set({ running: false, cancelling: false, stage: null });
      }
    },
    cancel: () => {
      cancelled = true;
      set({ cancelling: true });
      void cancelActive?.().catch(() => undefined);
    },
  };
});
