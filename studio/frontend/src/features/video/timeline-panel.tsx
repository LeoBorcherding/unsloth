// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowDown01Icon, ArrowUp01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  cancelVideoGeneration,
  cancelVideoRestyle,
  composeVideoTimeline,
  fetchGalleryVideoBlob,
  fetchVideoRestyleClip,
  fetchVideoRestyleProgress,
  fetchVideoRestyleStatus,
  generateVideo,
  getVideoGenerateProgress,
  startVideoRestyle,
  type VideoRestyleStatus,
} from "./api";
import {
  composeSpec,
  DEFAULT_TIMELINE_LENGTH,
  hasSource,
  newSeconds,
  renderOrder,
  segmentProblem,
  TIMELINE_FPS,
  TIMELINE_HEIGHT,
  TIMELINE_KINDS,
  TIMELINE_LENGTHS,
  TIMELINE_WIDTH,
  type TimelineKind,
  trimProblem,
  trimSpec,
} from "./video-timeline";

const POLL_MS = 1000;

type PartState = "idle" | "waiting" | "running" | "done" | "error";

interface Part {
  id: number;
  kind: TimelineKind;
  file: File | null;
  duration?: number;
  inS: string;
  outS: string;
  prompt: string;
  frames: number;
  seed: string;
  state: PartState;
  note: string | null;
  output: Blob | null;
}

const KIND_STYLE: Record<TimelineKind, string> = {
  real: "bg-sky-600/80 text-white",
  restyle: "bg-violet-600/80 text-white",
  continue: "bg-amber-500/85 text-black",
  bridge: "bg-emerald-600/80 text-white",
};

const STATE_LABEL: Record<PartState, string> = {
  idle: "Not rendered",
  waiting: "Waiting",
  running: "Rendering",
  done: "Rendered",
  error: "Failed",
};

let nextId = 1;
function newPart(kind: TimelineKind): Part {
  return {
    id: nextId++,
    kind,
    file: null,
    inS: "0",
    outS: "",
    prompt: "",
    frames: DEFAULT_TIMELINE_LENGTH,
    seed: "",
    state: "idle",
    note: null,
    output: null,
  };
}

function label(kind: TimelineKind): string {
  return TIMELINE_KINDS.find((k) => k.value === kind)?.label ?? kind;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the clip."));
    reader.readAsDataURL(blob);
  });
}

function parseSeconds(value: string): number | null {
  if (value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : Number.NaN;
}

function useObjectUrl(blob: Blob | null): string | null {
  const url = useMemo(() => (blob ? URL.createObjectURL(blob) : null), [blob]);
  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url]);
  return url;
}

class Cancelled extends Error {}

function PartPreview({ blob }: { blob: Blob | null }) {
  const url = useObjectUrl(blob);
  if (!url) return null;
  return (
    <video src={url} muted={true} loop={true} playsInline={true} controls={true} className="w-full rounded-md bg-black" />
  );
}

/** Build one movie part by part: real clips, restyled clips, continuations and bridges between real clips. */
export function VideoTimelinePanel({
  active,
  continuationReady,
}: {
  active: boolean;
  // The loaded video model takes continuation_video (Cosmos3-Nano); null while the status is unknown.
  continuationReady: boolean | null;
}) {
  const [parts, setParts] = useState<Part[]>(() => [newPart("real"), newPart("continue"), newPart("bridge"), newPart("real")]);
  const [running, setRunning] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [movie, setMovie] = useState<Blob | null>(null);
  const [movieKinds, setMovieKinds] = useState<TimelineKind[]>([]);
  const [restyle, setRestyle] = useState<VideoRestyleStatus | null>(null);
  const [stage, setStage] = useState<{ index: number; step: number; total: number } | null>(null);
  const cancelRef = useRef<null | (() => Promise<unknown>)>(null);
  const cancelled = useRef(false);
  const movieUrl = useObjectUrl(movie);

  useEffect(() => {
    if (!active) return;
    fetchVideoRestyleStatus().then(setRestyle).catch(() => setRestyle(null));
  }, [active]);

  const update = (id: number, patch: Partial<Part>) =>
    setParts((prev) => prev.map((p) => (p.id === id ? { ...p, ...patch } : p)));
  // A generated part is rendered from its neighbours, so any change to the list makes every generated render
  // stale (an uploaded clip's render stays good unless that part itself changed), and with them the movie.
  const restage = (list: Part[], changed?: number): Part[] =>
    list.map((p) =>
      p.id === changed || !hasSource(p.kind) ? { ...p, state: "idle", note: null, output: null } : p,
    );
  const edit = (id: number, patch: Partial<Part>) => {
    setParts((prev) => restage(prev.map((p) => (p.id === id ? { ...p, ...patch } : p)), id));
    setMovie(null);
  };
  const move = (index: number, by: number) => {
    setParts((prev) => {
      const next = [...prev];
      const [item] = next.splice(index, 1);
      next.splice(index + by, 0, item);
      return restage(next);
    });
    setMovie(null);
  };

  const shapes = parts.map((p) => ({ kind: p.kind, hasFile: p.file !== null, prompt: p.prompt }));
  const problems = parts.map((p, i) => {
    const shape = segmentProblem(shapes, i);
    if (shape) return shape;
    if (hasSource(p.kind)) return trimProblem(Number(p.inS || 0), parseSeconds(p.outS), p.duration);
    return null;
  });
  const needsNano = parts.some((p) => !hasSource(p.kind));
  const needsRestyle = parts.some((p) => p.kind === "restyle");
  const blocker =
    parts.length === 0
      ? "Add a part to start."
      : needsNano && continuationReady === false
        ? "Continue and Bridge parts need Cosmos3-Nano. Pick it in the model menu above."
        : needsRestyle && restyle && !restyle.available
          ? `Restyle is not set up here: ${restyle.reason ?? "sd-server with Cosmos3 is missing."}`
          : null;
  const ready = !blocker && problems.every((p) => p === null);

  const checkCancel = () => {
    if (cancelled.current) throw new Cancelled();
  };

  const renderSource = async (part: Part): Promise<Blob> => {
    const trimmed = await composeVideoTimeline(
      [part.file as File],
      trimSpec(Number(part.inS || 0), parseSeconds(part.outS)),
    );
    if (part.kind === "real") return trimmed;
    const seed = part.seed.trim() === "" ? null : Number.parseInt(part.seed, 10);
    const started = await startVideoRestyle({
      prompt: part.prompt,
      seed,
      video: new File([trimmed], "part.mp4", { type: "video/mp4" }),
    });
    cancelRef.current = cancelVideoRestyle;
    for (;;) {
      await sleep(POLL_MS);
      const p = await fetchVideoRestyleProgress();
      if (p.id && started.id && p.id !== started.id) throw new Error("Another restyle took over the GPU.");
      if (p.phase === "done" && p.id) return fetchVideoRestyleClip(p.id, "output");
      if (p.phase === "cancelled") throw new Cancelled();
      if (p.phase === "error") throw new Error(p.error ?? "Restyle failed.");
      checkCancel();
    }
  };

  const renderGenerated = async (index: number, list: Part[], outputs: Array<Blob | null>): Promise<Blob> => {
    const part = list[index];
    const before = outputs[index - 1];
    if (!before) throw new Error("The part before it has not rendered.");
    const after = part.kind === "bridge" ? outputs[index + 1] : null;
    const seed = part.seed.trim() === "" ? undefined : Number.parseInt(part.seed, 10);
    const previous = await getVideoGenerateProgress().catch(() => null);
    await generateVideo({
      prompt: part.prompt,
      continuation_video: await blobToDataUrl(before),
      continuation_keep: "last",
      ...(after ? { bridge_video: await blobToDataUrl(after) } : {}),
      num_frames: part.frames,
      fps: TIMELINE_FPS,
      width: TIMELINE_WIDTH,
      height: TIMELINE_HEIGHT,
      ...(seed === undefined ? {} : { seed }),
      live_preview: false,
    });
    cancelRef.current = cancelVideoGeneration;
    const stale = previous?.video?.id ?? null;
    for (;;) {
      await sleep(POLL_MS);
      const p = await getVideoGenerateProgress();
      if (p.active && p.total > 0) setStage({ index, step: p.step, total: p.total });
      if (p.phase === "completed" && p.video && p.video.id !== stale) return fetchGalleryVideoBlob(p.video.id);
      if (p.phase === "failed") {
        const msg = p.error || "Video generation failed.";
        if (msg.toLowerCase().includes("cancelled")) throw new Cancelled();
        throw new Error(msg);
      }
      checkCancel();
    }
  };

  const render = async () => {
    if (running || !ready) return;
    const list = parts;
    setRunning(true);
    setError(null);
    setMovie(null);
    cancelled.current = false;
    const outputs: Array<Blob | null> = list.map((p) => (p.state === "done" ? p.output : null));
    for (const p of list) if (p.state !== "done") update(p.id, { state: "waiting", note: null });
    try {
      for (const index of renderOrder(list.map((p) => p.kind))) {
        if (outputs[index]) continue;
        checkCancel();
        const part = list[index];
        update(part.id, { state: "running" });
        setStage({ index, step: 0, total: 0 });
        try {
          const out = hasSource(part.kind) ? await renderSource(part) : await renderGenerated(index, list, outputs);
          outputs[index] = out;
          update(part.id, { state: "done", output: out });
        } catch (err) {
          update(part.id, {
            state: err instanceof Cancelled ? "idle" : "error",
            note: err instanceof Cancelled ? null : err instanceof Error ? err.message : String(err),
          });
          throw err;
        } finally {
          cancelRef.current = null;
        }
      }
      setStage(null);
      const joined = await composeVideoTimeline(outputs as Blob[], composeSpec(list.map((p) => p.kind)));
      setMovie(joined);
      setMovieKinds(list.map((p) => p.kind));
    } catch (err) {
      if (!(err instanceof Cancelled)) setError(err instanceof Error ? err.message : String(err));
      setParts((prev) => prev.map((q) => (q.state === "waiting" ? { ...q, state: "idle" } : q)));
    } finally {
      setRunning(false);
      setCancelling(false);
      setStage(null);
    }
  };

  const cancel = () => {
    cancelled.current = true;
    setCancelling(true);
    void cancelRef.current?.().catch(() => undefined);
  };

  const download = () => {
    if (!movieUrl) return;
    const a = document.createElement("a");
    a.href = movieUrl;
    a.download = "timeline.mp4";
    a.click();
  };

  return (
    <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto @[50rem]:flex-row @[50rem]:overflow-hidden">
      <div className="flex w-full shrink-0 flex-col gap-4 border-b border-border/60 px-10 pt-9 pb-6 max-sm:px-5 @[50rem]:w-[min(var(--media-rail-width,calc(408px*var(--ui-space-scale,1))),calc(100%-var(--media-header-min-width)))] @[50rem]:overflow-y-auto @[50rem]:border-r @[50rem]:border-b-0">
        <div className="grid gap-1.5">
          <h2 className="font-heading text-xl font-medium leading-none text-foreground">Build a timeline</h2>
          <p className="text-xs text-muted-foreground">
            Line up real clips and let Cosmos3-Nano fill the gaps: Continue carries on from the part before, Bridge
            travels from the part before to the start of the clip after it. Everything is joined at{" "}
            {TIMELINE_WIDTH} x {TIMELINE_HEIGHT}, {TIMELINE_FPS} fps.
          </p>
        </div>

        {blocker && (
          <Alert data-testid="timeline-blocker">
            <AlertTitle>Not ready to render</AlertTitle>
            <AlertDescription>{blocker}</AlertDescription>
          </Alert>
        )}

        <ol className="grid gap-3" aria-label="Timeline parts">
          {parts.map((part, index) => {
            const problem = problems[index];
            const source = hasSource(part.kind);
            const fieldId = `timeline-${part.id}`;
            return (
              <li
                key={part.id}
                data-testid="timeline-part"
                className="grid gap-2 rounded-lg border border-border/70 bg-card/40 p-3"
              >
                <div className="flex items-center gap-2">
                  <span className={`rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${KIND_STYLE[part.kind]}`}>
                    {index + 1}
                  </span>
                  <Select
                    value={part.kind}
                    disabled={running}
                    onValueChange={(v) => edit(part.id, { kind: v as TimelineKind })}
                  >
                    <SelectTrigger className="h-8 min-w-0 flex-1 text-sm" aria-label={`Part ${index + 1} type`}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {TIMELINE_KINDS.map((k) => (
                        <SelectItem
                          key={k.value}
                          value={k.value}
                          disabled={k.value === "restyle" && restyle !== null && !restyle.available}
                        >
                          {k.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="size-8"
                    aria-label={`Move part ${index + 1} up`}
                    disabled={running || index === 0}
                    onClick={() => move(index, -1)}
                  >
                    <HugeiconsIcon icon={ArrowUp01Icon} className="size-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="size-8"
                    aria-label={`Move part ${index + 1} down`}
                    disabled={running || index === parts.length - 1}
                    onClick={() => move(index, 1)}
                  >
                    <HugeiconsIcon icon={ArrowDown01Icon} className="size-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="size-8"
                    aria-label={`Remove part ${index + 1}`}
                    disabled={running}
                    onClick={() => {
                      setParts((prev) => restage(prev.filter((p) => p.id !== part.id)));
                      setMovie(null);
                    }}
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">{TIMELINE_KINDS.find((k) => k.value === part.kind)?.hint}</p>

                {source && (
                  <div className="grid gap-2">
                    <div className="flex min-w-0 items-center gap-2">
                      <Button asChild={true} variant="outline" size="sm" disabled={running}>
                        <label>
                          {part.file ? "Change clip" : "Choose clip"}
                          <input
                            type="file"
                            accept="video/*"
                            className="hidden"
                            disabled={running}
                            onChange={(e) => {
                              const file = e.target.files?.[0] ?? null;
                              e.target.value = "";
                              edit(part.id, { file, duration: undefined, inS: "0", outS: "" });
                              if (!file) return;
                              const probe = document.createElement("video");
                              const url = URL.createObjectURL(file);
                              probe.preload = "metadata";
                              probe.onloadedmetadata = () => {
                                if (Number.isFinite(probe.duration)) update(part.id, { duration: probe.duration });
                                URL.revokeObjectURL(url);
                              };
                              probe.onerror = () => URL.revokeObjectURL(url);
                              probe.src = url;
                            }}
                          />
                        </label>
                      </Button>
                      <span className="min-w-0 truncate text-xs text-muted-foreground" title={part.file?.name}>
                        {part.file
                          ? `${part.file.name}${part.duration !== undefined ? `, ${part.duration.toFixed(1)} s` : ""}`
                          : "No clip chosen"}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="grid gap-1">
                        <Label htmlFor={`${fieldId}-in`} className="text-xs">
                          In (s)
                        </Label>
                        <Input
                          id={`${fieldId}-in`}
                          inputMode="decimal"
                          value={part.inS}
                          disabled={running}
                          onChange={(e) => edit(part.id, { inS: e.target.value.replace(/[^0-9.]/g, "") })}
                        />
                      </div>
                      <div className="grid gap-1">
                        <Label htmlFor={`${fieldId}-out`} className="text-xs">
                          Out (s)
                        </Label>
                        <Input
                          id={`${fieldId}-out`}
                          inputMode="decimal"
                          value={part.outS}
                          placeholder="End of clip"
                          disabled={running}
                          onChange={(e) => edit(part.id, { outS: e.target.value.replace(/[^0-9.]/g, "") })}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {(part.kind !== "real") && (
                  <div className="grid gap-1">
                    <Label htmlFor={`${fieldId}-prompt`} className="text-xs">
                      {part.kind === "restyle" ? "Look" : "What happens"}
                    </Label>
                    <Textarea
                      id={`${fieldId}-prompt`}
                      rows={3}
                      value={part.prompt}
                      disabled={running}
                      placeholder={
                        part.kind === "restyle"
                          ? "A red rally car at golden hour"
                          : part.kind === "bridge"
                            ? "The car slides through the corner and drives off toward the barrier"
                            : "The car keeps going down the forest road"
                      }
                      onChange={(e) => edit(part.id, { prompt: e.target.value })}
                    />
                  </div>
                )}

                {!source && (
                  <div className="grid grid-cols-2 gap-2">
                    <div className="grid gap-1">
                      <Label className="text-xs">Length</Label>
                      <Select
                        value={String(part.frames)}
                        disabled={running}
                        onValueChange={(v) => edit(part.id, { frames: Number(v) })}
                      >
                        <SelectTrigger className="h-9 text-sm" aria-label={`Part ${index + 1} length`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {TIMELINE_LENGTHS.map((f) => (
                            <SelectItem key={f} value={String(f)}>
                              {newSeconds(part.kind, f).toFixed(1)} s new
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid gap-1">
                      <Label htmlFor={`${fieldId}-seed`} className="text-xs">
                        Seed
                      </Label>
                      <Input
                        id={`${fieldId}-seed`}
                        inputMode="numeric"
                        value={part.seed}
                        placeholder="Random"
                        disabled={running}
                        onChange={(e) => edit(part.id, { seed: e.target.value.replace(/[^0-9]/g, "") })}
                      />
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between gap-2 text-xs">
                  <span
                    data-testid="timeline-part-state"
                    className={
                      part.state === "error"
                        ? "text-destructive"
                        : part.state === "done"
                          ? "text-foreground"
                          : "text-muted-foreground"
                    }
                  >
                    {part.state === "running" && <Spinner className="mr-1 inline size-3" />}
                    {problem && part.state === "idle" ? problem : STATE_LABEL[part.state]}
                  </span>
                </div>
                {part.note && <p className="text-xs text-destructive">{part.note}</p>}
                {part.state === "done" && <PartPreview blob={part.output} />}
              </li>
            );
          })}
        </ol>

        <div className="flex flex-wrap gap-2">
          {TIMELINE_KINDS.map((k) => (
            <Button
              key={k.value}
              size="sm"
              variant="outline"
              disabled={running || (k.value === "restyle" && restyle !== null && !restyle.available)}
              onClick={() => {
                setParts((prev) => restage([...prev, newPart(k.value)]));
                setMovie(null);
              }}
            >
              + {k.label}
            </Button>
          ))}
        </div>

        {running ? (
          <Button variant="outline" className="h-11 text-ui-14" disabled={cancelling} onClick={cancel}>
            <Spinner className="mr-2 size-4" />
            {cancelling ? "Stopping..." : "Cancel"}
          </Button>
        ) : (
          <Button
            variant="primary"
            className="h-11 text-ui-14 bg-control-accent text-control-accent-foreground hover:bg-control-accent/85 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
            disabled={!ready}
            onClick={() => void render()}
          >
            Render timeline
          </Button>
        )}
        {running && stage && (
          <div className="grid gap-1.5" data-testid="timeline-progress">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {cancelling ? "Stopping" : `Part ${stage.index + 1}: ${label(parts[stage.index]?.kind ?? "real")}`}
              </span>
              {stage.total > 0 && (
                <span>
                  Step {stage.step} of {stage.total}
                </span>
              )}
            </div>
            <Progress value={stage.total > 0 ? (100 * stage.step) / stage.total : undefined} indeterminate={stage.total === 0} />
          </div>
        )}
        {error && <p className="text-xs text-destructive">{error}</p>}
      </div>

      <div className="flex min-h-[60dvh] min-w-0 flex-1 flex-col items-center justify-center-safe gap-4 p-6 @[50rem]:min-h-0 @[50rem]:overflow-y-auto">
        {movieUrl ? (
          <div className="grid w-full max-w-5xl gap-3" data-testid="timeline-movie">
            <video src={movieUrl} controls={true} playsInline={true} className="w-full rounded-lg bg-black" />
            <div className="flex h-7 w-full overflow-hidden rounded-md" aria-label="Parts in the movie">
              {movieKinds.map((kind, i) => (
                <div
                  key={`${kind}-${i}`}
                  className={`flex min-w-0 flex-1 items-center justify-center truncate px-1 text-[11px] font-semibold uppercase tracking-wide ${KIND_STYLE[kind]}`}
                  title={`Part ${i + 1}: ${label(kind)}`}
                >
                  {i + 1} {label(kind)}
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" onClick={download}>
                Download MP4
              </Button>
              <span className="text-xs text-muted-foreground">
                Generated parts are also in the video gallery.
              </span>
            </div>
          </div>
        ) : (
          <p className="max-w-md text-center text-sm text-muted-foreground">
            The joined movie plays here. Each part renders in turn: uploaded clips first, then the generated parts
            from front to back.
          </p>
        )}
      </div>
    </div>
  );
}
