// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowDown01Icon, ArrowUp01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  fetchVideoRestyleStatus,
  getVideoStatus,
  loadVideoModel,
  type VideoRestyleStatus,
  type VideoStatus,
} from "../video/api";
import { parseSeconds, type SegmentState, useVideoEditingStore } from "./editing-store";
import {
  COSMOS3_NANO_REPO,
  EDIT_FPS,
  EDIT_HEIGHT,
  EDIT_WIDTH,
  enginesNeeded,
  hasSource,
  kindInfo,
  newSeconds,
  SEGMENT_KINDS,
  SEGMENT_LENGTHS,
  type SegmentKind,
  segmentProblem,
  trimProblem,
} from "./segments";

// One hue per kind, used for the number badge and the movie's segment bar.
const KIND_STYLE: Record<SegmentKind, string> = {
  real: "bg-sky-600 text-white",
  restyle: "bg-violet-600 text-white",
  continue: "bg-amber-500 text-black",
  bridge: "bg-emerald-600 text-white",
};

const STATE_STYLE: Record<SegmentState, string> = {
  idle: "bg-muted text-muted-foreground",
  waiting: "bg-muted text-muted-foreground",
  running: "bg-primary/15 text-foreground",
  done: "bg-emerald-600/15 text-emerald-700 dark:text-emerald-300",
  error: "bg-destructive/15 text-destructive",
};

const STATE_LABEL: Record<SegmentState, string> = {
  idle: "Not rendered",
  waiting: "Waiting",
  running: "Rendering",
  done: "Rendered",
  error: "Failed",
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function useObjectUrl(blob: Blob | null): string | null {
  const url = useMemo(() => (blob ? URL.createObjectURL(blob) : null), [blob]);
  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url]);
  return url;
}

function ClipPreview({ blob, label }: { blob: Blob | null; label: string }) {
  const url = useObjectUrl(blob);
  if (!url) return null;
  return (
    <video
      src={url}
      aria-label={label}
      muted={true}
      loop={true}
      playsInline={true}
      controls={true}
      className="w-full rounded-md bg-black"
    />
  );
}

function SetupRow({ name, value, action }: { name: string; value: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-ui-11 text-muted-foreground">{name}</span>
      <div className="flex min-h-8 min-w-0 items-center gap-2 text-ui-13 text-foreground">{value}{action}</div>
    </div>
  );
}

/** Build one movie segment by segment: real clips, restyled clips, continuations, and bridges onto the next clip. */
export function VideoEditingPage() {
  const segments = useVideoEditingStore((s) => s.segments);
  const running = useVideoEditingStore((s) => s.running);
  const cancelling = useVideoEditingStore((s) => s.cancelling);
  const error = useVideoEditingStore((s) => s.error);
  const movie = useVideoEditingStore((s) => s.movie);
  const movieKinds = useVideoEditingStore((s) => s.movieKinds);
  const stage = useVideoEditingStore((s) => s.stage);
  const { patch: update, edit, move, remove, add, setError, render, cancel } = useVideoEditingStore.getState();
  const [restyle, setRestyle] = useState<VideoRestyleStatus | null>(null);
  const [status, setStatus] = useState<VideoStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const movieUrl = useObjectUrl(movie);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getVideoStatus());
    } catch {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    fetchVideoRestyleStatus().then(setRestyle).catch(() => setRestyle(null));
  }, [refreshStatus]);

  const nanoReady = status?.supports_video_continuation === true;

  const loadNano = async () => {
    setLoading(true);
    setError(null);
    try {
      await loadVideoModel({ model_path: COSMOS3_NANO_REPO });
      for (;;) {
        await sleep(2000);
        const next = await getVideoStatus();
        setStatus(next);
        if (next.loaded && next.supports_video_continuation) break;
        if (next.loaded) throw new Error(`${next.repo_id ?? "Another model"} loaded instead of Cosmos3-Nano.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const shapes = segments.map((s) => ({ kind: s.kind, hasFile: s.file !== null, prompt: s.prompt }));
  const problems = segments.map((s, i) => {
    const shape = segmentProblem(shapes, i);
    if (shape) return shape;
    if (hasSource(s.kind)) return trimProblem(Number(s.inS || 0), parseSeconds(s.outS), s.duration);
    return null;
  });
  const engines = enginesNeeded(segments.map((s) => s.kind));
  const blocker =
    segments.length === 0
      ? "Add a segment to start."
      : engines.has("cosmos3-nano") && !nanoReady
        ? "Continue and Bridge segments need Cosmos3-Nano loaded. Use Load above."
        : engines.has("cosmos3-sim-transfer") && restyle && !restyle.available
          ? `Restyle is not set up here: ${restyle.reason ?? "sd-server with Cosmos3 is missing."}`
          : null;
  const ready = !blocker && problems.every((p) => p === null);

  const download = () => {
    if (!movieUrl) return;
    const a = document.createElement("a");
    a.href = movieUrl;
    a.download = "video-edit.mp4";
    a.click();
  };

  const renderButton = running ? (
    <Button variant="outline" className="rounded-full" disabled={cancelling} onClick={cancel}>
      <Spinner className="mr-2 size-4" />
      {cancelling ? "Stopping..." : "Cancel"}
    </Button>
  ) : (
    <Button
      variant="primary"
      className="rounded-full bg-control-accent text-control-accent-foreground hover:bg-control-accent/85 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
      disabled={!ready || loading}
      onClick={() => void render()}
    >
      Render video
    </Button>
  );

  return (
    <main className="mx-auto w-full max-w-6xl px-6 pb-10 pt-8 font-heading max-sm:px-4 sm:px-10">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
          Video editing
        </h1>
        {renderButton}
      </div>
      <p className="mt-4 text-ui-15 text-muted-foreground">
        Line up real clips and let Cosmos fill the gaps. Continue carries on from the segment before; Bridge travels
        from the segment before to the start of the clip after it; Restyle repaints a clip from a prompt. Everything is
        joined at {EDIT_WIDTH} x {EDIT_HEIGHT}, {EDIT_FPS} fps.
      </p>

      <section
        aria-label="Setup"
        className="mt-6 grid grid-cols-1 gap-3 rounded-xl border border-border/60 bg-muted/20 p-4 md:grid-cols-3"
      >
        <SetupRow
          name="Cosmos3-Nano (Continue, Bridge)"
          value={
            <span className={cn("truncate", !nanoReady && "text-muted-foreground")}>
              {nanoReady
                ? "Loaded"
                : status?.loaded
                  ? `${status.display_repo_id ?? status.repo_id ?? "Another model"} is loaded`
                  : "Not loaded"}
            </span>
          }
          action={
            !nanoReady && (
              <Button size="sm" variant="outline" disabled={loading || running} onClick={() => void loadNano()}>
                {loading && <Spinner className="mr-1.5 size-3.5" />}
                {loading ? "Loading..." : status?.loaded ? "Switch to Cosmos3-Nano" : "Load"}
              </Button>
            )
          }
        />
        <SetupRow
          name="Cosmos3-Nano-Sim-Transfer (Restyle)"
          value={
            <span className={cn("truncate", !restyle?.available && "text-muted-foreground")} title={restyle?.reason ?? undefined}>
              {restyle === null ? "Checking..." : restyle.available ? "Ready" : "Not set up"}
            </span>
          }
        />
        <SetupRow name="Output" value={`${EDIT_WIDTH} x ${EDIT_HEIGHT}, ${EDIT_FPS} fps, MP4`} />
      </section>

      {blocker && (
        <Alert className="mt-4" data-testid="edit-blocker">
          <AlertTitle>Not ready to render</AlertTitle>
          <AlertDescription>{blocker}</AlertDescription>
        </Alert>
      )}
      {error && (
        <Alert variant="destructive" className="mt-4" data-testid="edit-error">
          <AlertTitle>That did not work</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <section aria-label="Segments" className="grid content-start gap-3">
          <h2 className="text-ui-15 font-semibold text-foreground">Segments</h2>
          <ol className="grid gap-3">
            {segments.map((seg, index) => {
              const info = kindInfo(seg.kind);
              const problem = problems[index];
              const fieldId = `edit-${seg.id}`;
              return (
                <li
                  key={seg.id}
                  data-testid="edit-segment"
                  className="grid gap-2.5 rounded-xl border border-border/60 bg-card/40 p-3.5"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "flex size-6 shrink-0 items-center justify-center rounded-md text-ui-11 font-semibold",
                        KIND_STYLE[seg.kind],
                      )}
                    >
                      {index + 1}
                    </span>
                    <Select value={seg.kind} disabled={running} onValueChange={(v) => edit(seg.id, { kind: v as SegmentKind })}>
                      <SelectTrigger className="h-8 min-w-0 flex-1 text-ui-13" aria-label={`Segment ${index + 1} type`}>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {SEGMENT_KINDS.map((k) => (
                          <SelectItem
                            key={k.value}
                            value={k.value}
                            disabled={k.engine === "cosmos3-sim-transfer" && restyle !== null && !restyle.available}
                          >
                            {k.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <span
                      data-testid="edit-segment-state"
                      className={cn(
                        "shrink-0 rounded-full px-2 py-0.5 text-ui-11 font-medium",
                        STATE_STYLE[seg.state],
                      )}
                    >
                      {seg.state === "running" && <Spinner className="mr-1 inline size-3" />}
                      {STATE_LABEL[seg.state]}
                    </span>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-8"
                      aria-label={`Move segment ${index + 1} up`}
                      disabled={running || index === 0}
                      onClick={() => move(index, -1)}
                    >
                      <HugeiconsIcon icon={ArrowUp01Icon} className="size-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-8"
                      aria-label={`Move segment ${index + 1} down`}
                      disabled={running || index === segments.length - 1}
                      onClick={() => move(index, 1)}
                    >
                      <HugeiconsIcon icon={ArrowDown01Icon} className="size-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-8"
                      aria-label={`Remove segment ${index + 1}`}
                      disabled={running}
                      onClick={() => remove(seg.id)}
                    >
                      <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                    </Button>
                  </div>
                  <p className="text-ui-12 text-muted-foreground">{info.hint}</p>

                  {info.source === "upload" && (
                    <div className="grid gap-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <Button asChild={true} variant="outline" size="sm" disabled={running}>
                          <label>
                            {seg.file ? "Change clip" : "Choose clip"}
                            <input
                              type="file"
                              accept="video/*"
                              className="hidden"
                              disabled={running}
                              onChange={(e) => {
                                const file = e.target.files?.[0] ?? null;
                                e.target.value = "";
                                edit(seg.id, { file, duration: undefined, inS: "0", outS: "" });
                                if (!file) return;
                                const probe = document.createElement("video");
                                const url = URL.createObjectURL(file);
                                probe.preload = "metadata";
                                probe.onloadedmetadata = () => {
                                  if (Number.isFinite(probe.duration)) update(seg.id, { duration: probe.duration });
                                  URL.revokeObjectURL(url);
                                };
                                probe.onerror = () => URL.revokeObjectURL(url);
                                probe.src = url;
                              }}
                            />
                          </label>
                        </Button>
                        <span className="min-w-0 truncate text-ui-12 text-muted-foreground" title={seg.file?.name}>
                          {seg.file
                            ? `${seg.file.name}${seg.duration !== undefined ? `, ${seg.duration.toFixed(1)} s` : ""}`
                            : "No clip chosen"}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="grid gap-1">
                          <Label htmlFor={`${fieldId}-in`} className="text-ui-12">
                            In (s)
                          </Label>
                          <Input
                            id={`${fieldId}-in`}
                            inputMode="decimal"
                            value={seg.inS}
                            disabled={running}
                            onChange={(e) => edit(seg.id, { inS: e.target.value.replace(/[^0-9.]/g, "") })}
                          />
                        </div>
                        <div className="grid gap-1">
                          <Label htmlFor={`${fieldId}-out`} className="text-ui-12">
                            Out (s)
                          </Label>
                          <Input
                            id={`${fieldId}-out`}
                            inputMode="decimal"
                            value={seg.outS}
                            placeholder="End of clip"
                            disabled={running}
                            onChange={(e) => edit(seg.id, { outS: e.target.value.replace(/[^0-9.]/g, "") })}
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {info.prompt && (
                    <div className="grid gap-1">
                      <Label htmlFor={`${fieldId}-prompt`} className="text-ui-12">
                        {info.prompt.label}
                      </Label>
                      <Textarea
                        id={`${fieldId}-prompt`}
                        rows={3}
                        value={seg.prompt}
                        disabled={running}
                        placeholder={info.prompt.placeholder}
                        onChange={(e) => edit(seg.id, { prompt: e.target.value })}
                      />
                    </div>
                  )}

                  {(info.source === "previous" || seg.kind === "restyle") && (
                    <div className="grid grid-cols-2 gap-2">
                      {info.source === "previous" ? (
                        <div className="grid gap-1">
                          <Label className="text-ui-12">Length</Label>
                          <Select
                            value={String(seg.frames)}
                            disabled={running}
                            onValueChange={(v) => edit(seg.id, { frames: Number(v) })}
                          >
                            <SelectTrigger className="h-9 text-ui-13" aria-label={`Segment ${index + 1} length`}>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {SEGMENT_LENGTHS.map((f) => (
                                <SelectItem key={f} value={String(f)}>
                                  {newSeconds(seg.kind, f).toFixed(1)} s of new footage
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      ) : (
                        <span />
                      )}
                      <div className="grid gap-1">
                        <Label htmlFor={`${fieldId}-seed`} className="text-ui-12">
                          Seed
                        </Label>
                        <Input
                          id={`${fieldId}-seed`}
                          inputMode="numeric"
                          value={seg.seed}
                          placeholder="Random"
                          disabled={running}
                          onChange={(e) => edit(seg.id, { seed: e.target.value.replace(/[^0-9]/g, "") })}
                        />
                      </div>
                    </div>
                  )}

                  {problem && seg.state === "idle" && (
                    <p className="text-ui-12 text-muted-foreground" data-testid="edit-segment-problem">
                      {problem}
                    </p>
                  )}
                  {seg.note && <p className="text-ui-12 text-destructive">{seg.note}</p>}
                  {seg.state === "done" && <ClipPreview blob={seg.output} label={`Segment ${index + 1} output`} />}
                </li>
              );
            })}
          </ol>
          <div className="flex flex-wrap gap-2">
            {SEGMENT_KINDS.map((k) => (
              <Button
                key={k.value}
                size="sm"
                variant="outline"
                disabled={running || (k.engine === "cosmos3-sim-transfer" && restyle !== null && !restyle.available)}
                onClick={() => add(k.value)}
              >
                + {k.label}
              </Button>
            ))}
          </div>
        </section>

        <section aria-label="Movie" className="grid content-start gap-3 lg:sticky lg:top-6 lg:self-start">
          <h2 className="text-ui-15 font-semibold text-foreground">Movie</h2>
          {running && stage && (
            <div className="grid gap-1.5 rounded-xl border border-border/60 bg-muted/20 p-3" data-testid="edit-progress">
              <div className="flex justify-between text-ui-12 text-muted-foreground">
                <span>
                  {cancelling
                    ? "Stopping"
                    : `Segment ${stage.index + 1} of ${segments.length}: ${kindInfo(segments[stage.index]?.kind ?? "real").label}`}
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
          {movieUrl ? (
            <div className="grid gap-3" data-testid="edit-movie">
              <video src={movieUrl} controls={true} playsInline={true} className="w-full rounded-lg bg-black" />
              <div className="flex h-7 w-full overflow-hidden rounded-md" aria-label="Segments in the movie">
                {movieKinds.map((kind, i) => (
                  <div
                    key={`${kind}-${i}`}
                    className={cn(
                      "flex min-w-0 flex-1 items-center justify-center truncate px-1 text-ui-11 font-semibold",
                      KIND_STYLE[kind],
                    )}
                    title={`Segment ${i + 1}: ${kindInfo(kind).label}`}
                  >
                    {i + 1} {kindInfo(kind).label}
                  </div>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant="outline" onClick={download}>
                  Download MP4
                </Button>
                <span className="text-ui-12 text-muted-foreground">Generated segments are also in the video gallery.</span>
              </div>
            </div>
          ) : (
            <div className="flex aspect-video w-full items-center justify-center rounded-lg border border-dashed border-border/70 p-6 text-center text-ui-13 text-muted-foreground">
              The joined movie plays here. Uploaded clips render first, then the generated segments from front to back.
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
