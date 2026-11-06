// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionBox } from "@/components/section-box";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { hfApiToken, useHfTokenStore } from "@/features/hub";
import { GitMergeIcon } from "@hugeicons/core-free-icons";
import { useEffect, useRef, useState } from "react";
import {
  type ExportLogEntry,
  type MergeMethod,
  type MergeProbe,
  cancelExport,
  fetchExportLogs,
  fetchMergeProbe,
  getExportStatus,
  isRecoverableTransportError,
  mergeModels,
} from "../api/export-api";
import { getExportLogLineClass } from "../lib/log-style";
import { newExportOpId } from "../lib/op-id";
import { recoverViaStatus } from "../stores/export-runtime-store";
import {
  DEFAULT_MERGE_SETTINGS,
  EMPTY_MERGE_ROW,
  MERGE_DTYPES,
  MERGE_GATE_MODES,
  MERGE_METHODS,
  MERGE_TOKENIZERS,
  MOE_ARCHITECTURES,
  type MergeSettings,
  mergeBlockReason,
  mergeMethodInfo,
  mergeRequest,
  rowsForMethod,
} from "../lib/merge";

const LOG_LINES = 400;

function Label({ htmlFor, children }: { htmlFor?: string; children: string }) {
  return (
    <label
      htmlFor={htmlFor}
      className="text-xs font-medium text-muted-foreground"
    >
      {children}
    </label>
  );
}

/** Merges models with MergeKit; the result lands in outputs and is exported like a training run. */
export function MergeModelsCard({
  onMerged,
}: {
  onMerged: (runName: string) => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const [s, setS] = useState<MergeSettings>(DEFAULT_MERGE_SETTINGS);
  const [consent, setConsent] = useState<MergeProbe | null>(null);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<ExportLogEntry[]>([]);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(
    null,
  );
  const runningRef = useRef(false);
  const opIdRef = useRef<string | null>(null);
  // Leaving the page stops the log poll; the merge itself keeps running server-side.
  useEffect(
    () => () => {
      runningRef.current = false;
    },
    [],
  );

  const info = mergeMethodInfo(s.method);
  const moe = s.method === "moe";
  const blockReason = mergeBlockReason(s);
  const update = (patch: Partial<MergeSettings>) =>
    setS((prev) => ({ ...prev, ...patch }));
  const updateRow = (index: number, patch: Partial<MergeSettings["rows"][0]>) =>
    setS((prev) => ({
      ...prev,
      rows: prev.rows.map((r, i) => (i === index ? { ...r, ...patch } : r)),
    }));

  async function pollLogs(opId: string): Promise<void> {
    let cursor: number | null = null;
    while (true) {
      try {
        if (cursor === null) {
          // Queued behind another export, its lines are not ours: wait until this op starts,
          // then read from its run start (the merge clears the buffer as it begins).
          const st = await getExportStatus(opId);
          if (st.active_op_id === opId || st.last_op_id === opId) {
            const page = await fetchExportLogs(null);
            cursor = page.cursor;
            setLogs(page.entries.slice(-LOG_LINES));
          }
        } else {
          const page = await fetchExportLogs(cursor);
          cursor = page.cursor;
          if (page.entries.length > 0) {
            setLogs((prev) => [...prev, ...page.entries].slice(-LOG_LINES));
          }
        }
      } catch {
        // A dropped poll is retried on the next tick.
      }
      if (!runningRef.current) return;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  async function runMerge(installMissing: boolean) {
    setConsent(null);
    setRunning(true);
    runningRef.current = true;
    setLogs([]);
    setResult(null);
    let poller: Promise<void> | null = null;
    const opId = newExportOpId();
    opIdRef.current = opId;
    try {
      poller = pollLogs(opId);
      const baseline = await getExportStatus()
        .then((st) => st.last_op_seq ?? 0)
        .catch(() => null);
      let outputPath: string | null;
      try {
        const response = await mergeModels({
          ...mergeRequest(s),
          hf_token: hfApiToken(hfToken) ?? null,
          install_missing_dependencies: installMissing,
          op_id: opId,
        });
        outputPath = response.details?.output_path ?? null;
      } catch (error) {
        // A tunnel timeout cuts the POST off while the merge keeps running: settle it via status.
        if (!isRecoverableTransportError(error)) throw error;
        ({ outputPath } = await recoverViaStatus(
          baseline,
          () => opIdRef.current === opId,
          opId,
        ));
      }
      const run = s.outputName.trim();
      setResult({
        ok: true,
        text: `Saved to ${outputPath ?? run}. It is listed under Training runs.`,
      });
      onMerged(run);
    } catch (error) {
      setResult({
        ok: false,
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      runningRef.current = false;
      await poller;
      setRunning(false);
    }
  }

  async function handleMerge() {
    setResult(null);
    let probe: MergeProbe;
    try {
      probe = await fetchMergeProbe();
    } catch (error) {
      setResult({
        ok: false,
        text: error instanceof Error ? error.message : String(error),
      });
      return;
    }
    if (probe.blocked_reason) {
      setResult({ ok: false, text: probe.blocked_reason });
    } else if (!probe.ready) {
      setConsent(probe);
    } else {
      await runMerge(false);
    }
  }

  return (
    <SectionBox
      icon={GitMergeIcon}
      chipTint="var(--chart-3)"
      title="Merge models"
      description="Combine two or more models with MergeKit. The merged model is saved as a training run you can then export or push to the Hub."
    >
      <div className="flex flex-col gap-6">
        <div className="grid gap-4 @md/train-section:grid-cols-3">
          <div className="flex flex-col gap-2">
            <Label>Method</Label>
            <Select
              value={s.method}
              onValueChange={(v) => {
                const method = v as MergeMethod;
                update({ method, rows: rowsForMethod(method, s.rows) });
              }}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MERGE_METHODS.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-2">
            <Label>Precision</Label>
            <Select
              value={s.dtype}
              onValueChange={(v) =>
                update({ dtype: v as MergeSettings["dtype"] })
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MERGE_DTYPES.map((d) => (
                  <SelectItem key={d} value={d}>
                    {d}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="merge-output">Output name</Label>
            <Input
              id="merge-output"
              value={s.outputName}
              onChange={(e) => update({ outputName: e.target.value })}
            />
          </div>
        </div>
        <p className="-mt-3 text-xs text-muted-foreground">{info.hint}</p>

        {!moe && (
          <div className="flex flex-col gap-2">
            <Label>Tokenizer</Label>
            <Select
              value={s.tokenizerSource}
              onValueChange={(v) =>
                update({
                  tokenizerSource: v as MergeSettings["tokenizerSource"],
                })
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MERGE_TOKENIZERS.map((o) => (
                  <SelectItem key={o.value} value={o.value}>
                    {o.value === "base" && s.method === "linear"
                      ? "Models' own (vocabularies must match)"
                      : o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {info.base && (
          <div className="flex flex-col gap-2">
            <Label htmlFor="merge-base">
              {moe
                ? "Base model (attention, embeddings and tokenizer)"
                : "Base model"}
            </Label>
            <Input
              id="merge-base"
              placeholder="mistralai/Mistral-7B-v0.1 or a local folder"
              value={s.baseModel}
              onChange={(e) => update({ baseModel: e.target.value })}
            />
          </div>
        )}

        <div className="flex flex-col gap-2">
          <Label>{moe ? "Experts" : "Models"}</Label>
          {s.rows.map((row, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: rows have no identity besides their position
            <div key={i} className="flex flex-wrap items-center gap-2">
              <Input
                aria-label={`Model ${i + 1}`}
                className="min-w-48 flex-1"
                placeholder="Hugging Face repo or local folder"
                value={row.model}
                onChange={(e) => updateRow(i, { model: e.target.value })}
              />
              {moe && (
                <Textarea
                  aria-label={`Prompts ${i + 1}`}
                  className="min-w-48 flex-1"
                  rows={1}
                  placeholder="Prompts for this expert, one per line or comma separated"
                  value={row.prompts}
                  onChange={(e) => updateRow(i, { prompts: e.target.value })}
                />
              )}
              {s.method !== "slerp" && !moe && (
                <Input
                  aria-label={`Weight ${i + 1}`}
                  className="w-24"
                  type="number"
                  step="0.05"
                  title="Weight"
                  value={row.weight}
                  onChange={(e) =>
                    updateRow(i, { weight: e.target.valueAsNumber })
                  }
                />
              )}
              {info.density && (
                <Input
                  aria-label={`Density ${i + 1}`}
                  className="w-24"
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  title="Density"
                  value={row.density}
                  onChange={(e) =>
                    updateRow(i, { density: e.target.valueAsNumber })
                  }
                />
              )}
              {s.method !== "slerp" && s.rows.length > 1 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    update({ rows: s.rows.filter((_, j) => j !== i) })
                  }
                >
                  Remove
                </Button>
              )}
            </div>
          ))}
          {s.method !== "slerp" && (
            <Button
              variant="outline"
              size="sm"
              className="self-start"
              onClick={() => update({ rows: [...s.rows, EMPTY_MERGE_ROW] })}
            >
              {moe ? "Add expert" : "Add model"}
            </Button>
          )}
        </div>

        <div className="flex flex-wrap items-end gap-6">
          {s.method === "slerp" && (
            <div className="flex flex-col gap-2">
              <Label htmlFor="merge-t">t</Label>
              <Input
                id="merge-t"
                className="w-24"
                type="number"
                step="0.05"
                min={0}
                max={1}
                value={s.t}
                onChange={(e) => update({ t: e.target.valueAsNumber })}
              />
            </div>
          )}
          {moe && (
            <>
              <div className="flex flex-col gap-2">
                <Label>Router</Label>
                <Select
                  value={s.gateMode}
                  onValueChange={(v) =>
                    update({ gateMode: v as MergeSettings["gateMode"] })
                  }
                >
                  <SelectTrigger className="w-64">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MERGE_GATE_MODES.map((g) => (
                      <SelectItem key={g.value} value={g.value}>
                        {g.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="merge-experts-per-token">
                  Experts per token
                </Label>
                <Input
                  id="merge-experts-per-token"
                  className="w-24"
                  type="number"
                  step="1"
                  min={1}
                  max={s.rows.length}
                  value={s.expertsPerToken}
                  onChange={(e) =>
                    update({ expertsPerToken: e.target.valueAsNumber })
                  }
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label>Output architecture</Label>
                <Select
                  value={s.architecture}
                  onValueChange={(v) =>
                    update({ architecture: v as MergeSettings["architecture"] })
                  }
                >
                  <SelectTrigger className="w-64">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MOE_ARCHITECTURES.map((a) => (
                      <SelectItem key={a.value} value={a.value}>
                        {a.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </>
          )}
          <div className="flex items-center gap-2">
            <Switch
              id="merge-cuda"
              checked={s.cuda}
              onCheckedChange={(cuda) => update({ cuda })}
            />
            <Label htmlFor="merge-cuda">Merge on the GPU</Label>
          </div>
        </div>

        {consent && (
          <div className="flex flex-col gap-3 rounded-lg border border-border/60 p-3 text-xs">
            <p>{consent.install_summary}</p>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setConsent(null)}>
                Not now
              </Button>
              <Button variant="dark" onClick={() => void runMerge(true)}>
                Install and merge
              </Button>
            </div>
          </div>
        )}

        {logs.length > 0 && (
          <div className="h-48 w-full overflow-auto scroll-rounded rounded-lg border border-border/40 bg-black/85 p-3 font-mono text-ui-11 leading-[1.45] text-emerald-200/90 whitespace-pre-wrap break-words">
            {logs.map((entry, idx) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: log lines are append-only
              <div key={idx} className={getExportLogLineClass(entry)}>
                {entry.line}
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center justify-between gap-3">
          <p
            className={
              result?.ok === false
                ? "text-xs text-destructive"
                : "text-xs text-muted-foreground"
            }
          >
            {result?.text ??
              blockReason ??
              "MergeKit is installed on first use."}
          </p>
          {running ? (
            <Button
              variant="destructive"
              onClick={() => void cancelExport(opIdRef.current ?? undefined)}
            >
              Cancel
            </Button>
          ) : (
            <Button
              variant="dark"
              disabled={!!blockReason || !!consent}
              onClick={() => void handleMerge()}
            >
              Merge
            </Button>
          )}
        </div>
      </div>
    </SectionBox>
  );
}
