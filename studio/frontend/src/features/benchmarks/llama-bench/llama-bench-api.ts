// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// llama-bench runs server-side under /api/benchmarks/llama-bench; finished runs are saved
// with the config sweeps as kind "llama-bench".

import { authFetch } from "@/features/auth";

export interface LlamaBenchConfig {
  prompt_tokens: number[];
  gen_tokens: number[];
  depths: number[];
  repetitions: number;
  flash_attn: "auto" | "on" | "off";
  n_gpu_layers?: number | null;
  /** "upstream" runs the ggml-org build the machine names in UNSLOTH_LLAMA_BENCH_UPSTREAM. */
  build?: LlamaBenchBuild;
}

export type LlamaBenchBuild = "unsloth" | "upstream";

export interface LlamaBenchRow {
  test: string;
  n_prompt: number;
  n_gen: number;
  n_depth: number;
  avg_ts: number;
  stddev_ts: number;
  samples_ts: number[];
  n_gpu_layers?: number | null;
  flash_attn?: number | boolean | string | null;
}

export interface LlamaBenchMeta {
  build_commit?: string | null;
  build_number?: number | null;
  gpu_info?: string | null;
  backends?: string | null;
  model_type?: string | null;
  model_size?: number | null;
  model_n_params?: number | null;
  build?: LlamaBenchBuild | null;
  /** The linked instance it ran on; absent for this machine. */
  machine?: string | null;
}

export type LlamaBenchStatus = "running" | "done" | "error" | "cancelled";

export interface LlamaBenchJob {
  id: string;
  status: LlamaBenchStatus;
  stage: string;
  error: string | null;
  model: string;
  ggufVariant: string | null;
  config: LlamaBenchConfig;
  rows: LlamaBenchRow[];
  meta: LlamaBenchMeta;
  done: number;
  total: number;
  log: string[];
  createdAt: number;
  finishedAt: number | null;
}

/** A saved run as /runs lists it; the rows ride in `outcomes`. */
export interface SavedLlamaBenchRun {
  id: string;
  model: string;
  ggufVariant: string | null;
  config: LlamaBenchConfig;
  meta: LlamaBenchMeta;
  outcomes: LlamaBenchRow[];
  createdAt: number;
  finishedAt: number | null;
}

async function parse<T>(res: Response, what: string): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : typeof detail?.message === "string"
          ? detail.message
          : `${what} failed (${res.status})`;
    throw new Error(message);
  }
  return (await res.json()) as T;
}

const LOCAL = "/api/benchmarks/llama-bench";

/** A linked instance's llama-bench goes through this server, which adds that instance's key. */
const base = (machine?: string | null) =>
  machine
    ? `/api/linked-instances/${encodeURIComponent(machine)}/proxy/api/benchmarks/llama-bench`
    : LOCAL;

export async function getLlamaBenchStatus(
  signal?: AbortSignal,
  machine?: string | null,
): Promise<{
  available: boolean;
  upstreamAvailable?: boolean;
  model: string | null;
  ggufVariant: string | null;
  job: LlamaBenchJob | null;
}> {
  return parse(
    await authFetch(`${base(machine)}/status`, { signal }),
    "Reading llama-bench",
  );
}

export async function startLlamaBench(
  config: LlamaBenchConfig,
  machine?: string | null,
): Promise<LlamaBenchJob> {
  const res = await authFetch(`${base(machine)}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return parse(res, "Starting llama-bench");
}

export async function getLlamaBenchJob(
  signal?: AbortSignal,
  machine?: string | null,
): Promise<LlamaBenchJob | null> {
  const res = await authFetch(`${base(machine)}/run`, { signal });
  return (await parse<{ job: LlamaBenchJob | null }>(res, "Reading llama-bench"))
    .job;
}

export async function cancelLlamaBench(machine?: string | null): Promise<void> {
  await authFetch(`${base(machine)}/run`, { method: "DELETE" });
}

/** Keeps a linked instance's finished run in this machine's history, tagged with where it ran. */
export async function saveRemoteLlamaBenchRun(
  job: LlamaBenchJob,
  machine: string,
): Promise<void> {
  const res = await authFetch(
    `/api/benchmarks/runs/${encodeURIComponent(job.id)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: job.id,
        kind: "llama-bench",
        sweep: "llama-bench",
        model: job.model,
        ggufVariant: job.ggufVariant,
        config: job.config,
        meta: { ...job.meta, machine },
        outcomes: job.rows,
        createdAt: job.createdAt,
        finishedAt: job.finishedAt,
      }),
    },
  );
  if (!res.ok) throw new Error(`Saving the run here failed (${res.status})`);
}

export async function listLinkedMachines(
  signal?: AbortSignal,
): Promise<string[]> {
  const res = await authFetch("/api/linked-instances", { signal });
  if (!res.ok) return [];
  const list = (await res.json()) as { name: string }[];
  return list.map((i) => i.name);
}

export async function listLlamaBenchRuns(
  signal?: AbortSignal,
): Promise<SavedLlamaBenchRun[]> {
  const res = await authFetch("/api/benchmarks/runs?kind=llama-bench", {
    signal,
  });
  return (await parse<{ runs: SavedLlamaBenchRun[] }>(res, "Listing runs"))
    .runs;
}

export async function deleteLlamaBenchRun(id: string): Promise<void> {
  const res = await authFetch(
    `/api/benchmarks/runs/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!res.ok && res.status !== 404)
    throw new Error(`Deleting the run failed (${res.status})`);
}
