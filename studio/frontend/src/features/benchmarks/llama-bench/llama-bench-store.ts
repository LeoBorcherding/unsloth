// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The run lives on the server; this keeps the tab's view of it and owns the model swap
// around it, so leaving the tab mid-run still puts chat's model back afterwards.

import {
  getInferenceStatus,
  loadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { restore } from "../api/bench-runner";
import { chatBaseLoad } from "../api/chat-base";
import {
  type LlamaBenchConfig,
  type LlamaBenchJob,
  type SavedLlamaBenchRun,
  cancelLlamaBench,
  deleteLlamaBenchRun,
  getLlamaBenchJob,
  getLlamaBenchStatus,
  listLinkedMachines,
  listLlamaBenchRuns,
  saveRemoteLlamaBenchRun,
  startLlamaBench,
} from "./llama-bench-api";

export const DEFAULT_LLAMA_BENCH: LlamaBenchConfig = {
  prompt_tokens: [512],
  gen_tokens: [128],
  depths: [0],
  repetitions: 5,
  flash_attn: "auto",
};

const TERMINAL = new Set(["done", "error", "cancelled"]);
const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

interface LlamaBenchState {
  config: LlamaBenchConfig;
  available: boolean | null;
  upstreamAvailable: boolean;
  /** A linked instance's name, or null for this machine. */
  machine: string | null;
  machines: string[];
  setMachine: (machine: string | null) => void;
  job: LlamaBenchJob | null;
  /** Client-side phase around the server job: swapping the model in, or putting it back. */
  phase: "idle" | "loading" | "running" | "restoring";
  error: string | null;
  runs: SavedLlamaBenchRun[];
  shownId: string | null;
  setConfig: (patch: Partial<LlamaBenchConfig>) => void;
  refresh: () => Promise<void>;
  start: (model: string | null, variant: string | null) => Promise<void>;
  cancel: () => Promise<void>;
  show: (id: string | null) => void;
  remove: (id: string) => Promise<void>;
}

async function pollUntilDone(
  set: (s: Partial<LlamaBenchState>) => void,
  machine: string | null = null,
): Promise<LlamaBenchJob | null> {
  for (;;) {
    const job = await getLlamaBenchJob(undefined, machine).catch(() => null);
    if (job) set({ job });
    if (!job || TERMINAL.has(job.status)) return job;
    await sleep(1000);
  }
}

export const useLlamaBenchStore = create<LlamaBenchState>()(
  persist(
    (set, get) => ({
      config: DEFAULT_LLAMA_BENCH,
      available: null,
      upstreamAvailable: false,
      machine: null,
      machines: [],
      setMachine: (machine) => {
        if (get().phase !== "idle") return;
        set({ machine, available: null, job: null, error: null });
        void get().refresh();
      },
      job: null,
      phase: "idle",
      error: null,
      runs: [],
      shownId: null,
      setConfig: (patch) => set({ config: { ...get().config, ...patch } }),
      refresh: async () => {
        const machine = get().machine;
        const [status, runs, machines] = await Promise.all([
          getLlamaBenchStatus(undefined, machine).catch(() => null),
          listLlamaBenchRuns().catch(() => get().runs),
          listLinkedMachines().catch(() => get().machines),
        ]);
        set({
          available: status?.available ?? null,
          upstreamAvailable: status?.upstreamAvailable ?? false,
          runs,
          machines,
        });
        if (machine && !machines.includes(machine)) set({ machine: null });
        // A run started in another tab or before a reload: follow it.
        if (status?.job && get().phase === "idle") {
          set({ job: status.job });
          if (status.job.status === "running") {
            set({ phase: "running" });
            await pollUntilDone(set, machine);
            set({ phase: "idle", runs: await listLlamaBenchRuns() });
          }
        }
      },
      start: async (model, variant) => {
        if (get().phase !== "idle") return;
        const machine = get().machine;
        if (machine) {
          // A linked instance benchmarks whatever its own chat has loaded; this machine's
          // model and settings are left alone.
          set({ error: null, shownId: null, phase: "running" });
          try {
            const job = await startLlamaBench(get().config, machine);
            set({ job });
            const done = await pollUntilDone(set, machine);
            if (done?.status === "error")
              set({ error: done.error ?? "llama-bench failed" });
            if (done?.rows.length) await saveRemoteLlamaBenchRun(done, machine);
          } catch (err) {
            set({ error: err instanceof Error ? err.message : String(err) });
          } finally {
            set({
              phase: "idle",
              runs: await listLlamaBenchRuns().catch(() => get().runs),
            });
          }
          return;
        }
        set({ error: null, shownId: null, phase: "loading" });
        await useChatRuntimeStore.getState().hydratePersistedSettings();
        let status = await getInferenceStatus();
        const original = status;
        const originalLoad = chatBaseLoad(original);
        let touched = false;
        try {
          if (
            model &&
            (model !== status.active_model ||
              (variant ?? null) !== (status.gguf_variant ?? null))
          ) {
            touched = true;
            await loadModel(
              {
                ...chatBaseLoad({
                  ...status,
                  active_model: model,
                  gguf_variant: variant,
                }),
                gguf_variant: variant,
                force_reload: true,
              },
              { runtime: "chat" },
            );
            status = await getInferenceStatus();
          }
          if (!status.active_model || status.is_gguf === false)
            throw new Error(
              "Load a GGUF model first. llama-bench measures the model chat has loaded.",
            );
          const job = await startLlamaBench(get().config);
          // The server unloaded chat's model to give llama-bench the GPU.
          touched = true;
          set({ job, phase: "running" });
          const done = await pollUntilDone(set);
          if (done?.status === "error")
            set({ error: done.error ?? "llama-bench failed" });
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) });
        } finally {
          if (touched && original.active_model) {
            set({ phase: "restoring" });
            await restore(original, originalLoad).catch((err) =>
              set({
                error: `The run finished, but chat's model didn't load back: ${
                  err instanceof Error ? err.message : String(err)
                }`,
              }),
            );
          }
          set({
            phase: "idle",
            runs: await listLlamaBenchRuns().catch(() => get().runs),
          });
        }
      },
      cancel: async () => {
        await cancelLlamaBench(get().machine).catch(() => undefined);
      },
      show: (id) => set({ shownId: id }),
      remove: async (id) => {
        await deleteLlamaBenchRun(id);
        set({
          runs: get().runs.filter((r) => r.id !== id),
          shownId: get().shownId === id ? null : get().shownId,
        });
      },
    }),
    {
      name: "unsloth-llama-bench",
      partialize: (s) => ({ config: s.config, machine: s.machine }),
    },
  ),
);
