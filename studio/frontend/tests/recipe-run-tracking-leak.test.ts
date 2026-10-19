// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A run resumed on recipe A must stop updating the store once recipe B is open.

import assert from "node:assert/strict";
import test from "node:test";
import { createStore } from "zustand/vanilla";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Rec = { id: string; recipeId: string; status: string; jobId?: string };
type Store = {
  getState: () => {
    executions: Rec[];
    selectedExecutionId: string | null;
    upsertExecution: (record: Rec) => void;
    resetForRecipe: (recipeId: string) => void;
  };
};

// Delays shrink 100x so a 1.2 s poll takes 12 ms.
(globalThis as { window?: unknown }).window = {
  setTimeout: (fn: () => void, ms: number) => setTimeout(fn, ms / 100),
};

const tick = (ms: number) => new Promise((r) => setTimeout(r, ms));

function sameDeps(a: unknown[] | undefined, b: unknown[] | undefined) {
  return (
    !!a && !!b && a.length === b.length && a.every((v, i) => Object.is(v, b[i]))
  );
}

// One component's worth of useState/useCallback/useEffect, with renders the test drives.
function stubReact() {
  const slots: { deps?: unknown[]; value?: unknown; cleanup?: unknown }[] = [];
  let i = 0;
  let pending: (() => void)[] = [];
  const react = {
    useState: (init: unknown) => {
      const k = i++;
      slots[k] ??= { value: typeof init === "function" ? init() : init };
      const set = (v: unknown) => {
        slots[k].value = v;
      };
      return [slots[k].value, set];
    },
    useCallback: (fn: unknown, deps: unknown[]) => {
      const k = i++;
      if (!sameDeps(slots[k]?.deps, deps)) slots[k] = { deps, value: fn };
      return slots[k].value;
    },
    useEffect: (fn: () => unknown, deps: unknown[]) => {
      const k = i++;
      const prev = slots[k];
      if (sameDeps(prev?.deps, deps)) return;
      slots[k] = { deps };
      pending.push(() => {
        (prev?.cleanup as (() => void) | undefined)?.();
        slots[k].cleanup = fn();
      });
    },
  };
  const render = <T>(run: () => T): T => {
    i = 0;
    const out = run();
    const effects = pending;
    pending = [];
    for (const effect of effects) effect();
    return out;
  };
  return { react, render };
}

test("leaving a recipe stops its resumed run from taking over the next recipe", async (t) => {
  let polls = 0;
  let jobDone = false;
  // Ends the job so a tracker that was never stopped lets the test exit.
  t.after(() => {
    jobDone = true;
  });
  const api = {
    getRecipeJobStatus: async (jobId: string) => {
      polls += 1;
      return { job_id: jobId, status: jobDone ? "completed" : "running" };
    },
    getRecipeJobDataset: async () => {
      throw new Error("404");
    },
    getRecipeJobAnalysis: async () => ({}),
    streamRecipeJobEvents: ({ signal }: { signal: AbortSignal }) =>
      new Promise<void>((resolve) =>
        signal.addEventListener("abort", () => resolve()),
      ),
  };
  const saved: Rec[] = [];
  const db = {
    listRecipeExecutions: async (recipeId: string): Promise<Rec[]> =>
      recipeId === "recipe-A"
        ? [{ id: "exec-A", recipeId, status: "running", jobId: "job-A" }]
        : [{ id: "exec-B", recipeId, status: "completed", jobId: "job-B" }],
    saveRecipeExecution: async (record: Rec) => {
      saved.push(record);
    },
  };
  const common = {
    "@/features/hub": { bumpInventoryVersion: () => {} },
    "@/shared/toast": { toastError: () => {}, toastSuccess: () => {} },
    "../api": api,
    "../data/executions-db": db,
  };
  const zustand = {
    create: (init: never) => {
      const store = createStore(init);
      return Object.assign(
        (select: (s: unknown) => unknown) => select(store.getState()),
        store,
      );
    },
  };
  const { useRecipeExecutionsStore: store } = loadWithStubs<{
    useRecipeExecutionsStore: Store;
  }>(
    new URL(
      "../src/features/recipe-studio/stores/recipe-executions.ts",
      import.meta.url,
    ),
    { ...common, zustand },
    { relativePassthrough: true },
  );
  const { react, render } = stubReact();
  const { useRecipeExecutions } = loadWithStubs<{
    useRecipeExecutions: (params: Record<string, unknown>) => unknown;
  }>(
    new URL(
      "../src/features/recipe-studio/hooks/use-recipe-executions.ts",
      import.meta.url,
    ),
    {
      ...common,
      react,
      "zustand/react/shallow": { useShallow: (select: unknown) => select },
      "../stores/recipe-executions": { useRecipeExecutionsStore: store },
      "@/config/env": {},
      "@/features/chat": {},
      "@/features/chat/presets/preset-policy": {},
      "@/features/hub/lib/model-identity": {},
      "@/features/model-picker": {},
      "@/lib/toast": {},
    },
    { relativePassthrough: true },
  );

  const payloadResult = { errors: [], payload: null };
  const open = (recipeId: string) =>
    render(() =>
      useRecipeExecutions({
        recipeId,
        currentSignature: "",
        payloadResult,
        initialRunRows: 5,
      }),
    );

  open("recipe-A");
  await tick(60);
  assert.ok(polls > 0, "the resumed run on recipe A is polled");
  assert.equal(store.getState().selectedExecutionId, "exec-A");

  open("recipe-B");
  await tick(10);
  const pollsAtSwitch = polls;
  await tick(100);

  const state = store.getState();
  assert.deepEqual(
    state.executions.map((e) => `${e.recipeId}/${e.id}`),
    ["recipe-B/exec-B"],
  );
  assert.equal(state.selectedExecutionId, "exec-B");
  assert.ok(
    polls - pollsAtSwitch <= 1,
    `job-A kept polling: ${polls - pollsAtSwitch} polls`,
  );
});

test("a stray update for another recipe is ignored by the open one", () => {
  const zustand = {
    create: (init: never) => Object.assign(() => {}, createStore(init)),
  };
  const { useRecipeExecutionsStore: store } = loadWithStubs<{
    useRecipeExecutionsStore: Store;
  }>(
    new URL(
      "../src/features/recipe-studio/stores/recipe-executions.ts",
      import.meta.url,
    ),
    { zustand },
    { relativePassthrough: true },
  );
  store.getState().resetForRecipe("recipe-B");
  store
    .getState()
    .upsertExecution({ id: "exec-B", recipeId: "recipe-B", status: "running" });
  store
    .getState()
    .upsertExecution({ id: "exec-A", recipeId: "recipe-A", status: "running" });
  assert.equal(store.getState().selectedExecutionId, "exec-B");
  assert.deepEqual(
    store.getState().executions.map((e) => e.id),
    ["exec-B"],
  );
});

test("a status reply that lands after leaving starts no more work", async () => {
  let reply: () => void = () => {};
  let statusCalls = 0;
  let datasetCalls = 0;
  const api = {
    getRecipeJobStatus: (jobId: string) => {
      statusCalls += 1;
      return new Promise((resolve) => {
        reply = () => resolve({ job_id: jobId, status: "running" });
      });
    },
    getRecipeJobDataset: async () => {
      datasetCalls += 1;
      return { dataset: [], total: 0 };
    },
    getRecipeJobAnalysis: async () => ({}),
    streamRecipeJobEvents: ({ signal }: { signal: AbortSignal }) =>
      new Promise<void>((resolve) =>
        signal.addEventListener("abort", () => resolve()),
      ),
  };
  const { trackRecipeExecution } = loadWithStubs<{
    trackRecipeExecution: (params: Record<string, unknown>) => Promise<unknown>;
  }>(
    new URL(
      "../src/features/recipe-studio/executions/tracker.ts",
      import.meta.url,
    ),
    {
      "@/features/hub": { bumpInventoryVersion: () => {} },
      "@/shared/toast": { toastError: () => {}, toastSuccess: () => {} },
      "../api": api,
    },
    { relativePassthrough: true },
  );
  const leaving = new AbortController();
  const tracking = trackRecipeExecution({
    label: "Preview",
    kind: "preview",
    rows: 1,
    jobId: "job-A",
    initialExecution: {
      id: "exec-A",
      recipeId: "recipe-A",
      status: "running",
      jobId: "job-A",
    },
    notify: false,
    onUpsert: () => {},
    onSetPreviewErrors: () => {},
    signal: leaving.signal,
  });
  await tick(5);
  leaving.abort();
  reply();

  assert.deepEqual(await tracking, { success: false, terminal: false });
  assert.equal(statusCalls, 1);
  assert.equal(datasetCalls, 0);
});

test("leaving during a completed run's final status check fetches no results", async () => {
  let reply: () => void = () => {};
  let statusCalls = 0;
  let resultCalls = 0;
  const api = {
    getRecipeJobStatus: (jobId: string) => {
      statusCalls += 1;
      if (statusCalls !== 2) {
        return Promise.resolve({ job_id: jobId, status: "completed" });
      }
      return new Promise((resolve) => {
        reply = () => resolve({ job_id: jobId, status: "completed" });
      });
    },
    getRecipeJobDataset: async () => {
      resultCalls += 1;
      return { dataset: [], total: 0 };
    },
    getRecipeJobAnalysis: async () => {
      resultCalls += 1;
      return {};
    },
    streamRecipeJobEvents: ({ signal }: { signal: AbortSignal }) =>
      new Promise<void>((resolve) =>
        signal.addEventListener("abort", () => resolve()),
      ),
  };
  const { trackRecipeExecution } = loadWithStubs<{
    trackRecipeExecution: (params: Record<string, unknown>) => Promise<unknown>;
  }>(
    new URL(
      "../src/features/recipe-studio/executions/tracker.ts",
      import.meta.url,
    ),
    {
      "@/features/hub": { bumpInventoryVersion: () => {} },
      "@/shared/toast": { toastError: () => {}, toastSuccess: () => {} },
      "../api": api,
    },
    { relativePassthrough: true },
  );
  const leaving = new AbortController();
  const tracking = trackRecipeExecution({
    label: "Preview",
    kind: "preview",
    rows: 1,
    jobId: "job-A",
    initialExecution: {
      id: "exec-A",
      recipeId: "recipe-A",
      status: "running",
      jobId: "job-A",
    },
    notify: false,
    onUpsert: () => {},
    onSetPreviewErrors: () => {},
    signal: leaving.signal,
  });
  await tick(5);
  assert.equal(statusCalls, 2, "the final status check is in flight");
  leaving.abort();
  reply();

  assert.deepEqual(await tracking, { success: false, terminal: false });
  assert.equal(statusCalls, 2);
  assert.equal(resultCalls, 0);
});

test("leaving during a final check's wait starts no further status call", async (t) => {
  let statusCalls = 0;
  const api = {
    getRecipeJobStatus: async (jobId: string) => {
      statusCalls += 1;
      return { job_id: jobId, status: "completed" };
    },
    getRecipeJobDataset: async () => ({ dataset: [], total: 0 }),
    getRecipeJobAnalysis: async () => ({}),
    streamRecipeJobEvents: ({ signal }: { signal: AbortSignal }) =>
      new Promise<void>((resolve) =>
        signal.addEventListener("abort", () => resolve()),
      ),
  };
  const { trackRecipeExecution } = loadWithStubs<{
    trackRecipeExecution: (params: Record<string, unknown>) => Promise<unknown>;
  }>(
    new URL(
      "../src/features/recipe-studio/executions/tracker.ts",
      import.meta.url,
    ),
    {
      "@/features/hub": { bumpInventoryVersion: () => {} },
      "@/shared/toast": { toastError: () => {}, toastSuccess: () => {} },
      "../api": api,
    },
    { relativePassthrough: true },
  );
  const leaving = new AbortController();
  const win = (globalThis as { window: { setTimeout: unknown } }).window;
  const original = win.setTimeout;
  t.after(() => {
    win.setTimeout = original;
  });
  // The first 250 ms wait between final checks is where the recipe is left.
  win.setTimeout = (fn: () => void, ms: number) => {
    if (ms === 250) leaving.abort();
    return setTimeout(fn, ms / 100);
  };

  const result = await trackRecipeExecution({
    label: "Preview",
    kind: "preview",
    rows: 1,
    jobId: "job-A",
    initialExecution: {
      id: "exec-A",
      recipeId: "recipe-A",
      status: "running",
      jobId: "job-A",
    },
    notify: false,
    onUpsert: () => {},
    onSetPreviewErrors: () => {},
    signal: leaving.signal,
  });

  assert.deepEqual(result, { success: false, terminal: false });
  assert.equal(statusCalls, 2);
});
