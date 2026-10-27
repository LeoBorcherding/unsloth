// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Run stays clickable while /validate and the local model load run, the Runs view is wider
// than the page with the sidebar open, the progress island shows only "--" before the first
// rows and sits on the graph, and the payload carries a field nothing reads.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

registerBundlerResolver();

const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);
const { createEmptyRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/empty.ts"
);
const helpers = await import(
  "../src/features/recipe-studio/components/executions/executions-view-helpers.ts"
);
const fitView = await import(
  "../src/features/recipe-studio/utils/graph/fit-view.ts"
);
const { isExecutionInProgress } = await import(
  "../src/features/recipe-studio/executions/execution-helpers.ts"
);

type Hook = {
  runPreview: () => Promise<boolean>;
  runFull: () => Promise<boolean>;
};

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

function mountExecutionsHook({ holdJob = false } = {}) {
  const calls = { validate: 0, createJob: 0, loading: [] as boolean[] };
  const validations: Array<ReturnType<typeof deferred<unknown>>> = [];
  const jobs: Array<ReturnType<typeof deferred<unknown>>> = [];
  const state = {
    runDialogOpen: false,
    runDialogKind: "preview",
    previewRows: 5,
    fullRows: 10,
    fullRunName: "named run",
    runErrors: [],
    runSettings: {},
    previewLoading: false,
    fullLoading: false,
    executions: [],
    selectedExecutionId: null,
    setRunDialogOpen: () => {},
    setRunDialogKind: () => {},
    setPreviewRows: () => {},
    setFullRows: () => {},
    setFullRunName: () => {},
    setRunErrors: () => {},
    setRunSettings: () => {},
    setPreviewLoading: (value: boolean) => calls.loading.push(value),
    setFullLoading: () => {},
    setExecutions: () => {},
    upsertExecution: () => {},
    selectExecution: () => {},
    resetForRecipe: () => {},
  };
  // One render: callbacks are the closures React would hand the buttons.
  const react = {
    useState: (initial: unknown) => [initial, () => {}],
    useCallback: (fn: unknown) => fn,
    useEffect: () => {},
    useRef: (initial: unknown) => ({ current: initial }),
  };
  const { useRecipeExecutions } = loadWithStubs<{
    useRecipeExecutions: (params: Record<string, unknown>) => Hook;
  }>(
    new URL(
      "../src/features/recipe-studio/hooks/use-recipe-executions.ts",
      import.meta.url,
    ),
    {
      react,
      "zustand/react/shallow": { useShallow: (fn: unknown) => fn },
      "../stores/recipe-executions": {
        useRecipeExecutionsStore: (select: (s: typeof state) => unknown) =>
          select(state),
      },
      "../api": {
        validateRecipe: () => {
          calls.validate += 1;
          const next = deferred<unknown>();
          validations.push(next);
          return next.promise;
        },
        createRecipeJob: () => {
          calls.createJob += 1;
          if (holdJob) {
            const next = deferred<unknown>();
            jobs.push(next);
            return next.promise;
          }
          // biome-ignore lint/style/useNamingConvention: api schema
          return Promise.resolve({ job_id: "job-1" });
        },
        cancelRecipeJob: () => Promise.resolve(),
        getRecipeJobDataset: () => Promise.resolve({}),
      },
      "../executions/run-settings": {
        buildExecutionPayload: () => ({}),
        sanitizeExecutionRows: (rows: number) => rows,
      },
      "../executions/runtime": {
        createBaseExecutionRecord: () => ({ id: "exec-1" }),
      },
      "../executions/tracker": {
        trackRecipeExecution: () =>
          Promise.resolve({ success: true, terminal: true }),
      },
      "../executions/execution-helpers": {
        executionLabel: () => "Preview",
        toErrorMessage: (_e: unknown, fallback: string) => fallback,
        withExecutionDefaults: (record: unknown) => record,
      },
      "../data/executions-db": {
        saveRecipeExecution: () => Promise.resolve(),
      },
      "../executions/hydration": {},
      "@/shared/toast": { toastError: () => {} },
      "@/lib/toast": {},
      "@/config/env": {},
      "@/features/chat": {},
      "@/features/chat/presets/preset-policy": {},
      "@/features/hub/lib/model-identity": {},
      "@/features/model-picker": {},
    },
  );
  const hook = useRecipeExecutions({
    recipeId: "recipe-1",
    currentSignature: "sig",
    payloadResult: { errors: [], payload: { recipe: {}, run: {}, ui: {} } },
  });
  return { hook, calls, validations, jobs };
}

// Both clicks reach /validate synchronously if nothing guards re-entry, so count before awaiting.
test("a second Run click while validation is pending starts nothing", { timeout: 10_000 }, async () => {
  const { hook, calls, validations } = mountExecutionsHook();
  const first = hook.runPreview();
  const second = hook.runPreview();
  assert.equal(calls.validate, 1, "the second click must not validate again");
  assert.equal(await second, false);

  validations[0].resolve({ valid: true, errors: [] });
  assert.equal(await first, true);
  assert.equal(calls.createJob, 1);
});

// Easy's Run reads only fullLoading and the job lock, and a preview has neither until /jobs answers.
test("Run stays refused while the job request is in flight, from any Run button", { timeout: 10_000 }, async () => {
  const { hook, calls, validations, jobs } = mountExecutionsHook({ holdJob: true });
  const first = hook.runPreview();
  validations[0].resolve({ valid: true, errors: [] });
  for (let i = 0; i < 100 && calls.createJob === 0; i++) {
    await new Promise((r) => setImmediate(r));
  }
  assert.equal(calls.createJob, 1);

  const second = hook.runFull();
  assert.equal(calls.validate, 1, "a second run must not start while /jobs is pending");
  assert.equal(await second, false);

  // biome-ignore lint/style/useNamingConvention: api schema
  jobs[0].resolve({ job_id: "job-1" });
  assert.equal(await first, true);
  const next = hook.runPreview();
  assert.equal(calls.validate, 2, "a run after the first settles must go through");
  validations[1].resolve({ valid: false, errors: [] });
  await next;
});

test("the Run buttons read busy from the first click, and free up after a failed check", { timeout: 10_000 }, async () => {
  const { hook, calls, validations } = mountExecutionsHook();
  const first = hook.runPreview();
  assert.deepEqual(calls.loading, [true], "busy must be set before /validate answers");

  validations[0].resolve({ valid: false, errors: [{ message: "bad" }] });
  assert.equal(await first, false);
  assert.deepEqual(calls.loading, [true, false]);

  const retry = hook.runPreview();
  assert.equal(calls.validate, 2, "a run after a failed check must go through");
  validations[1].resolve({ valid: false, errors: [] });
  await retry;
});

test("the dialog's Start button is disabled while Check runs", () => {
  const dialog = readSrc("features/recipe-studio/dialogs/preview-dialog.tsx");
  const start = dialog.slice(
    dialog.indexOf("onClick={onRun}"),
    dialog.indexOf("Start ${kindLabel"),
  );
  assert.match(start, /disabled=\{[^}]*validateLoading[^}]*\}/);
});

test("the Runs view can shrink below its table's width", () => {
  // A flex child's min-width defaults to its content; without min-w-0 the row is as wide as
  // the data table, overflows the page, and the page's overflow-hidden clips Download.
  const view = readSrc(
    "features/recipe-studio/components/executions/executions-view.tsx",
  );
  const root = /return \([\s\S]*?<div className="([^"]+)">\s*<ExecutionSidebar/.exec(
    view,
  );
  assert.ok(root, "the root row of ExecutionsView");
  const classes = root[1].split(/\s+/);
  assert.ok(classes.includes("min-w-0"), `root classes: ${root[1]}`);
  assert.ok(classes.includes("flex-1"), `root classes: ${root[1]}`);
});

function textOf(node: unknown): string {
  if (node === null || node === undefined || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  const props = (node as { props?: Record<string, unknown> }).props ?? {};
  return textOf(props.children);
}

function findAll(
  node: unknown,
  match: (props: Record<string, unknown>) => boolean,
  out: Record<string, unknown>[] = [],
): Record<string, unknown>[] {
  if (Array.isArray(node)) {
    for (const child of node) findAll(child, match, out);
    return out;
  }
  if (!node || typeof node !== "object") return out;
  const props = (node as { props?: Record<string, unknown> }).props ?? {};
  if (match(props)) out.push(props);
  findAll(props.children, match, out);
  return out;
}

function renderIsland(execution: Record<string, unknown>): unknown {
  const { ExecutionProgressIsland } = loadWithStubs<{
    ExecutionProgressIsland: (props: Record<string, unknown>) => unknown;
  }>(
    new URL(
      "../src/features/recipe-studio/components/runtime/execution-progress-island.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/progress": { Progress: "Progress" },
      "@/components/ui/spinner": { Spinner: "Spinner" },
      "@/lib/chevron-icons": {},
      "@/lib/utils": {
        cn: (...parts: unknown[]) => parts.filter(Boolean).join(" "),
      },
      "@hugeicons/core-free-icons": {},
      "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
      "../../executions/execution-helpers": { isExecutionInProgress },
      "../executions/executions-view-helpers": helpers,
    },
  );
  return ExecutionProgressIsland({
    execution,
    currentColumnIcon: {},
    minimized: false,
    onMinimizedChange: () => {},
    onViewExecutions: () => {},
  });
}

test("before the first rows the island says what the run is doing, not a row of dashes", () => {
  const island = renderIsland({
    status: "active",
    stage: "healthcheck",
    progress: null,
    // biome-ignore lint/style/useNamingConvention: api schema
    current_column: null,
  });
  const text = textOf(island);
  assert.match(text, /Preparing: Checking the models/);
  const grids = findAll(
    island,
    (props) => typeof props.className === "string" && props.className.includes("grid-cols-2"),
  );
  assert.equal(grids.length, 1);
  assert.match(String(grids[0].className), /\bhidden\b/);

  const running = textOf(
    renderIsland({
      status: "active",
      stage: "generating",
      progress: { done: 3, total: 10, percent: 30 },
      // biome-ignore lint/style/useNamingConvention: api schema
      current_column: "answer",
    }),
  );
  assert.doesNotMatch(running, /Preparing/);
  assert.match(running, /Column: answer/);
});

test("stage names map to plain words, unknown ones to a generic start", () => {
  assert.equal(helpers.preparingStageLabel("source"), "Reading the seed data");
  assert.equal(helpers.preparingStageLabel("column_config"), "Setting up columns");
  assert.equal(helpers.preparingStageLabel(null), "Starting the run");
  assert.equal(helpers.preparingStageLabel("something-new"), "Starting the run");
});

test("a fit while the island shows keeps its height clear at the top", () => {
  assert.equal(fitView.fitViewPadding(false), fitView.FIT_VIEW_PADDING);
  const padding = fitView.fitViewPadding(true) as Record<string, unknown>;
  // The island measured 141 px tall in the audit's 1440x900 run.
  assert.ok(Number.parseFloat(String(padding.top)) >= 141, String(padding.top));
  assert.match(String(padding.top), /px$/);

  const page = readSrc("features/recipe-studio/recipe-studio-page.tsx");
  assert.match(page, /padding: fitViewPadding\(runIslandVisibleRef\.current\)/);
});

// Center view and Auto layout fit through their own controls, not the page's scheduled fit.
test("Center view and Auto layout keep the island's clearance while it shows", () => {
  const fits: Array<Record<string, unknown>> = [];
  const stubs = {
    "react/jsx-runtime": stubJsxRuntime(),
    react: { useCallback: (fn: unknown) => fn },
    "@xyflow/react": {
      Panel: "Panel",
      useReactFlow: () => ({
        fitView: (options: Record<string, unknown>) => fits.push(options),
        getNodes: () => [{ id: "a", type: "builder", data: {} }],
        zoomIn: () => {},
        zoomOut: () => {},
      }),
      useUpdateNodeInternals: () => () => {},
    },
    "@/components/ui/button": { Button: "Button" },
    "lucide-react": new Proxy({}, { get: (_t, key) => String(key) }),
    "@hugeicons/core-free-icons": { PlusSignIcon: "PlusSignIcon" },
    "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
    "../../utils/graph/fit-view": fitView,
    "../recipe-floating-icon-button-class": {
      // biome-ignore lint/style/useNamingConvention: constant name
      RECIPE_FLOATING_ICON_BUTTON_CLASS: "",
    },
  };
  const { ViewportControls } = loadWithStubs<{
    ViewportControls: (props: Record<string, unknown>) => unknown;
  }>(
    new URL(
      "../src/features/recipe-studio/components/controls/viewport-controls.tsx",
      import.meta.url,
    ),
    stubs,
  );
  const { LayoutControls } = loadWithStubs<{
    LayoutControls: (props: Record<string, unknown>) => unknown;
  }>(
    new URL(
      "../src/features/recipe-studio/components/controls/layout-controls.tsx",
      import.meta.url,
    ),
    stubs,
  );
  const previousRaf = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
    cb(0);
    return 0;
  }) as typeof requestAnimationFrame;
  try {
    for (const runIslandVisible of [true, false]) {
      fits.length = 0;
      const viewport = ViewportControls({
        interactive: true,
        onToggleInteractive: () => {},
        maximized: false,
        onToggleMaximize: () => {},
        runIslandVisible,
      });
      const center = findAll(viewport, (props) => props["aria-label"] === "Center view");
      (center[0].onClick as () => void)();
      const layout = LayoutControls({
        direction: "LR",
        onLayout: () => {},
        onToggleDirection: () => {},
        runIslandVisible,
      });
      for (const button of findAll(layout, (props) => typeof props.onClick === "function")) {
        (button.onClick as () => void)();
      }
      assert.equal(fits.length, 3);
      for (const fit of fits) {
        assert.deepEqual(fit.padding, fitView.fitViewPadding(runIslandVisible));
      }
    }
  } finally {
    globalThis.requestAnimationFrame = previousRaf;
  }
});

test("the run payload no longer sends output_formats", () => {
  const built = buildRecipePayload({}, [], [], [], "LR", {});
  assert.equal("output_formats" in built.payload.run, false);
  assert.equal("output_formats" in createEmptyRecipePayload().run, false);
});
