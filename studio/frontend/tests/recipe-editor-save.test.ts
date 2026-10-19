// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  loadWithStubs,
  type StubElement,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

type Persisted = { name: string; payload: unknown; overwrite?: boolean };
type ToastOptions = {
  action?: { onClick: () => void };
  cancel?: { onClick: () => void };
};

// Just enough React to drive one hook: renders are explicit, effects run on commit().
function hookRuntime() {
  const slots: unknown[] = [];
  const effects: { deps?: unknown[]; cleanup?: (() => void) | void }[] = [];
  const pending: (() => void)[] = [];
  let cursor = 0;
  const react = {
    useState: (initial: unknown) => {
      const i = cursor++;
      if (!(i in slots)) slots[i] = initial;
      return [
        slots[i],
        (value: unknown) => {
          slots[i] = value;
        },
      ];
    },
    useRef: (initial: unknown) => {
      const i = cursor++;
      if (!(i in slots)) slots[i] = { current: initial };
      return slots[i];
    },
    useMemo: (fn: () => unknown) => {
      cursor++;
      return fn();
    },
    useCallback: (fn: unknown) => {
      cursor++;
      return fn;
    },
    useEffect: (fn: () => (() => void) | void, deps?: unknown[]) => {
      const i = cursor++;
      const prev = effects[i];
      if (
        !prev ||
        !deps ||
        !prev.deps ||
        deps.some((value, j) => value !== prev.deps?.[j])
      ) {
        pending.push(() => {
          if (typeof prev?.cleanup === "function") prev.cleanup();
          effects[i] = { deps, cleanup: fn() };
        });
      }
    },
  };
  return {
    react,
    begin() {
      cursor = 0;
    },
    commit() {
      while (pending.length) pending.shift()?.();
    },
    unmount() {
      for (const effect of effects) {
        if (typeof effect?.cleanup === "function") effect.cleanup();
      }
    },
  };
}

function fakeWindow() {
  const timers = new Map<number, () => void>();
  const listeners = new Map<string, Set<(event: unknown) => void>>();
  let next = 0;
  globalThis.window = {
    setTimeout: (fn: () => void) => {
      next += 1;
      timers.set(next, fn);
      return next;
    },
    clearTimeout: (id: number) => void timers.delete(id),
    addEventListener: (type: string, fn: (event: unknown) => void) => {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)?.add(fn);
    },
    removeEventListener: (type: string, fn: (event: unknown) => void) =>
      void listeners.get(type)?.delete(fn),
  } as unknown as Window & typeof globalThis;
  return {
    runTimers() {
      const due = [...timers.values()];
      timers.clear();
      for (const fn of due) fn();
    },
    listeners: (type: string) => [...(listeners.get(type) ?? [])],
  };
}

const settle = async () => {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
};

function mountEditor(persist: (input: Persisted) => Promise<{ updatedAt: number }>) {
  const runtime = hookRuntime();
  const win = fakeWindow();
  const toasts: { title: string; options: ToastOptions }[] = [];
  const saves: Persisted[] = [];
  const reloads: number[] = [];
  const quiet = (..._args: unknown[]) => undefined;
  const { useRecipePersistence } = loadWithStubs<{
    useRecipePersistence: (params: Record<string, unknown>) => {
      persistRecipe: () => Promise<void>;
    };
  }>(
    new URL(
      "../src/features/recipe-studio/hooks/use-recipe-persistence.ts",
      import.meta.url,
    ),
    {
      react: runtime.react,
      "@/lib/copy-to-clipboard": { copyToClipboard: async () => true },
      "@/lib/toast": {
        toast: {
          error: (title: string, options: ToastOptions) =>
            toasts.push({ title, options }),
          dismiss: quiet,
        },
      },
      "@/shared/toast": {
        toastError: (title: string) => toasts.push({ title, options: {} }),
        toastSuccess: quiet,
      },
      "@/utils": {
        normalizeNonEmptyName: (value: string, fallback = "Unnamed") =>
          value.trim() || fallback,
      },
      "../api": { removeUnstructuredBlock: async () => undefined },
      "../executions/execution-helpers": {
        buildSignature: (name: string, payload: unknown) =>
          JSON.stringify([name, payload]),
        formatSavedLabel: () => "",
      },
      "../stores/recipe-studio": {
        useRecipeStudioStore: {
          getState: () => ({ pendingUploadCleanups: [] }),
          setState: quiet,
        },
      },
      "../utils/import": {
        importRecipePayload: () => ({ snapshot: {}, errors: [] }),
      },
    },
  );

  const initialPayload = { v: "saved" };
  let payload: unknown = initialPayload;
  const params: Record<string, unknown> = {
    recipeId: "r1",
    initialRecipeName: "Recipe",
    initialPayload,
    initialSavedAt: 1,
    onPersistRecipe: (input: Persisted) => {
      saves.push(input);
      return persist(input);
    },
    onReloadRecipe: () => reloads.push(1),
    resetRecipe: quiet,
    loadRecipe: quiet,
    getCurrentPayloadFromStore: () => params.initialPayload,
  };
  const render = () => {
    runtime.begin();
    const result = useRecipePersistence({ ...params, payloadResult: { payload } });
    runtime.commit();
    return result;
  };
  render();
  render();
  return {
    win,
    saves,
    toasts,
    reloads,
    render,
    edit(value: string) {
      payload = { v: value };
      return render();
    },
    // What Reload theirs does: the page hands the editor the server's record.
    reload(value: string, savedAt: number) {
      payload = { v: value };
      params.initialPayload = payload;
      params.initialSavedAt = savedAt;
      render();
      render();
    },
    unmount: runtime.unmount,
  };
}

test("leaving the editor inside the autosave debounce still saves the last edit", async () => {
  const editor = mountEditor(async () => ({ updatedAt: 2 }));
  editor.edit("typed just before leaving");
  editor.unmount();
  await settle();
  assert.deepEqual(
    editor.saves.map((save) => save.payload),
    [{ v: "typed just before leaving" }],
  );
});

test("a reload with unsaved edits asks before leaving, and stops asking once saved", async () => {
  const editor = mountEditor(async () => ({ updatedAt: 2 }));
  assert.equal(editor.win.listeners("beforeunload").length, 0);
  editor.edit("unsaved");
  const [warn] = editor.win.listeners("beforeunload");
  assert.ok(warn, "a dirty editor must guard beforeunload");
  let prevented = false;
  warn({ preventDefault: () => (prevented = true), returnValue: undefined });
  assert.equal(prevented, true);

  editor.win.runTimers();
  await settle();
  editor.render();
  editor.render();
  assert.equal(editor.saves.length, 1);
  assert.equal(editor.win.listeners("beforeunload").length, 0);
});

test("a save conflict pauses autosave and offers reload or overwrite once", async () => {
  const conflict = Object.assign(new Error("changed in another window"), {
    status: 409,
  });
  let refuse = true;
  const editor = mountEditor(async () => {
    if (refuse) throw conflict;
    return { updatedAt: 3 };
  });
  const error = console.error;
  console.error = () => undefined;
  try {
    editor.edit("first");
    editor.win.runTimers();
    await settle();
    editor.render();
    editor.edit("second");
    editor.edit("third");
    editor.win.runTimers();
    await settle();
    editor.render();
  } finally {
    console.error = error;
  }

  assert.equal(editor.saves.length, 1, "no autosave may retry into the conflict");
  assert.equal(editor.toasts.length, 1);
  const { options } = editor.toasts[0];
  assert.ok(options.action && options.cancel, "the toast must offer both choices");

  options.cancel?.onClick();
  assert.equal(editor.reloads.length, 1);

  refuse = false;
  options.action?.onClick();
  await settle();
  assert.deepEqual(editor.saves.at(-1), {
    id: "r1",
    name: "Recipe",
    payload: { v: "third" },
    overwrite: true,
  });
});

test("a 422 on save names the field instead of a bare status", async () => {
  const { formatFastApiDetail } = loadWithStubs<{
    formatFastApiDetail: (detail: unknown) => string | null;
  }>(new URL("../src/lib/format-fastapi-error.ts", import.meta.url), {});
  const { recipeRequest } = loadWithStubs<{
    recipeRequest: (path: string, init?: RequestInit) => Promise<unknown>;
  }>(new URL("../src/features/data-recipes/data/recipes-api.ts", import.meta.url), {
    "@/features/auth": {
      authFetch: async () =>
        new Response(
          JSON.stringify({
            detail: [
              {
                loc: ["body", "name"],
                msg: "String should have at most 10000 characters",
              },
            ],
          }),
          { status: 422 },
        ),
      getAuthSessionEpoch: () => 1,
    },
    "@/lib/format-fastapi-error": { formatFastApiDetail },
  });
  await assert.rejects(recipeRequest("/r1", { method: "PUT", body: "{}" }), {
    message: "name: String should have at most 10000 characters",
  });
});

function mountHeader() {
  const slots: unknown[] = [];
  let cursor = 0;
  const renames: string[] = [];
  const Input = Symbol("Input");
  const { RecipeStudioHeader } = loadWithStubs<{
    RecipeStudioHeader: (props: Record<string, unknown>) => StubElement;
  }>(
    new URL(
      "../src/features/recipe-studio/components/recipe-studio-header.tsx",
      import.meta.url,
    ),
    {
      react: {
        useState: (initial: unknown) => {
          const i = cursor++;
          if (!(i in slots)) slots[i] = initial;
          return [slots[i], (value: unknown) => (slots[i] = value)];
        },
        useRef: (initial: unknown) => {
          const i = cursor++;
          if (!(i in slots)) slots[i] = { current: initial };
          return slots[i];
        },
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/ui/badge": { Badge: "Badge" },
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/input": { Input },
      "@/components/ui/popover": {
        Popover: "Popover",
        PopoverContent: "PopoverContent",
        PopoverTrigger: "PopoverTrigger",
      },
      "@/components/ui/tabs": {
        Tabs: "Tabs",
        TabsList: "TabsList",
        TabsTrigger: "TabsTrigger",
      },
      "@hugeicons/core-free-icons": {},
      "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
      "@/utils": {
        normalizeNonEmptyName: (value: string, fallback = "Unnamed") =>
          value.trim() || fallback,
      },
      "../utils/ui-tones": {},
    },
  );
  let name = "Original";
  const find = (node: unknown, match: (el: StubElement) => boolean): StubElement | null => {
    if (!node || typeof node !== "object") return null;
    if (Array.isArray(node)) {
      for (const child of node) {
        const hit = find(child, match);
        if (hit) return hit;
      }
      return null;
    }
    const el = node as StubElement;
    if (el.props && match(el)) return el;
    return find(el.props?.children, match);
  };
  const render = () => {
    cursor = 0;
    return RecipeStudioHeader({
      activeView: "editor",
      saveLoading: false,
      saveTone: "success",
      savedAtLabel: "",
      workflowName: name,
      onWorkflowNameChange: (value: string) => {
        renames.push(value);
        name = value;
      },
      onViewChange: () => undefined,
      onSaveRecipe: () => undefined,
    });
  };
  const input = () => find(render(), (el) => el.type === Input);
  const open = () => {
    const button = find(
      render(),
      (el) => String(el.props["aria-label"] ?? "").startsWith("Edit recipe name"),
    );
    (button?.props.onClick as () => void)();
  };
  const type = (value: string) =>
    (input()?.props.onChange as (e: unknown) => void)({ target: { value } });
  const key = (k: string) =>
    (input()?.props.onKeyDown as (e: unknown) => void)({ key: k });
  return {
    renames,
    name: () => name,
    open,
    input,
    type,
    key,
    blur: () => (input()?.props.onBlur as () => void)(),
  };
}

test("Esc in the recipe name editor keeps the old name", () => {
  const header = mountHeader();
  header.open();
  header.type("G7 escape test");
  const blur = header.input()?.props.onBlur as () => void;
  header.key("Escape");
  blur();
  assert.equal(header.name(), "Original");
  assert.equal(header.input(), null);
});

test("a typed name reaches the recipe before any blur, so leaving the page keeps it", () => {
  const header = mountHeader();
  header.open();
  header.type("Renamed then left");
  assert.equal(header.name(), "Renamed then left");
});

test("an emptied recipe name falls back to Unnamed, and the input is capped", () => {
  const header = mountHeader();
  header.open();
  assert.equal(header.input()?.props.maxLength, 10_000);
  header.type("   ");
  header.key("Enter");
  assert.equal(header.name(), "Unnamed");
});

test("after Reload theirs, an edit matching the discarded copy still autosaves", async () => {
  const conflict = Object.assign(new Error("changed in another window"), {
    status: 409,
  });
  let refuse = true;
  const editor = mountEditor(async () => {
    if (refuse) throw conflict;
    return { updatedAt: 9 };
  });
  const error = console.error;
  console.error = () => undefined;
  try {
    editor.edit("mine");
    editor.win.runTimers();
    await settle();
    editor.render();
    editor.toasts[0].options.cancel?.onClick();
    refuse = false;
    editor.reload("theirs", 5);
    editor.edit("mine");
    editor.win.runTimers();
    await settle();
  } finally {
    console.error = error;
  }
  assert.deepEqual(
    editor.saves.map((save) => save.payload),
    [{ v: "mine" }, { v: "mine" }],
  );
});
