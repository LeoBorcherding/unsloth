// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

type Call = { path: string; method: string; body: Record<string, unknown> | null };
type Recipe = { id: string; name: string; updatedAt: number; createdAt: number };

class RecipeApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

// A react whose state lives in one slot list, so a test can render a hook, run its effects and render again.
function fakeReact() {
  const slots: unknown[] = [];
  let effects: (() => unknown)[] = [];
  let index = 0;
  const sameDeps = (a: unknown, b: unknown[]) =>
    Array.isArray(a) && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  const react = {
    useState: (init: unknown) => {
      const at = index++;
      if (!(at in slots)) slots[at] = typeof init === "function" ? init() : init;
      return [
        slots[at],
        (next: unknown) => {
          slots[at] = typeof next === "function" ? next(slots[at]) : next;
        },
      ];
    },
    useEffect: (fn: () => unknown, deps: unknown[]) => {
      const at = index++;
      if (sameDeps(slots[at], deps)) return;
      slots[at] = deps;
      effects.push(fn);
    },
    useCallback: (fn: unknown) => fn,
  };
  const render = <T>(hook: () => T): T => {
    index = 0;
    const result = hook();
    const pending = effects;
    effects = [];
    for (const effect of pending) effect();
    return result;
  };
  return { react, render };
}

function loadRecipesDb(respond: (call: Call) => unknown) {
  const calls: Call[] = [];
  const recipeRequest = async (path: string, init: RequestInit = {}) => {
    const call = {
      path,
      method: init.method ?? "GET",
      body: init.body ? JSON.parse(String(init.body)) : null,
    };
    calls.push(call);
    return respond(call);
  };
  const { react, render } = fakeReact();
  const db = loadWithStubs<Record<string, (...args: never[]) => unknown>>(
    new URL("../src/features/data-recipes/data/recipes-db.ts", import.meta.url),
    {
      "@/features/recipe-studio": {
        createEmptyRecipePayload: () => ({ recipe: { columns: [] }, ui: {} }),
      },
      "@/utils": {
        normalizeNonEmptyName: (name: string) => name.trim() || "Unnamed",
      },
      react,
      "./legacy-import": { importLegacyRecipes: () => Promise.resolve() },
      "./recipes-api": { recipeRequest, RecipeApiError },
    },
  );
  return { db, calls, render };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

test("a failed list fetch is an error with Retry, not an empty list", async () => {
  let fail = true;
  const { db, render } = loadRecipesDb((call) => {
    if (call.method === "GET" && call.path === "") {
      if (fail) throw new RecipeApiError("Request failed (500)", 500);
      return { recipes: [{ id: "a", name: "A", createdAt: 1, updatedAt: 2 }] };
    }
    return undefined;
  });
  const useRecipes = db.useRecipes as () => Record<string, unknown>;
  const error = console.error;
  console.error = () => undefined;
  try {
    render(useRecipes);
    await flush();
    const failed = render(useRecipes);
    assert.equal(failed.ready, false, "a failure is not a ready, empty list");
    assert.match(String(failed.error), /500/);
    assert.equal(typeof failed.retry, "function");

    fail = false;
    (failed.retry as () => void)();
    render(useRecipes);
    await flush();
    const recovered = render(useRecipes);
    assert.equal(recovered.ready, true);
    assert.equal(recovered.error, null);
    assert.equal((recovered.recipes as Recipe[]).length, 1);
  } finally {
    console.error = error;
  }
});

test("Start Empty saves nothing until the first edit", async () => {
  const { db, calls } = loadRecipesDb((call) =>
    call.method === "PUT"
      ? { ...call.body, updatedAt: 99 }
      : { recipes: [] },
  );
  const draft = (await db.createRecipeDraft()) as Recipe;
  assert.deepEqual(
    calls.filter((call) => call.method !== "GET"),
    [],
    "no write for an untouched draft",
  );
  assert.equal(
    (db.getCachedRecipe as (id: string) => Recipe | null)(draft.id)?.id,
    draft.id,
    "the editor can open it from the cache",
  );

  await (db.saveRecipe as (input: Record<string, unknown>) => Promise<Recipe>)({
    id: draft.id,
    name: "Unnamed",
    payload: {},
    baseUpdatedAt: draft.updatedAt,
  });
  const puts = calls.filter((call) => call.method === "PUT");
  assert.equal(puts.length, 1);
  assert.equal(
    puts[0].body?.baseUpdatedAt,
    undefined,
    "the first save creates the row, so it has no base version",
  );
});

test("a new recipe whose name is taken gets a numbered suffix", async () => {
  const { db, calls } = loadRecipesDb((call) =>
    call.method === "PUT"
      ? { ...call.body }
      : {
          recipes: [
            { id: "a", name: "Instruction from Answer", createdAt: 1, updatedAt: 2 },
          ],
        },
  );
  await (db.listRecipes as () => Promise<unknown>)();
  await (db.createRecipeFromLearningRecipe as (input: Record<string, unknown>) => Promise<unknown>)({
    templateId: "instruction-from-answer",
    templateTitle: "Instruction from Answer",
    payload: {},
  });
  const put = calls.find((call) => call.method === "PUT");
  assert.equal(put?.body?.name, "Instruction from Answer (2)");
});

function parseTsx(relative: string): ts.SourceFile {
  return ts.createSourceFile(
    relative,
    readFileSync(new URL(`../src/${relative}`, import.meta.url), "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
}

function tags(source: ts.SourceFile, name: string): ts.JsxOpeningLikeElement[] {
  const found: ts.JsxOpeningLikeElement[] = [];
  const visit = (node: ts.Node) => {
    const tag = openingTag(node);
    if (tag && tag.tagName.getText(source) === name) found.push(tag);
    ts.forEachChild(node, visit);
  };
  visit(source);
  return found;
}

function attr(source: ts.SourceFile, tag: ts.JsxOpeningLikeElement, name: string): string {
  for (const prop of tag.attributes.properties) {
    if (ts.isJsxAttribute(prop) && prop.name.getText(source) === name) {
      return prop.initializer?.getText(source) ?? "";
    }
  }
  return "";
}

const LIST_PAGE = "features/data-recipes/pages/data-recipes-page.tsx";

test("the trash button asks before deleting, and names which recipe", () => {
  const page = parseTsx(LIST_PAGE);
  const trash = tags(page, "Button").find((tag) =>
    attr(page, tag, "aria-label").includes("Delete"),
  );
  assert.ok(trash, "the row's trash button");
  assert.doesNotMatch(
    attr(page, trash, "onClick"),
    /deleteRecipe|handleDeleteRecipe/i,
    "a click alone doesn't delete",
  );
  assert.match(
    attr(page, trash, "aria-label"),
    /updatedAt/,
    "two recipes with one name still get different labels",
  );
  const confirm = tags(page, "AlertDialogAction");
  assert.equal(confirm.length, 1, "a confirm dialog");
  assert.match(attr(page, confirm[0], "onClick"), /handleDeleteRecipe/);
});

test("the learning recipes stay on the list page once a recipe exists", () => {
  const page = parseTsx(LIST_PAGE);
  // Empty state, the dialog, and now under the list.
  assert.equal(tags(page, "LearningRecipeCards").length, 3);
});

test("a recipe that fails to load shows an error with Retry, not Recipe not found", () => {
  const source = readFileSync(
    new URL(`../src/features/data-recipes/pages/edit-recipe-page.tsx`, import.meta.url),
    "utf8",
  );
  const catchBody = source.slice(source.indexOf(".catch("));
  assert.match(
    catchBody.slice(0, catchBody.indexOf("});")),
    /status: "error"/,
    "a thrown load is an error state",
  );
  assert.match(source, /onRetry=/);
});

test("Run and Check are disabled on an empty canvas", () => {
  const { RunValidateFloatingControls } = loadWithStubs<Record<string, (props: unknown) => StubElement>>(
    new URL(
      "../src/features/recipe-studio/components/controls/run-validate-floating-controls.tsx",
      import.meta.url,
    ),
    {
      "@hugeicons/core-free-icons": { CookBookIcon: "cook", TestTube01Icon: "tube" },
      "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
      "@/components/ui/button": { Button: "Button" },
      "react/jsx-runtime": stubJsxRuntime(),
    },
  );
  const tree = RunValidateFloatingControls({
    runBusy: false,
    runDialogKind: "preview",
    validateLoading: false,
    executionLocked: false,
    emptyRecipe: true,
    onOpenRunDialog: () => undefined,
    onValidate: () => undefined,
  });
  const buttons: StubElement[] = [];
  const walk = (node: unknown) => {
    if (Array.isArray(node)) return node.forEach(walk);
    if (!(node && typeof node === "object" && "props" in node)) return;
    const element = node as StubElement;
    if (element.type === "Button") buttons.push(element);
    walk(element.props.children);
  };
  walk(tree);
  assert.equal(buttons.length, 2);
  for (const button of buttons) {
    assert.equal(button.props.disabled, true);
  }
});
