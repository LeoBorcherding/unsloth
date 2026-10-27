// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs, stubJsxRuntime, type StubElement } from "./helpers/module-stubs.ts";

function elements(node: unknown): StubElement[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!node || typeof node !== "object" || !("props" in node)) return [];
  const element = node as StubElement;
  return [element, ...elements(element.props.children)];
}
function text(node: unknown): string {
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(text).join(" ");
  if (node && typeof node === "object" && "props" in node) return text((node as StubElement).props.children);
  return "";
}
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

function mount(reset: (password: string) => Promise<boolean>) {
  const state: unknown[] = [];
  let cursor = 0;
  const done: boolean[] = [];
  const api = loadWithStubs<{ DesktopOwnerReset: (props: { onDone: (replaced: boolean) => void }) => StubElement }>(
    new URL("../src/features/auth/components/desktop-owner-reset.tsx", import.meta.url),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState: (initial: unknown) => {
          const index = cursor++;
          if (!(index in state)) state[index] = initial;
          return [state[index], (value: unknown) => { state[index] = value; }];
        },
      },
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/input": { Input: "Input" },
      "@/components/ui/label": { Label: "Label" },
      "../tauri-auto-auth": { resetDesktopOwnerPassword: reset },
    },
  );
  const render = () => {
    cursor = 0;
    return api.DesktopOwnerReset({ onDone: (replaced) => done.push(replaced) });
  };
  const type = (id: string, value: string) => {
    const input = elements(render()).find((element) => element.props.id === id);
    (input?.props.onChange as (event: unknown) => void)({ target: { value } });
  };
  const submit = async () => {
    const form = elements(render()).find((element) => element.type === "form");
    (form?.props.onSubmit as (event: unknown) => void)({ preventDefault() {} });
    await tick();
  };
  return { render, type, submit, done };
}

test("the reset form stays closed until asked, then needs a matching password of the usual length", async () => {
  const calls: string[] = [];
  const ui = mount(async (password) => {
    calls.push(password);
    return false;
  });
  const link = ui.render();
  assert.equal(text(link).trim(), "Forgot password?");
  (link.props.onClick as () => void)();
  assert.match(text(ui.render()), /Other accounts: ask the owner for a new setup code/);
  for (const [next, confirm] of [["short", "short"], ["has spaces ok", "has spaces ok"], ["long-enough", "different!"]]) {
    ui.type("owner-reset-password", next);
    ui.type("owner-reset-confirm", confirm);
    await ui.submit();
  }
  assert.deepEqual(calls, []);
  ui.type("owner-reset-password", "long-enough");
  ui.type("owner-reset-confirm", "long-enough");
  await ui.submit();
  assert.deepEqual(calls, ["long-enough"]);
  assert.deepEqual(ui.done, [false]);
});

test("a refused reset shows the reason and keeps the form open", async () => {
  const ui = mount(async () => {
    throw new Error("Desktop authentication failed");
  });
  (ui.render().props.onClick as () => void)();
  ui.type("owner-reset-password", "long-enough");
  ui.type("owner-reset-confirm", "long-enough");
  await ui.submit();
  assert.deepEqual(ui.done, []);
  assert.match(text(ui.render()), /Desktop authentication failed/);
});
