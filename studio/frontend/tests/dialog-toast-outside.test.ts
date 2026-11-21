// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

type OutsideEvent = {
  target: { closest: (selector: string) => unknown };
  defaultPrevented: boolean;
  preventDefault: () => void;
};

function findByType(node: unknown, type: string): StubElement | null {
  if (Array.isArray(node)) {
    for (const child of node) {
      const hit = findByType(child, type);
      if (hit) return hit;
    }
    return null;
  }
  if (!node || typeof node !== "object") return null;
  const element = node as StubElement;
  if (element.type === type) return element;
  return findByType((element.props ?? {}).children, type);
}

function outsideEvent(ancestor: string | null): OutsideEvent {
  return {
    target: {
      closest: (selector) =>
        ancestor && selector.split(",").some((s) => s.trim() === ancestor) ? {} : null,
    },
    defaultPrevented: false,
    preventDefault() {
      this.defaultPrevented = true;
    },
  };
}

test("an interaction on a toast or a corner card does not dismiss the dialog under it", () => {
  const { DialogContent } = loadWithStubs<{
    DialogContent: (props: {
      onInteractOutside: (event: OutsideEvent) => void;
    }) => StubElement;
  }>(new URL("../src/components/ui/dialog.tsx", import.meta.url), {
    "react/jsx-runtime": stubJsxRuntime(),
    react: {
      createContext: () => ({ Provider: "Provider" }),
      useContext: () => null,
    },
    "radix-ui": {
      Dialog: new Proxy({}, { get: (_, part) => `Dialog.${String(part)}` }),
    },
    "@/components/app-readiness": { AppPortalGate: "AppPortalGate" },
    "@/components/ui/button": { Button: "Button" },
    "@/components/ui/dropdown-glow": {
      DropdownGlowContext: { Provider: "Provider" },
      dialogDropdownGlow: () => null,
    },
    "@/lib/surface-registry": {
      MODAL_SURFACE: "modal",
      useMergedRefs: () => undefined,
      useSurfaceRef: () => undefined,
    },
    "@/lib/utils": { cn: (...classes: unknown[]) => classes.join(" ") },
    "@hugeicons/core-free-icons": { Cancel01Icon: {} },
    "@/lib/folder-icons": { Cancel01Icon: {} },
    "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
  });

  const seen: OutsideEvent[] = [];
  const content = findByType(
    DialogContent({ onInteractOutside: (event) => seen.push(event) }),
    "Dialog.Content",
  );
  assert.ok(content);
  const onInteractOutside = content.props.onInteractOutside as (
    event: OutsideEvent,
  ) => void;

  const onToast = outsideEvent("[data-sonner-toaster]");
  const onRail = outsideEvent("[data-overlay-rail]");
  const onPanel = outsideEvent("[data-floating-panel]");
  const elsewhere = outsideEvent(null);
  onInteractOutside(onToast);
  onInteractOutside(onRail);
  onInteractOutside(onPanel);
  onInteractOutside(elsewhere);

  assert.equal(onToast.defaultPrevented, true, "a toast click closed the dialog");
  assert.equal(onRail.defaultPrevented, true, "a Downloads card click closed the dialog");
  assert.equal(onPanel.defaultPrevented, true, "grabbing the Live monitor closed the dialog");
  assert.equal(elsewhere.defaultPrevented, false, "a real outside click must still dismiss");
  assert.deepEqual(seen, [onToast, onRail, onPanel, elsewhere], "the caller's handler was dropped");
});

test("an interaction on a toast or a corner card does not dismiss the sheet under it", () => {
  const { SheetContent } = loadWithStubs<{
    SheetContent: (props: {
      onInteractOutside: (event: OutsideEvent) => void;
      showCloseButton: boolean;
    }) => StubElement;
  }>(new URL("../src/components/ui/sheet.tsx", import.meta.url), {
    "react/jsx-runtime": stubJsxRuntime(),
    react: { useState: () => [null, () => {}] },
    "radix-ui": {
      Dialog: new Proxy({}, { get: (_, part) => `Dialog.${String(part)}` }),
    },
    "@/components/app-readiness": { AppPortalGate: "AppPortalGate" },
    "@/components/ui/button": { Button: "Button" },
    "@/components/ui/dropdown-glow": {
      DropdownGlowContext: { Provider: "Provider" },
      dialogDropdownGlow: () => null,
    },
    "@/components/ui/dialog": {
      DialogPortalContainerContext: { Provider: "Provider" },
    },
    "@/lib/surface-registry": {
      MODAL_SURFACE: "modal",
      useMergedRefs: () => undefined,
      useSurfaceRef: () => undefined,
    },
    "@/lib/utils": { cn: (...classes: unknown[]) => classes.join(" ") },
    "@hugeicons/core-free-icons": { Cancel01Icon: {} },
    "@/lib/folder-icons": { Cancel01Icon: {} },
    "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
  });

  const seen: OutsideEvent[] = [];
  const content = findByType(
    SheetContent({
      onInteractOutside: (event) => seen.push(event),
      showCloseButton: false,
    }),
    "Dialog.Content",
  );
  assert.ok(content);
  const onInteractOutside = content.props.onInteractOutside as (
    event: OutsideEvent,
  ) => void;

  const onToast = outsideEvent("[data-sonner-toaster]");
  const onRail = outsideEvent("[data-overlay-rail]");
  const onPanel = outsideEvent("[data-floating-panel]");
  const elsewhere = outsideEvent(null);
  onInteractOutside(onToast);
  onInteractOutside(onRail);
  onInteractOutside(onPanel);
  onInteractOutside(elsewhere);

  assert.equal(onToast.defaultPrevented, true, "a toast click closed the sheet");
  assert.equal(onRail.defaultPrevented, true, "a Downloads card click closed the sheet");
  assert.equal(onPanel.defaultPrevented, true, "grabbing the Live monitor closed the sheet");
  assert.equal(elsewhere.defaultPrevented, false, "a real outside click must still dismiss");
  assert.deepEqual(seen, [onToast, onRail, onPanel, elsewhere], "the caller's handler was dropped");
});

test("both floating panels stay live over a modal and carry the marker it exempts", () => {
  for (const path of [
    "../src/components/floating-monitor.tsx",
    "../src/features/api-monitor/api-monitor-overlay.tsx",
  ]) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.match(source, /<motion\.div[\s\S]*?ref=\{monitorRef\}[\s\S]*?data-floating-panel=""/, path);
    // index.css forces pointer-events-auto to none under a modal lock unless it opts out.
    assert.match(source, /pointer-events-auto modal-pe-live /, path);
  }
});
