// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  loadWithStubs,
  type StubElement,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

type PublishPayload = { repo_id: string; hf_token?: string | null };

function mountDialog(savedToken: string) {
  const slots: unknown[] = [];
  let cursor = 0;
  const sent: PublishPayload[] = [];
  const quiet = () => undefined;
  const { PublishExecutionDialog } = loadWithStubs<{
    PublishExecutionDialog: (props: Record<string, unknown>) => StubElement;
  }>(
    new URL(
      "../src/features/recipe-studio/components/executions/publish-execution-dialog.tsx",
      import.meta.url,
    ),
    {
      react: {
        useState: (initial: unknown) => {
          const i = cursor++;
          if (!(i in slots)) slots[i] = initial;
          return [slots[i], (value: unknown) => (slots[i] = value)];
        },
        useMemo: (fn: () => unknown) => {
          cursor++;
          return fn();
        },
        useEffect: () => {
          cursor++;
        },
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "@hugeicons/core-free-icons": {},
      "lucide-react": { ChevronRightIcon: "ChevronRightIcon" },
      "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/dialog": {
        Dialog: "Dialog",
        DialogContent: "DialogContent",
        DialogDescription: "DialogDescription",
        DialogFooter: "DialogFooter",
        DialogHeader: "DialogHeader",
        DialogTitle: "DialogTitle",
      },
      "@/components/ui/input": { Input: "Input" },
      "@/components/ui/switch": { Switch: "Switch" },
      "@/components/ui/textarea": { Textarea: "Textarea" },
      "@/features/hub": {
        useHfTokenStore: (select: (s: { token: string }) => unknown) =>
          select({ token: savedToken }),
        hfApiToken: (token: string | null | undefined) =>
          token?.startsWith("hf_") ? token : undefined,
      },
      "@/lib/chevron-icons": { ChevronRightStandardIcon: "ChevronRightStandardIcon" },
      "@/lib/copy-icon": { Copy01Icon: "Copy01Icon" },
      "@/lib/copy-to-clipboard": { copyToClipboard: async () => true },
      "@/shared/toast": { toastError: quiet, toastSuccess: quiet },
    },
  );
  const find = (
    node: unknown,
    match: (el: StubElement) => boolean,
  ): StubElement | null => {
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
    return PublishExecutionDialog({
      open: true,
      onOpenChange: quiet,
      execution: {
        jobId: "job-1",
        artifact_path: "/runs/a",
        run_name: "Run",
        rows: 3,
      },
      onPublish: async (payload: PublishPayload) => {
        sent.push(payload);
        return { url: "https://huggingface.co/datasets/org/ds" };
      },
    });
  };
  return {
    sent,
    async publishWith(fields: Record<string, string>) {
      for (const [id, value] of Object.entries(fields)) {
        const input = find(render(), (el) => el.props.id === id);
        (input?.props.onChange as (e: unknown) => void)({ target: { value } });
      }
      const button = find(
        render(),
        (el) =>
          el.type === "Button" &&
          JSON.stringify(el.props.children ?? "").includes("Publish to Hugging Face"),
      );
      (button?.props.onClick as () => void)();
      for (let i = 0; i < 5; i++) await Promise.resolve();
    },
    text: () => JSON.stringify(render()),
  };
}

test("an empty token field publishes with the token saved in Settings", async () => {
  const dialog = mountDialog("hf_saved");
  const before = dialog.text();
  await dialog.publishWith({ "publish-repo-id": "org/ds" });
  assert.equal(dialog.sent.length, 1);
  assert.equal(dialog.sent[0].hf_token, "hf_saved");
  assert.match(before, /token saved in Settings/);
});

test("a token typed into the dialog still wins over the saved one", async () => {
  const dialog = mountDialog("hf_saved");
  await dialog.publishWith({
    "publish-repo-id": "org/ds",
    "publish-hf-token": "hf_typed",
  });
  assert.equal(dialog.sent[0].hf_token, "hf_typed");
});

test("with nothing saved and nothing typed, no token is sent", async () => {
  const dialog = mountDialog("");
  await dialog.publishWith({ "publish-repo-id": "org/ds" });
  assert.equal(dialog.sent[0].hf_token, null);
});
