// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import type { StreamState } from "../lib/stream-state";

const markdownClassName =
  "max-h-none border-0 bg-transparent p-0 text-sm leading-relaxed";

export function StreamView({
  state,
  className,
}: {
  state: StreamState;
  className?: string;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-2", className)}>
      {state.blocks.map((block, i) =>
        block.kind === "text" ? (
          // Blocks are append-only, so index keys are stable here.
          <MarkdownPreview
            key={`b-${i}`}
            markdown={block.text}
            className={markdownClassName}
          />
        ) : block.kind === "tool" ? (
          <div
            key={`b-${i}`}
            className="flex min-w-0 items-center gap-2 rounded-md border border-border/60 bg-muted/30 px-2.5 py-1.5 text-xs text-muted-foreground"
          >
            <span className="font-medium text-foreground">{block.name}</span>
            {block.summary ? (
              <span className="truncate font-mono">{block.summary}</span>
            ) : null}
          </div>
        ) : (
          <div
            key={`b-${i}`}
            className="whitespace-pre-wrap rounded-md border border-red-500/30 bg-red-500/10 px-2.5 py-1.5 text-xs text-red-600 dark:text-red-400"
          >
            {block.text}
          </div>
        ),
      )}
      {state.liveText ? (
        <MarkdownPreview markdown={state.liveText} className={markdownClassName} />
      ) : null}
      {state.running ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Spinner className="size-3.5" />
          <span>Claude is working...</span>
        </div>
      ) : null}
      {!state.running && state.costUsd != null ? (
        <div className="text-[11px] text-muted-foreground">
          Cost: ${state.costUsd.toFixed(4)}
          {state.model ? ` · ${state.model}` : ""}
        </div>
      ) : null}
    </div>
  );
}
