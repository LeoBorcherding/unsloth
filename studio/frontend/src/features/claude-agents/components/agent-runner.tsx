// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "@/lib/toast";
import { useEffect, useRef, useState } from "react";
import {
  cancelClaudeRun,
  extractPlaceholders,
  streamClaudeRun,
  type AgentScript,
} from "../api/claude-agents-api";
import {
  emptyStreamState,
  reduceStreamEvent,
  type StreamState,
} from "../lib/stream-state";
import { StreamView } from "./stream-view";

const FILE_LIKE_RE = /file|path|dir/i;

// Render with key={agent.id} so per-agent state resets on selection change.
export function AgentRunner({ agent }: { agent: AgentScript }) {
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [cwd, setCwd] = useState(agent.cwd);
  const [stream, setStream] = useState<StreamState>(emptyStreamState());
  const [running, setRunning] = useState(false);
  const runIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const placeholders = extractPlaceholders(agent.promptTemplate);

  // Abort any in-flight stream when the component unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  async function handleRun() {
    const missing = placeholders.filter((name) => !variables[name]?.trim());
    if (missing.length > 0) {
      toast.error(`Fill in: ${missing.map((m) => `{${m}}`).join(", ")}`);
      return;
    }
    setRunning(true);
    setStream({ ...emptyStreamState(), running: true });
    const abort = new AbortController();
    abortRef.current = abort;
    try {
      await streamClaudeRun(
        {
          agentId: agent.id,
          variables,
          cwd: cwd.trim() || undefined,
        },
        (event) => {
          if (event.type === "studio" && event.subtype === "run_started") {
            runIdRef.current = event.runId ?? null;
          }
          setStream((s) => reduceStreamEvent(s, event));
        },
        abort.signal,
      );
    } catch (err) {
      if (!abort.signal.aborted) {
        toast.error("Run failed", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
    } finally {
      setRunning(false);
      setStream((s) => ({ ...s, running: false }));
      runIdRef.current = null;
      abortRef.current = null;
    }
  }

  async function handleCancel() {
    const runId = runIdRef.current;
    if (runId) {
      try {
        await cancelClaudeRun(runId);
      } catch {
        // Fall through to aborting the stream below.
      }
    }
    abortRef.current?.abort();
  }

  return (
    <div className="flex min-w-0 flex-col gap-4">
      {agent.description ? (
        <p className="text-sm text-muted-foreground">{agent.description}</p>
      ) : null}

      {placeholders.length > 0 ? (
        <div className="flex flex-col gap-3 rounded-lg border border-border/60 bg-muted/20 p-3">
          <span className="text-xs font-medium text-muted-foreground">
            Bind values to the template variables. File variables take a path.
          </span>
          {placeholders.map((name) => (
            <div key={name} className="flex items-center gap-2">
              <Label
                htmlFor={`var-${name}`}
                className="w-28 shrink-0 truncate font-mono text-xs"
              >
                {`{${name}}`}
              </Label>
              <Input
                id={`var-${name}`}
                value={variables[name] ?? ""}
                placeholder={
                  FILE_LIKE_RE.test(name) ? "C:\\path\\to\\file" : "value"
                }
                className="font-mono text-xs"
                onChange={(e) =>
                  setVariables((v) => ({ ...v, [name]: e.target.value }))
                }
              />
            </div>
          ))}
        </div>
      ) : null}

      <div className="flex items-center gap-2">
        <Label htmlFor="run-cwd" className="w-28 shrink-0 text-xs">
          Working dir
        </Label>
        <Input
          id="run-cwd"
          value={cwd}
          placeholder="Optional working directory for the run"
          className="font-mono text-xs"
          onChange={(e) => setCwd(e.target.value)}
        />
      </div>

      <div className="flex items-center gap-2">
        {running ? (
          <Button variant="destructive" onClick={() => void handleCancel()}>
            Stop
          </Button>
        ) : (
          <Button onClick={() => void handleRun()}>Run agent</Button>
        )}
        <span className="font-mono text-[11px] text-muted-foreground">
          {agent.allowedTools ? `tools: ${agent.allowedTools}` : ""}
          {agent.permissionMode !== "default"
            ? ` · ${agent.permissionMode}`
            : ""}
        </span>
      </div>

      {stream.blocks.length > 0 || stream.liveText || stream.running ? (
        <div className="rounded-lg border border-border/60 bg-background p-3">
          <StreamView state={stream} />
        </div>
      ) : null}
    </div>
  );
}
