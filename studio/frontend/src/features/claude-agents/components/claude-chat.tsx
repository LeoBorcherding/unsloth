// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { XIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  cancelClaudeRun,
  streamClaudeRun,
} from "../api/claude-agents-api";
import {
  emptyStreamState,
  reduceStreamEvent,
  type StreamState,
} from "../lib/stream-state";
import { StreamView } from "./stream-view";

interface FileBinding {
  name: string;
  path: string;
}

interface ChatTurn {
  role: "user" | "assistant";
  /** User turns: rendered prompt text. */
  text?: string;
  /** Assistant turns: distilled stream state. */
  stream?: StreamState;
}

export function ClaudeChat({ cwd }: { cwd?: string }) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [bindings, setBindings] = useState<FileBinding[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const runIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Keep the latest turn in view while streaming.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns]);

  function addFileBinding() {
    setBindings((prev) => {
      const used = new Set(prev.map((b) => b.name));
      let i = 1;
      while (used.has(`file${i}`)) i += 1;
      const name = `file${i}`;
      // Drop the placeholder into the message at the cursor.
      const el = textareaRef.current;
      const token = `{${name}}`;
      if (el) {
        const start = el.selectionStart ?? input.length;
        const end = el.selectionEnd ?? input.length;
        setInput((v) => v.slice(0, start) + token + v.slice(end));
      } else {
        setInput((v) => (v ? `${v} ${token}` : token));
      }
      return [...prev, { name, path: "" }];
    });
  }

  function removeBinding(name: string) {
    setBindings((prev) => prev.filter((b) => b.name !== name));
  }

  async function handleSend() {
    const message = input.trim();
    if (!message || running) return;
    const missing = bindings.filter((b) => !b.path.trim());
    if (missing.length > 0) {
      toast.error(
        `Set a path for ${missing.map((b) => `{${b.name}}`).join(", ")}`,
      );
      return;
    }
    const variables = Object.fromEntries(
      bindings.map((b) => [b.name, b.path.trim()]),
    );

    setInput("");
    setBindings([]);
    setTurns((prev) => [
      ...prev,
      { role: "user", text: message },
      { role: "assistant", stream: { ...emptyStreamState(), running: true } },
    ]);
    setRunning(true);

    const abort = new AbortController();
    abortRef.current = abort;

    const updateLast = (fn: (s: StreamState) => StreamState) => {
      setTurns((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant" && last.stream) {
          next[next.length - 1] = { ...last, stream: fn(last.stream) };
        }
        return next;
      });
    };

    try {
      await streamClaudeRun(
        {
          promptTemplate: message,
          variables,
          // Live chat is a working session: let Claude read and run things,
          // and give it the saved script catalog on the first message.
          allowedTools: "Read,Glob,Grep,Bash,Edit,Write,WebFetch",
          sessionId: sessionId ?? undefined,
          includeAgentCatalog: !sessionId,
          cwd,
        },
        (event) => {
          if (event.type === "studio" && event.subtype === "run_started") {
            runIdRef.current = event.runId ?? null;
            return;
          }
          if (event.type === "system" && event.subtype === "init") {
            if (event.session_id) setSessionId(event.session_id);
          }
          updateLast((s) => reduceStreamEvent(s, event));
        },
        abort.signal,
      );
    } catch (err) {
      if (!abort.signal.aborted) {
        toast.error("Chat request failed", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
    } finally {
      setRunning(false);
      updateLast((s) => ({ ...s, running: false }));
      runIdRef.current = null;
      abortRef.current = null;
    }
  }

  async function handleStop() {
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

  function resetSession() {
    abortRef.current?.abort();
    setTurns([]);
    setSessionId(null);
    setBindings([]);
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div
        ref={scrollRef}
        className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto rounded-lg border border-border/60 bg-background p-3"
      >
        {turns.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-1 text-center text-sm text-muted-foreground">
            <span>Chat live with Claude Code.</span>
            <span className="text-xs">
              It knows all your saved agent scripts and can run them for you.
            </span>
          </div>
        ) : null}
        {turns.map((turn, i) =>
          turn.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] whitespace-pre-wrap rounded-xl bg-primary/10 px-3 py-2 text-sm">
                {turn.text}
              </div>
            </div>
          ) : (
            <StreamView key={i} state={turn.stream ?? emptyStreamState()} />
          ),
        )}
      </div>

      {bindings.length > 0 ? (
        <div className="flex flex-col gap-1.5 rounded-lg border border-border/60 bg-muted/20 p-2">
          {bindings.map((binding) => (
            <div key={binding.name} className="flex items-center gap-2">
              <span className="w-16 shrink-0 font-mono text-xs text-muted-foreground">
                {`{${binding.name}}`}
              </span>
              <Input
                value={binding.path}
                placeholder="C:\path\to\file"
                className="h-8 font-mono text-xs"
                onChange={(e) =>
                  setBindings((prev) =>
                    prev.map((b) =>
                      b.name === binding.name
                        ? { ...b, path: e.target.value }
                        : b,
                    ),
                  )
                }
              />
              <button
                type="button"
                aria-label={`Remove ${binding.name}`}
                className="text-muted-foreground hover:text-foreground"
                onClick={() => removeBinding(binding.name)}
              >
                <XIcon className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
      ) : null}

      <div className="flex flex-col gap-2">
        <Textarea
          ref={textareaRef}
          value={input}
          rows={3}
          placeholder="Ask Claude to run a script, review a PR, document a file..."
          className="min-h-16 resize-none text-sm"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void handleSend();
            }
          }}
        />
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={addFileBinding}
            disabled={running}
          >
            Attach file path
          </Button>
          <span
            className={cn(
              "text-[11px] text-muted-foreground",
              !sessionId && "opacity-0",
            )}
          >
            session {sessionId?.slice(0, 8)}
          </span>
          <div className="ml-auto flex items-center gap-2">
            {turns.length > 0 ? (
              <Button variant="ghost" size="sm" onClick={resetSession}>
                New session
              </Button>
            ) : null}
            {running ? (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => void handleStop()}
              >
                Stop
              </Button>
            ) : (
              <Button size="sm" onClick={() => void handleSend()}>
                Send
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
