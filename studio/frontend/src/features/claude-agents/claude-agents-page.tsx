// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  deleteAgentScript,
  getClaudeStatus,
  listAgentScripts,
  newAgentDraft,
  type AgentScript,
} from "./api/claude-agents-api";
import { AgentEditor } from "./components/agent-editor";
import { AgentRunner } from "./components/agent-runner";
import { ClaudeChat } from "./components/claude-chat";

// Claude's brand terracotta; used as an accent so it's clear this tab is
// powered by Anthropic's Claude Code rather than Studio's own models.
const CLAUDE_ACCENT = "#D97757";

function ClaudeMark({ className }: { className?: string }) {
  // Simple starburst in the spirit of the Claude logomark.
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className={className}
      style={{ color: CLAUDE_ACCENT }}
    >
      <title>Claude</title>
      {Array.from({ length: 8 }).map((_, i) => (
        <line
          key={i}
          x1="12"
          y1="12"
          x2={12 + 9 * Math.cos((i * Math.PI) / 4 + Math.PI / 8)}
          y2={12 + 9 * Math.sin((i * Math.PI) / 4 + Math.PI / 8)}
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}

export function ClaudeAgentsPage() {
  const [agents, setAgents] = useState<AgentScript[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState<AgentScript | null>(null);
  const [view, setView] = useState("run");
  const [cliStatus, setCliStatus] = useState<{
    available: boolean;
    version: string | null;
  } | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await listAgentScripts();
      setAgents(list);
      setSelectedId((id) => id ?? list[0]?.id ?? null);
    } catch (err) {
      toast.error("Failed to load agents", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setHasLoaded(true);
    }
  }, []);

  useEffect(() => {
    // Microtask defer keeps state updates out of the synchronous effect body
    // (same pattern as the app sidebar's collapsible sync).
    queueMicrotask(() => {
      void refresh();
      getClaudeStatus()
        .then(setCliStatus)
        .catch(() => setCliStatus({ available: false, version: null }));
    });
  }, [refresh]);

  const selected = useMemo(
    () => agents.find((a) => a.id === selectedId) ?? null,
    [agents, selectedId],
  );
  const workflows = useMemo(
    () =>
      [...new Set(agents.map((a) => a.workflow).filter((w): w is string => !!w))],
    [agents],
  );
  const grouped = useMemo(() => {
    const groups: Array<{ label: string; items: AgentScript[] }> = [
      { label: "Scripts", items: agents.filter((a) => !a.workflow) },
    ];
    for (const workflow of workflows) {
      groups.push({
        label: workflow,
        items: agents.filter((a) => a.workflow === workflow),
      });
    }
    return groups.filter((g) => g.items.length > 0);
  }, [agents, workflows]);

  async function commitDelete() {
    const agent = deleting;
    if (!agent) return;
    setDeleting(null);
    try {
      await deleteAgentScript(agent.id);
      setAgents((prev) => prev.filter((a) => a.id !== agent.id));
      if (selectedId === agent.id) setSelectedId(null);
      toast.success(`Deleted "${agent.name}".`);
    } catch (err) {
      toast.error("Failed to delete agent", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col">
      <header className="flex flex-wrap items-center gap-3 border-b border-border/60 px-4 py-3">
        <ClaudeMark className="size-6 shrink-0" />
        <div className="flex min-w-0 flex-col">
          <h1 className="font-heading text-lg font-semibold leading-tight tracking-tight">
            Claude Agent Scripts
          </h1>
          <span className="text-xs text-muted-foreground">
            Build, save and run Claude Code prompt-script agents
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Badge
            variant="outline"
            className="gap-1.5 border-[#D97757]/40 text-[11px]"
            style={{ color: CLAUDE_ACCENT }}
          >
            <ClaudeMark className="size-3" />
            Powered by Claude Code
          </Badge>
          {cliStatus && !cliStatus.available ? (
            <Badge variant="destructive" className="text-[11px]">
              claude CLI not found
            </Badge>
          ) : cliStatus?.version ? (
            <span className="font-mono text-[11px] text-muted-foreground">
              {cliStatus.version}
            </span>
          ) : null}
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Agent list */}
        <aside className="flex w-64 shrink-0 flex-col gap-1 overflow-y-auto border-r border-border/60 p-3">
          <Button
            size="sm"
            className="mb-2"
            onClick={() => {
              setCreating(true);
              setView("edit");
            }}
          >
            New agent
          </Button>
          {!hasLoaded ? (
            <>
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </>
          ) : null}
          {grouped.map((group) => (
            <div key={group.label} className="flex flex-col gap-0.5">
              <span className="px-2 pt-2 pb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                {group.label}
              </span>
              {group.items.map((agent) => (
                <button
                  key={agent.id}
                  type="button"
                  className={cn(
                    "flex flex-col items-start gap-0.5 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-accent",
                    !creating &&
                      selectedId === agent.id &&
                      "bg-accent font-medium",
                  )}
                  onClick={() => {
                    setCreating(false);
                    setSelectedId(agent.id);
                  }}
                >
                  <span className="w-full truncate">{agent.name}</span>
                  {agent.description ? (
                    <span className="w-full truncate text-[11px] font-normal text-muted-foreground">
                      {agent.description}
                    </span>
                  ) : null}
                </button>
              ))}
            </div>
          ))}
          {hasLoaded && agents.length === 0 ? (
            <span className="px-2 py-1 text-xs text-muted-foreground">
              No saved agents yet.
            </span>
          ) : null}
        </aside>

        {/* Main area */}
        <main className="flex min-h-0 min-w-0 flex-1 flex-col p-4">
          <Tabs
            value={view}
            onValueChange={setView}
            className="flex min-h-0 flex-1 flex-col"
          >
            <TabsList className="mb-3 w-fit">
              <TabsTrigger value="run">Run</TabsTrigger>
              <TabsTrigger value="edit">
                {creating ? "New agent" : "Edit"}
              </TabsTrigger>
              <TabsTrigger value="chat">Live chat</TabsTrigger>
            </TabsList>

            <TabsContent value="run" className="min-h-0 flex-1 overflow-y-auto">
              {creating || !selected ? (
                <p className="text-sm text-muted-foreground">
                  {creating
                    ? "Save the new agent first, then run it."
                    : "Select an agent on the left, or create one."}
                </p>
              ) : (
                <AgentRunner key={selected.id} agent={selected} />
              )}
            </TabsContent>

            <TabsContent value="edit" className="min-h-0 flex-1 overflow-y-auto">
              {creating ? (
                <AgentEditor
                  key="__new__"
                  agent={newAgentDraft()}
                  workflows={workflows}
                  onSaved={(saved) => {
                    setCreating(false);
                    setSelectedId(saved.id);
                    void refresh();
                    setView("run");
                  }}
                />
              ) : selected ? (
                <AgentEditor
                  key={selected.id}
                  agent={selected}
                  workflows={workflows}
                  onSaved={() => void refresh()}
                  onDelete={(agent) => setDeleting(agent)}
                />
              ) : (
                <p className="text-sm text-muted-foreground">
                  Select an agent on the left, or create one.
                </p>
              )}
            </TabsContent>

            <TabsContent
              value="chat"
              className="flex min-h-0 flex-1 flex-col data-[state=inactive]:hidden"
            >
              <ClaudeChat />
            </TabsContent>
          </Tabs>
        </main>
      </div>

      <Dialog
        open={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null);
        }}
      >
        <DialogContent className="corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete agent</DialogTitle>
            <DialogDescription>
              Delete{" "}
              <span className="font-medium text-foreground">
                &quot;{deleting?.name}&quot;
              </span>
              ? This removes its script file from disk.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button variant="ghost" onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => void commitDelete()}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
