// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { useState } from "react";
import {
  extractPlaceholders,
  saveAgentScript,
  type AgentScript,
} from "../api/claude-agents-api";

const PERMISSION_MODES = [
  { value: "default", label: "Default (prompt for permissions)" },
  { value: "plan", label: "Plan mode" },
  { value: "acceptEdits", label: "Accept edits" },
  { value: "dontAsk", label: "Don't ask (deny unlisted tools)" },
  { value: "bypassPermissions", label: "Bypass permissions" },
] as const;

function slugify(name: string): string {
  return (
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64) || `agent-${Date.now()}`
  );
}

// Render with key={agent.id} so the draft resets when the selection changes.
export function AgentEditor({
  agent,
  workflows,
  onSaved,
  onDelete,
}: {
  agent: AgentScript;
  workflows: string[];
  onSaved: (agent: AgentScript) => void;
  onDelete?: (agent: AgentScript) => void;
}) {
  const [draft, setDraft] = useState<AgentScript>(agent);
  const [saving, setSaving] = useState(false);
  const [newWorkflow, setNewWorkflow] = useState("");

  const placeholders = extractPlaceholders(draft.promptTemplate);
  const isNew = !agent.id;

  function set<K extends keyof AgentScript>(key: K, value: AgentScript[K]) {
    setDraft((d) => ({ ...d, [key]: value }));
  }

  async function handleSave() {
    const name = draft.name.trim();
    if (!name) {
      toast.error("Give the agent a name.");
      return;
    }
    if (!draft.promptTemplate.trim()) {
      toast.error("The prompt template is empty.");
      return;
    }
    const workflow = newWorkflow.trim()
      ? slugify(newWorkflow)
      : draft.workflow;
    const now = Date.now();
    const toSave: AgentScript = {
      ...draft,
      id: draft.id || slugify(name),
      name,
      workflow: workflow || null,
      prebuilt: false,
      createdAt: draft.createdAt || now,
      updatedAt: now,
    };
    setSaving(true);
    try {
      const saved = await saveAgentScript(toSave);
      toast.success(`Saved "${saved.name}".`);
      onSaved(saved);
    } catch (err) {
      toast.error("Failed to save agent", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="agent-name">Name</Label>
          <Input
            id="agent-name"
            value={draft.name}
            maxLength={200}
            placeholder="e.g. PR Review"
            onChange={(e) => set("name", e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="agent-folder">Folder</Label>
          <div className="flex gap-2">
            <Select
              value={draft.workflow ?? "__scripts__"}
              onValueChange={(v) =>
                set("workflow", v === "__scripts__" ? null : v)
              }
            >
              <SelectTrigger id="agent-folder" className="flex-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__scripts__">Scripts (standalone)</SelectItem>
                {workflows.map((w) => (
                  <SelectItem key={w} value={w}>
                    {w}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input
              value={newWorkflow}
              placeholder="or new workflow..."
              className="flex-1"
              onChange={(e) => setNewWorkflow(e.target.value)}
            />
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="agent-description">Description</Label>
        <Input
          id="agent-description"
          value={draft.description}
          maxLength={2000}
          placeholder="What does this agent do?"
          onChange={(e) => set("description", e.target.value)}
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="agent-template">Prompt template</Label>
        <Textarea
          id="agent-template"
          value={draft.promptTemplate}
          rows={10}
          placeholder={
            "Generate a doc for this file that would fit the codebase: {file1}\n\nUse {placeholders} for values you want to bind at run time."
          }
          className="min-h-40 font-mono text-xs leading-relaxed"
          onChange={(e) => set("promptTemplate", e.target.value)}
        />
        {placeholders.length > 0 ? (
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <span>Variables:</span>
            {placeholders.map((name) => (
              <Badge key={name} variant="secondary" className="font-mono">
                {`{${name}}`}
              </Badge>
            ))}
          </div>
        ) : null}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="agent-tools">Allowed tools</Label>
          <Input
            id="agent-tools"
            value={draft.allowedTools}
            maxLength={1000}
            placeholder="Read,Glob,Grep,Bash(git diff *)"
            className="font-mono text-xs"
            onChange={(e) => set("allowedTools", e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="agent-permission">Permission mode</Label>
          <Select
            value={draft.permissionMode}
            onValueChange={(v) => set("permissionMode", v)}
          >
            <SelectTrigger id="agent-permission" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PERMISSION_MODES.map((mode) => (
                <SelectItem key={mode.value} value={mode.value}>
                  {mode.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="agent-cwd">Default working directory (optional)</Label>
        <Input
          id="agent-cwd"
          value={draft.cwd}
          maxLength={1000}
          placeholder="C:\git_dev\my-repo"
          className="font-mono text-xs"
          onChange={(e) => set("cwd", e.target.value)}
        />
      </div>

      <div className="flex items-center gap-2">
        <Button onClick={() => void handleSave()} disabled={saving}>
          {saving ? "Saving..." : isNew ? "Create agent" : "Save changes"}
        </Button>
        {!isNew && onDelete ? (
          <Button variant="destructive" onClick={() => onDelete(agent)}>
            Delete
          </Button>
        ) : null}
      </div>
    </div>
  );
}
