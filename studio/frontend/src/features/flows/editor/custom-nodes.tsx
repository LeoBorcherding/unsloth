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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  type SandboxCapability,
  loadSandboxCapability,
} from "@/features/chat/api/sandbox-capability";
import { CollapsibleSectionTriggerButton } from "@/features/recipe-studio/dialogs/shared/collapsible-section-trigger";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  createCustomNode,
  deleteCustomNode,
  downloadCustomNodeTemplate,
  getCustomNodeSandbox,
  getSandboxLibraries,
  parseCustomNode,
  replaceCustomNode,
} from "../api";
import {
  loadCustomNodes,
  newerLibraryVersion,
  useCustomNodesStore,
} from "../stores/custom-nodes-store";
import { useDocStore } from "../stores/doc-store";
import type {
  CustomNodeEntry,
  CustomNodeSandbox,
  NodeSpec,
  SandboxLibraries,
} from "../types";
import {
  Download01Icon,
  MoreHorizontalIcon,
  Upload01Icon,
} from "./icon-data";
import { portSummary, updateNodesTo } from "./custom-node-actions";
import { downloadBlob } from "./node-actions";

export const CUSTOM_COLOR = "#64748b";

/** Up front in the dialog: custom nodes need the OS sandbox, and whether this machine has it. */
function SandboxRequirement({ open }: { open: boolean }) {
  const [status, setStatus] = useState<CustomNodeSandbox | null>(null);
  const [capability, setCapability] = useState<SandboxCapability | null>(null);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    getCustomNodeSandbox()
      .then((found) => alive && setStatus(found))
      .catch(() => alive && setStatus(null));
    void loadSandboxCapability().then(
      (found) => alive && setCapability(found),
    );
    return () => {
      alive = false;
    };
  }, [open]);

  return (
    <div className="space-y-1.5 text-xs" data-testid="custom-node-requirement">
      <p>
        Custom nodes only run inside Studio's OS sandbox (bubblewrap on Linux,
        Seatbelt on macOS). Without it they refuse to run; unlike the Python
        block they never fall back to software safeguards. They do not run on
        Windows, where Studio's sandbox reports it is not a security boundary.
        Uploading never runs the script.
      </p>
      {status && (
        <p
          className={status.ready ? "text-muted-foreground" : "text-destructive"}
          data-testid="custom-node-isolation"
        >
          {status.ready
            ? `This machine meets it${capability?.backend ? ` (${capability.backend})` : ""}.`
            : `This machine does not: ${status.message}`}
        </p>
      )}
    </div>
  );
}

function SandboxFacts({ open }: { open: boolean }) {
  const [libraries, setLibraries] = useState<SandboxLibraries | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    getSandboxLibraries()
      .then((found) => alive && setLibraries(found))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [open]);

  return (
    <div className="space-y-2 text-xs" data-testid="custom-node-sandbox">
      <div>
        <p className="mb-1 font-medium text-foreground">
          {libraries
            ? `Python ${libraries.python} with the standard library and ${libraries.packages.length} packages${libraries.source === "sandbox" ? ", read from the sandbox" : ", read from Studio's Python (the sandbox probe did not answer)"}:`
            : "Libraries the sandbox can import:"}
        </p>
        {libraries ? (
          <div
            className="flex max-h-28 flex-wrap gap-1 overflow-y-auto rounded-md border border-border p-1.5"
            data-testid="custom-node-libraries"
          >
            {libraries.packages.map((p) => (
              <Badge
                key={p.name}
                variant="outline"
                className="font-normal text-ui-10"
                title={p.version}
              >
                {p.name}
              </Badge>
            ))}
          </div>
        ) : failed ? (
          <p className="text-muted-foreground">Could not read the list.</p>
        ) : (
          <Spinner className="size-3.5" />
        )}
      </div>
    </div>
  );
}

export function CustomNodeDialog({
  open,
  onOpenChange,
  entry,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  entry: CustomNodeEntry | null;
}) {
  const [source, setSource] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<NodeSpec | null>(null);
  const [busy, setBusy] = useState(false);
  const file = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setSource(entry?.source ?? "");
    setError(null);
    setPreview(entry?.spec ?? null);
  }, [open, entry]);

  const check = async (text: string) => {
    setError(null);
    setPreview(null);
    if (!text.trim()) return;
    try {
      setPreview((await parseCustomNode(text)).spec);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not read the script");
    }
  };

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const saved = entry
        ? await replaceCustomNode(entry.id, source)
        : await createCustomNode(source);
      await loadCustomNodes();
      toast.success(`${saved.name} is in your custom nodes`);
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the script");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl" data-testid="custom-node-dialog">
        <DialogHeader>
          <DialogTitle>
            {entry ? `Edit ${entry.name}` : "Upload a custom node"}
          </DialogTitle>
          <DialogDescription>
            One Python file: a <code>NODE = {"{...}"}</code> header and a{" "}
            <code>def run(inputs, params)</code> that returns a dict. Start
            from the template if you have not written one yet.
          </DialogDescription>
        </DialogHeader>
        <SandboxRequirement open={open} />
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={() => file.current?.click()}
            data-testid="custom-node-pick-file"
          >
            <HugeiconsIcon icon={Upload01Icon} />
            Pick a .py file
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => void downloadTemplate()}
          >
            <HugeiconsIcon icon={Download01Icon} />
            Download template
          </Button>
          <input
            ref={file}
            type="file"
            accept=".py,text/x-python,text/plain"
            className="hidden"
            data-testid="custom-node-file"
            onChange={(e) => {
              const picked = e.target.files?.[0];
              e.target.value = "";
              if (!picked) return;
              void picked.text().then((text) => {
                setSource(text);
                void check(text);
              });
            }}
          />
        </div>
        <Textarea
          value={source}
          onChange={(e) => setSource(e.target.value)}
          onBlur={() => void check(source)}
          placeholder="Paste your script here"
          spellCheck={false}
          className="h-56 font-mono text-xs"
          data-testid="custom-node-source"
        />
        {error && (
          <p
            className="whitespace-pre-wrap text-destructive text-xs"
            data-testid="custom-node-error"
          >
            {error}
          </p>
        )}
        {preview && !error && (
          <p className="text-xs" data-testid="custom-node-preview">
            <span className="font-semibold">{preview.title}</span>:{" "}
            {portSummary(preview)}
          </p>
        )}
        <SandboxFacts open={open} />
        <DialogFooter>
          <Button
            type="button"
            variant="ghost"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            disabled={busy || !source.trim()}
            onClick={() => void save()}
            data-testid="custom-node-save"
          >
            {entry ? "Save" : "Add node"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

async function downloadTemplate(): Promise<void> {
  try {
    downloadBlob(await downloadCustomNodeTemplate(), "my_node_template.py");
  } catch (e) {
    toast.error(e instanceof Error ? e.message : "Could not get the template");
  }
}

async function replaceFromFile(entry: CustomNodeEntry, picked: File) {
  try {
    await replaceCustomNode(entry.id, await picked.text());
    await loadCustomNodes();
    toast.success(`Replaced ${entry.name}`);
  } catch (e) {
    toast.error(e instanceof Error ? e.message : "Could not replace it");
  }
}

async function remove(entry: CustomNodeEntry) {
  if (
    !window.confirm(
      `Delete ${entry.name} from your custom nodes? Flows that use it keep their own copy.`,
    )
  )
    return;
  try {
    await deleteCustomNode(entry.id);
    await loadCustomNodes();
  } catch (e) {
    toast.error(e instanceof Error ? e.message : "Could not delete it");
  }
}

/** The palette's Custom section: the account's scripts as nodes, plus upload and the template. */
export function CustomNodesSection({
  query,
  renderItem,
}: {
  query: string;
  renderItem: (spec: NodeSpec) => ReactNode;
}) {
  const entries = useCustomNodesStore((s) => s.entries);
  const loaded = useCustomNodesStore((s) => s.loaded);
  const nodes = useDocStore((s) => s.graph.nodes);
  const [open, setOpen] = useState(true);
  const [dialog, setDialog] = useState<{ entry: CustomNodeEntry | null } | null>(
    null,
  );
  const replacing = useRef<CustomNodeEntry | null>(null);
  const file = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!loaded) void loadCustomNodes().catch(() => undefined);
  }, [loaded]);

  const shown = useMemo(
    () =>
      entries.filter(
        (e) =>
          !query ||
          e.name.toLowerCase().includes(query) ||
          (e.spec?.description ?? "").toLowerCase().includes(query),
      ),
    [entries, query],
  );
  const stale = useMemo(() => {
    const found: Record<string, number> = {};
    for (const node of Object.values(nodes)) {
      const entry = newerLibraryVersion(node, entries);
      if (entry) found[entry.id] = (found[entry.id] ?? 0) + 1;
    }
    return found;
  }, [nodes, entries]);

  if (query && shown.length === 0) return null;
  return (
    <div className="mb-2" data-testid="palette-custom">
      <CollapsibleSectionTriggerButton
        label="Custom"
        open={open || Boolean(query)}
        summary={String(entries.length)}
        className="px-3 py-1.5"
        onClick={() => setOpen((v) => !v)}
      />
      {(open || query) && (
        <>
          <div className="flex gap-1 px-3 pb-1.5">
            <Button
              type="button"
              variant="outline"
              size="xs"
              onClick={() => setDialog({ entry: null })}
              data-testid="custom-node-upload"
            >
              <HugeiconsIcon icon={Upload01Icon} />
              Upload script
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={() => void downloadTemplate()}
              data-testid="custom-node-template"
            >
              <HugeiconsIcon icon={Download01Icon} />
              Template
            </Button>
          </div>
          {shown.map((entry) => (
            <div
              key={entry.id}
              className="group/custom relative"
              data-testid={`custom-node-${entry.id}`}
            >
              {entry.spec ? (
                renderItem(entry.spec)
              ) : (
                <p className="px-3 py-1.5 text-destructive text-ui-11">
                  {entry.name}: {entry.error}
                </p>
              )}
              {stale[entry.id] && (
                <button
                  type="button"
                  className="mx-3 mb-1 text-left text-primary text-ui-11 underline"
                  onClick={() => {
                    const n = updateNodesTo(entry);
                    toast.success(`Updated ${n} node${n === 1 ? "" : "s"}`);
                  }}
                  data-testid="custom-node-update"
                >
                  {stale[entry.id]} node{stale[entry.id] === 1 ? " uses" : "s use"}{" "}
                  an older version; update
                </button>
              )}
              <div className="absolute top-1.5 right-1.5 opacity-0 transition group-hover/custom:opacity-100 focus-within:opacity-100">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild={true}>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`${entry.name} actions`}
                      data-testid="custom-node-menu"
                    >
                      <HugeiconsIcon icon={MoreHorizontalIcon} size={14} />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onSelect={() => setDialog({ entry })}>
                      Edit
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={() => {
                        replacing.current = entry;
                        file.current?.click();
                      }}
                    >
                      Replace from file
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      variant="destructive"
                      onSelect={() => void remove(entry)}
                    >
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          ))}
          {entries.length === 0 && (
            <p className="px-3 py-1 text-muted-foreground text-ui-11">
              Turn a Python script into a node. It runs in Studio's sandbox.
            </p>
          )}
        </>
      )}
      <input
        ref={file}
        type="file"
        accept=".py,text/x-python,text/plain"
        className="hidden"
        onChange={(e) => {
          const picked = e.target.files?.[0];
          const entry = replacing.current;
          e.target.value = "";
          replacing.current = null;
          if (picked && entry) void replaceFromFile(entry, picked);
        }}
      />
      <CustomNodeDialog
        open={dialog !== null}
        onOpenChange={(next) => !next && setDialog(null)}
        entry={dialog?.entry ?? null}
      />
    </div>
  );
}
