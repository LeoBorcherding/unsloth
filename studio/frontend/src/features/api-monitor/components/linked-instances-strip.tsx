// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import { useSettingsDialogStore } from "@/features/settings";
import {
  type LinkedInstance,
  type LinkedInstanceStatus,
  fetchLinkedInstances,
  fetchLinkedInstancesStatus,
} from "@/features/settings/api/linked-instances";
import { cn } from "@/lib/utils";
import { Link01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";

const POLL_MS = 30_000;

function summary(prefix: string, status: LinkedInstanceStatus | undefined) {
  if (!status) return "Checking…";
  if (!status.online) return status.error ?? "Offline";
  const loaded = status.loaded[0]?.slice(prefix.length);
  if (loaded) return loaded;
  return `${status.models.length} models, none loaded`;
}

/** Linked instances served through this base URL, beside the endpoint card. */
export function LinkedInstancesStrip() {
  const isOwner = useIsAccountOwner();
  const [instances, setInstances] = useState<LinkedInstance[]>([]);
  const [statuses, setStatuses] = useState<Record<string, LinkedInstanceStatus>>(
    {},
  );

  useEffect(() => {
    if (!isOwner) return;
    let cancelled = false;
    let timer: number | null = null;
    const refresh = async () => {
      try {
        const [list, status] = await Promise.all([
          fetchLinkedInstances(),
          fetchLinkedInstancesStatus(),
        ]);
        if (cancelled) return;
        setInstances(list);
        setStatuses(Object.fromEntries(status.map((s) => [s.id, s])));
      } catch {
        // Keep the last answer; the settings card reports errors.
      }
      if (!cancelled) timer = window.setTimeout(refresh, POLL_MS);
    };
    void refresh();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [isOwner]);

  if (!isOwner || instances.length === 0) return null;

  return (
    <section className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-border/60 bg-card px-4 py-3">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border/60 bg-muted/40">
          <HugeiconsIcon icon={Link01Icon} strokeWidth={1.75} className="size-4" />
        </span>
        <div className="flex min-w-0 flex-col">
          <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
            Linked instances
          </span>
          <button
            type="button"
            onClick={() =>
              useSettingsDialogStore.getState().openDialog("api-keys")
            }
            className="w-fit text-ui-12 text-muted-foreground underline decoration-border underline-offset-2 transition-colors hover:text-foreground hover:decoration-foreground"
          >
            Manage
          </button>
        </div>
      </div>
      {instances.map((instance) => {
        const status = statuses[instance.id];
        const prefix = `@${instance.name}/`;
        return (
          <div key={instance.id} className="flex min-w-0 max-w-72 flex-col">
            <span className="flex items-center gap-1.5 font-mono text-ui-12 text-foreground">
              <span
                aria-hidden={true}
                className={cn(
                  "size-2 shrink-0 rounded-full",
                  !status
                    ? "animate-pulse bg-muted-foreground/50"
                    : status.online
                      ? "bg-emerald-500"
                      : "bg-red-500",
                )}
              />
              @{instance.name}
              {status?.latency_ms != null ? (
                <span className="font-sans text-ui-11 tabular-nums text-muted-foreground">
                  {status.latency_ms} ms
                </span>
              ) : null}
            </span>
            <span
              className={cn(
                "truncate text-ui-12",
                status && !status.online
                  ? "text-destructive"
                  : "text-muted-foreground",
              )}
              title={instance.base_url}
            >
              {summary(prefix, status)}
            </span>
          </div>
        );
      })}
    </section>
  );
}
