// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useIsAccountOwner } from "@/features/auth";
import { useSettingsDialogStore } from "@/features/settings";
import type {
  LinkedInstance,
  LinkedInstanceInfo,
  LinkedInstanceStatus,
} from "@/features/settings/api/linked-instances";
import {
  GpuMeter,
  LinkedInstanceDetailsDialog,
} from "@/features/settings/components/linked-instance-details-dialog";
import {
  acceleratorLabel,
  connectionKind,
  hostOf,
  platformLabel,
} from "@/features/settings/components/linked-instance-format";
import { useLinkedInstancesOverview } from "@/features/settings/hooks/use-linked-instances-overview";
import { cn } from "@/lib/utils";
import { Link01Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";

const CONNECTION = {
  tunnel: "Cloudflare",
  loopback: "Localhost",
  lan: "LAN",
  public: "Public",
} as const;
const GPUS_SHOWN = 2;

function Chip({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "default" | "warn";
}) {
  return (
    <span
      className={cn(
        "rounded-md border px-1.5 py-0.5 text-ui-10 font-medium tabular-nums",
        tone === "warn"
          ? "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-500"
          : "border-border/60 bg-muted/40 text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

function MachineCard({
  instance,
  status,
  info,
  onOpen,
}: {
  instance: LinkedInstance;
  status: LinkedInstanceStatus | undefined;
  info: LinkedInstanceInfo | undefined;
  onOpen: () => void;
}) {
  const online = status?.online;
  const offline = online === false;
  const prefix = `@${instance.name}/`;
  const serving = status?.loaded[0]?.slice(prefix.length);
  const ready = info?.online === true;
  const accelerator = ready ? acceleratorLabel(info) : null;
  const platform = ready ? platformLabel(info.platform) : null;

  return (
    <button
      type="button"
      onClick={onOpen}
      className="group flex min-w-0 flex-col gap-3 rounded-xl border border-border/60 bg-card px-4 py-3 text-left transition-colors hover:border-border hover:bg-accent/20 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
    >
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col">
          <span className="flex items-center gap-1.5 font-mono text-ui-13 font-medium text-foreground">
            <span
              aria-hidden={true}
              className={cn(
                "size-2 shrink-0 rounded-full",
                online === undefined
                  ? "animate-pulse bg-muted-foreground/50"
                  : online
                    ? "bg-emerald-500"
                    : "bg-red-500",
              )}
            />
            @{instance.name}
          </span>
          <span
            className="truncate font-mono text-ui-11 text-muted-foreground"
            title={instance.base_url}
          >
            {hostOf(instance.base_url)}
          </span>
        </div>
        <span className="shrink-0 text-ui-11 tabular-nums text-muted-foreground">
          {CONNECTION[connectionKind(instance.base_url)]}
          {status?.latency_ms != null ? ` · ${status.latency_ms} ms` : ""}
        </span>
      </div>

      {offline ? (
        <p className="text-ui-12 text-destructive">
          {status?.error ?? "Offline"}
        </p>
      ) : ready ? (
        <div className="flex flex-col gap-2">
          {info.gpus.length > 0 ? (
            info.gpus
              .slice(0, GPUS_SHOWN)
              // biome-ignore lint/suspicious/noArrayIndexKey: identical cards share a name
              .map((gpu, i) => <GpuMeter key={i} gpu={gpu} />)
          ) : (
            <span className="text-ui-12 text-muted-foreground">
              No GPU reported
            </span>
          )}
          {info.gpus.length > GPUS_SHOWN ? (
            <span className="text-ui-11 text-muted-foreground">
              +{info.gpus.length - GPUS_SHOWN} more
            </span>
          ) : null}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-3.5 w-2/3" />
          <Skeleton className="h-1.5 w-full" />
        </div>
      )}

      {!offline ? (
        <div className="flex min-w-0 items-baseline gap-2 text-ui-12">
          <span className="shrink-0 text-muted-foreground">Serving</span>
          <span
            className={cn(
              "truncate",
              serving
                ? "font-mono text-ui-11 text-foreground"
                : "text-muted-foreground",
            )}
            title={serving}
          >
            {serving ??
              (status
                ? `${status.models.length} ${status.models.length === 1 ? "model" : "models"}, none loaded`
                : "…")}
          </span>
        </div>
      ) : null}

      {ready ? (
        <div className="mt-auto flex flex-wrap items-center gap-1.5 border-t border-border/50 pt-2.5">
          {info.version ? <Chip>Unsloth {info.version}</Chip> : null}
          {accelerator ? <Chip>{accelerator}</Chip> : null}
          {platform ? <Chip>{platform}</Chip> : null}
          {info.update_available ? (
            <Chip tone="warn">Update available</Chip>
          ) : null}
        </div>
      ) : null}
    </button>
  );
}

/** Every linked machine at a glance, beside the endpoint card; click one for its details. */
export function LinkedInstancesPanel() {
  const isOwner = useIsAccountOwner();
  const { instances, statuses, infos, refresh, refreshing } =
    useLinkedInstancesOverview(isOwner);
  const [openId, setOpenId] = useState<string | null>(null);

  if (!isOwner || instances.length === 0) return null;

  const online = instances.filter((i) => statuses[i.id]?.online).length;
  const settled = instances.every((i) => statuses[i.id] !== undefined);
  const open = instances.find((i) => i.id === openId) ?? null;

  return (
    <section className="flex min-w-0 flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border/60 bg-muted/40">
            <HugeiconsIcon
              icon={Link01Icon}
              strokeWidth={1.75}
              className="size-4"
            />
          </span>
          <div className="flex flex-col">
            <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
              Linked instances
            </span>
            <span className="flex items-center gap-1.5 text-ui-12 text-foreground">
              <span
                aria-hidden={true}
                className={cn(
                  "size-2 rounded-full",
                  !settled
                    ? "bg-muted-foreground"
                    : online === instances.length
                      ? "bg-emerald-500"
                      : "bg-amber-500",
                )}
              />
              {online} of {instances.length} online · served here as
              <span className="font-mono text-ui-11">@name/model</span>
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="size-8 p-0 text-muted-foreground hover:text-foreground"
            onClick={() => void refresh()}
            disabled={refreshing}
            aria-label="Refresh linked instances"
            title="Refresh"
          >
            <HugeiconsIcon
              icon={RefreshIcon}
              className={cn("size-4", refreshing && "animate-spin")}
            />
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() =>
              useSettingsDialogStore.getState().openDialog("api-keys")
            }
          >
            Manage
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {instances.map((instance) => (
          <MachineCard
            key={instance.id}
            instance={instance}
            status={statuses[instance.id]}
            info={infos[instance.id]}
            onOpen={() => setOpenId(instance.id)}
          />
        ))}
      </div>

      <LinkedInstanceDetailsDialog
        instance={open}
        status={open ? statuses[open.id] : undefined}
        info={open ? infos[open.id] : undefined}
        onOpenChange={(value) => !value && setOpenId(null)}
        onRefresh={() => void refresh()}
        refreshing={refreshing}
      />
    </section>
  );
}
