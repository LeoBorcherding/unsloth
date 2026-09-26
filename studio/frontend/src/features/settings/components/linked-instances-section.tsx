// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useState } from "react";
import {
  type LinkedInstance,
  type LinkedInstanceStatus,
  createLinkedInstance,
  deleteLinkedInstance,
  fetchLinkedInstances,
  testLinkedInstance,
} from "../api/linked-instances";
import { SettingsSection } from "./settings-section";

const MODELS_SHOWN = 3;

function LinkedInstanceRow({
  instance,
  status,
  onTest,
  onRemove,
}: {
  instance: LinkedInstance;
  status: LinkedInstanceStatus | "checking" | undefined;
  onTest: () => void;
  onRemove: () => void;
}) {
  const t = useT();
  const checking = status === undefined || status === "checking";
  const online = !checking && status.online;
  const models = checking ? [] : status.models;
  return (
    <div className="flex flex-col gap-1 border-b border-border/60 py-2.5 last:border-b-0">
      <div className="flex flex-wrap items-center gap-2">
        <span
          aria-hidden={true}
          className={cn(
            "size-2 shrink-0 rounded-full",
            checking
              ? "bg-muted-foreground/40"
              : online
                ? "bg-emerald-500"
                : "bg-destructive",
          )}
        />
        <span className="font-mono text-sm font-medium text-foreground">
          @{instance.name}
        </span>
        <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
          {instance.base_url}
        </span>
        <span className="text-xs text-muted-foreground">
          {checking
            ? t("settings.apiKeys.linkedInstances.checking")
            : online
              ? t("settings.apiKeys.linkedInstances.modelCount", {
                  count: String(models.length),
                })
              : (status.error ?? t("settings.apiKeys.linkedInstances.offline"))}
        </span>
        <Button variant="outline" size="sm" onClick={onTest} disabled={checking}>
          {t("settings.apiKeys.linkedInstances.test")}
        </Button>
        <Button variant="outline" size="sm" onClick={onRemove}>
          {t("settings.apiKeys.linkedInstances.remove")}
        </Button>
      </div>
      {models.length > 0 ? (
        <p className="truncate pl-4 font-mono text-ui-11 text-muted-foreground">
          {models.slice(0, MODELS_SHOWN).join(", ")}
          {models.length > MODELS_SHOWN
            ? ` +${models.length - MODELS_SHOWN}`
            : ""}
        </p>
      ) : null}
    </div>
  );
}

export function LinkedInstancesSection() {
  const t = useT();
  const [instances, setInstances] = useState<LinkedInstance[]>([]);
  const [statuses, setStatuses] = useState<
    Record<string, LinkedInstanceStatus | "checking">
  >({});
  const [name, setName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const test = useCallback(async (id: string) => {
    setStatuses((prev) => ({ ...prev, [id]: "checking" }));
    try {
      const status = await testLinkedInstance(id);
      setStatuses((prev) => ({ ...prev, [id]: status }));
    } catch (e) {
      setStatuses((prev) => ({
        ...prev,
        [id]: {
          id,
          online: false,
          error: e instanceof Error ? e.message : null,
          models: [],
        },
      }));
    }
  }, []);

  const load = useCallback(async () => {
    try {
      const loaded = await fetchLinkedInstances();
      setInstances(loaded);
      setError(null);
      for (const instance of loaded) void test(instance.id);
    } catch {
      setError(t("settings.apiKeys.linkedInstances.loadError"));
    }
  }, [t, test]);

  useEffect(() => {
    void load();
  }, [load]);

  const canSubmit =
    name.trim() !== "" && baseUrl.trim() !== "" && apiKey.trim() !== "";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit || saving) return;
    setSaving(true);
    setError(null);
    try {
      await createLinkedInstance({
        name: name.trim(),
        base_url: baseUrl.trim(),
        api_key: apiKey.trim(),
      });
      setName("");
      setBaseUrl("");
      setApiKey("");
      await load();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : t("settings.apiKeys.linkedInstances.saveError"),
      );
    } finally {
      setSaving(false);
    }
  };

  const remove = async (id: string) => {
    try {
      await deleteLinkedInstance(id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : null);
    }
  };

  return (
    <SettingsSection
      title={t("settings.apiKeys.linkedInstances.title")}
      description={t("settings.apiKeys.linkedInstances.description")}
    >
      <form onSubmit={handleSubmit} className="flex flex-wrap gap-2 py-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={t("settings.apiKeys.linkedInstances.namePlaceholder")}
          aria-label={t("settings.apiKeys.linkedInstances.name")}
          className="h-9 w-[calc(120px*var(--ui-space-scale,1))] text-sm"
        />
        <Input
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          placeholder={t("settings.apiKeys.linkedInstances.urlPlaceholder")}
          aria-label={t("settings.apiKeys.linkedInstances.url")}
          className="h-9 min-w-[calc(200px*var(--ui-space-scale,1))] flex-1 text-sm"
        />
        <Input
          type="password"
          autoComplete="off"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder={t("settings.apiKeys.linkedInstances.apiKeyPlaceholder")}
          aria-label={t("settings.apiKeys.linkedInstances.apiKey")}
          className="h-9 w-[calc(180px*var(--ui-space-scale,1))] text-sm"
        />
        <Button type="submit" size="sm" disabled={!canSubmit || saving}>
          {saving
            ? t("settings.apiKeys.linkedInstances.adding")
            : t("settings.apiKeys.linkedInstances.add")}
        </Button>
      </form>
      {error ? (
        <div className="rounded-md border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive">
          {error}
        </div>
      ) : null}
      {instances.length === 0 ? (
        <p className="py-4 text-center text-xs text-muted-foreground">
          {t("settings.apiKeys.linkedInstances.empty")}
        </p>
      ) : (
        <div className="flex flex-col">
          {instances.map((instance) => (
            <LinkedInstanceRow
              key={instance.id}
              instance={instance}
              status={statuses[instance.id]}
              onTest={() => void test(instance.id)}
              onRemove={() => void remove(instance.id)}
            />
          ))}
        </div>
      )}
    </SettingsSection>
  );
}
