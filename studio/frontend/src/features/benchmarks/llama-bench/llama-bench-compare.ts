// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Saved llama-bench runs of one GGUF, lined up by machine and build: the upstream-vs-ours table.

import type { SavedLlamaBenchRun } from "./llama-bench-api";

export interface CompareCell {
  avg: number;
  sd: number;
  best: boolean;
}

export interface CompareTable {
  model: string;
  variant: string | null;
  columns: { key: string; label: string; detail: string }[];
  rows: { test: string; cells: (CompareCell | null)[] }[];
}

const columnKey = (r: SavedLlamaBenchRun) =>
  `${r.meta.machine ?? ""}|${r.meta.build ?? r.config.build ?? "unsloth"}`;

function columnLabel(r: SavedLlamaBenchRun): string {
  const where = r.meta.machine ? `@${r.meta.machine}` : "This machine";
  const build = (r.meta.build ?? r.config.build) === "upstream" ? "upstream" : "Unsloth";
  return `${where} · ${build}`;
}

const testOrder = (t: string) => {
  const m = /^(pp|tg)(\d+)(?: @ d(\d+))?$/.exec(t);
  return m ? [m[1] === "pp" ? 0 : 1, Number(m[3] ?? 0), Number(m[2])] : [2, 0, 0];
};

/** One table per GGUF, newest run per machine and build. A row's best cell is marked only when
 * it beats the runner-up by more than both their spreads, so noise never gets bolded. */
export function compareTables(runs: SavedLlamaBenchRun[]): CompareTable[] {
  const byModel = new Map<string, SavedLlamaBenchRun[]>();
  for (const r of [...runs].sort((a, b) => b.createdAt - a.createdAt)) {
    const k = `${r.model}|${r.ggufVariant ?? ""}`;
    byModel.set(k, [...(byModel.get(k) ?? []), r]);
  }
  const out: CompareTable[] = [];
  for (const group of byModel.values()) {
    const latest = new Map<string, SavedLlamaBenchRun>();
    for (const r of group) if (!latest.has(columnKey(r))) latest.set(columnKey(r), r);
    if (latest.size < 2) continue;
    const cols = [...latest.values()];
    const tests = [...new Set(cols.flatMap((r) => r.outcomes.map((o) => o.test)))].sort(
      (a, b) => {
        const [x, y] = [testOrder(a), testOrder(b)];
        return x[0] - y[0] || x[1] - y[1] || x[2] - y[2];
      },
    );
    out.push({
      model: group[0].model,
      variant: group[0].ggufVariant,
      columns: cols.map((r) => ({
        key: columnKey(r),
        label: columnLabel(r),
        detail: [r.meta.gpu_info, r.meta.backends, r.meta.build_number ? `b${r.meta.build_number}` : null]
          .filter(Boolean)
          .join(" · "),
      })),
      rows: tests.map((test) => {
        const cells = cols.map((r) => {
          const o = r.outcomes.find((x) => x.test === test);
          return o ? { avg: o.avg_ts, sd: o.stddev_ts || 0, best: false } : null;
        });
        const ranked = cells.filter((c): c is CompareCell => c !== null).sort((a, b) => b.avg - a.avg);
        if (ranked.length > 1 && ranked[0].avg - ranked[1].avg > ranked[0].sd + ranked[1].sd)
          ranked[0].best = true;
        return { test, cells };
      }),
    });
  }
  return out;
}

export function toCompareMarkdown(t: CompareTable): string {
  const fmt = (c: CompareCell | null) =>
    c ? `${c.best ? "**" : ""}${c.avg.toFixed(2)} ± ${c.sd.toFixed(2)}${c.best ? "**" : ""}` : "";
  return [
    `| test | ${t.columns.map((c) => c.label).join(" | ")} |`,
    `| ---: | ${t.columns.map(() => "--:").join(" | ")} |`,
    `| | ${t.columns.map((c) => c.detail).join(" | ")} |`,
    ...t.rows.map((r) => `| ${r.test} | ${r.cells.map(fmt).join(" | ")} |`),
  ].join("\n");
}
