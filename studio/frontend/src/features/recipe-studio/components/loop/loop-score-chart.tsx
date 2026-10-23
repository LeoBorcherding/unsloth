// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import type { LoopRound } from "../../loop/types";

const WIDTH = 280;
const PAD_X = 12;
const PAD_TOP = 14;
const PAD_BOTTOM = 8;

/** Pass rate per round, 0 to 100%, against the target as a dashed line. */
export function LoopScoreChart({
  rounds,
  target,
  maxRounds,
  height = 64,
  labels = false,
}: {
  rounds: LoopRound[];
  target: number;
  maxRounds: number;
  height?: number;
  labels?: boolean;
}): ReactElement {
  const slots = Math.max(maxRounds, rounds.length, 2);
  const bottom = height - PAD_BOTTOM - (labels ? 12 : 0);
  const x = (index: number) => PAD_X + ((index - 1) / (slots - 1)) * (WIDTH - PAD_X * 2);
  const y = (score: number) => PAD_TOP + (1 - score) * (bottom - PAD_TOP);
  const points = rounds.flatMap((round) =>
    round.status === "done" && round.result
      ? [{ index: round.index, score: round.result.score }]
      : [],
  );
  const path = points
    .map((point, i) => `${i === 0 ? "M" : "L"} ${x(point.index)} ${y(point.score)}`)
    .join(" ");
  const pending = rounds.find((round) => round.status !== "done" && round.status !== "error");
  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${height}`}
      className="h-auto w-full overflow-visible"
      role="img"
      aria-label={`Score per round: ${points.map((point) => `${Math.round(point.score * 100)}%`).join(", ") || "none yet"}`}
    >
      <line
        x1={PAD_X}
        x2={WIDTH - PAD_X}
        y1={y(target)}
        y2={y(target)}
        stroke="currentColor"
        strokeDasharray="4 4"
        className="text-emerald-500/70"
      />
      <text
        x={WIDTH - PAD_X}
        y={y(target) - 4}
        textAnchor="end"
        className="fill-emerald-600 text-[10px] dark:fill-emerald-400"
      >
        target {Math.round(target * 100)}%
      </text>
      <line
        x1={PAD_X}
        x2={WIDTH - PAD_X}
        y1={bottom}
        y2={bottom}
        stroke="currentColor"
        className="text-border"
      />
      {path && (
        <path
          d={path}
          fill="none"
          stroke="var(--primary)"
          strokeWidth={2}
          strokeLinejoin="round"
        />
      )}
      {points.map((point) => (
        <g key={point.index}>
          <circle cx={x(point.index)} cy={y(point.score)} r={3.5} fill="var(--primary)" />
          {labels && (
            <text
              x={x(point.index)}
              y={y(point.score) - 8}
              textAnchor="middle"
              className="fill-foreground text-[11px] font-medium tabular-nums"
            >
              {Math.round(point.score * 100)}%
            </text>
          )}
        </g>
      ))}
      {pending && (
        <circle
          cx={x(pending.index)}
          cy={bottom}
          r={4}
          fill="none"
          stroke="var(--primary)"
          strokeDasharray="2 2"
          className="animate-pulse"
        />
      )}
      {labels &&
        Array.from({ length: slots }, (_, i) => (
          <text
            // biome-ignore lint/suspicious/noArrayIndexKey: fixed axis
            key={i}
            x={x(i + 1)}
            y={height - 2}
            textAnchor="middle"
            className="fill-muted-foreground text-[10px]"
          >
            Round {i + 1}
          </text>
        ))}
    </svg>
  );
}
