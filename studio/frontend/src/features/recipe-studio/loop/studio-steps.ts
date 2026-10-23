// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { TrainingRunDetailResponse } from "@/features/training";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { EVAL_LOSS_METRIC } from "./controller";
import { buildLoopTrainingRequest, leafName } from "./training-request";
import type { LoopSteps, LoopTrainResult } from "./types";

const POLL_MS = 3000;
const TERMINAL_PHASES = new Set(["completed", "error", "stopped"]);

type StatusResponse = {
  job_id: string;
  phase: string;
  message: string;
  error: string | null;
  start_request_state?: "pending" | "accepted" | "rejected" | null;
  details?: { output_dir?: string | null } | null;
};

function abortError(): Error {
  const error = new Error("Stopped.");
  error.name = "AbortError";
  return error;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError());
      return;
    }
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

async function readStatus(signal: AbortSignal): Promise<StatusResponse> {
  const res = await authFetch("/api/train/status", { signal });
  if (!res.ok) throw new Error(await readFastApiError(res));
  return (await res.json()) as StatusResponse;
}

async function stopRun(jobId: string): Promise<void> {
  await authFetch("/api/train/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ save: false, expected_job_id: jobId }),
  }).catch(() => undefined);
}

async function trainThroughStudio(
  input: Parameters<LoopSteps["train"]>[0],
): Promise<LoopTrainResult> {
  const { signal } = input;
  const startRequestId = `recipe-loop-${crypto.randomUUID()}`;
  const res = await authFetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...buildLoopTrainingRequest(input),
      start_request_id: startRequestId,
    }),
    signal,
  });
  if (!res.ok) throw new Error(await readFastApiError(res));
  const started = (await res.json()) as {
    job_id: string;
    status: string;
    message: string;
    error?: string | null;
  };
  if (started.status === "error") {
    void authFetch(
      `/api/train/start-requests/${encodeURIComponent(startRequestId)}/acknowledge`,
      { method: "POST" },
    ).catch(() => undefined);
    throw new Error(started.error || started.message || "Training refused.");
  }
  const jobId = started.job_id;
  let outputDir: string | null = null;
  try {
    while (true) {
      await sleep(POLL_MS, signal);
      const status = await readStatus(signal);
      if (status.job_id !== jobId) continue;
      outputDir = status.details?.output_dir ?? outputDir;
      if (status.start_request_state === "rejected") {
        throw new Error(status.error || status.message);
      }
      if (!TERMINAL_PHASES.has(status.phase)) continue;
      if (status.phase === "completed") break;
      throw new Error(
        status.error ||
          (status.phase === "stopped" ? "Training was stopped." : status.message),
      );
    }
  } catch (error) {
    if (signal.aborted) await stopRun(jobId);
    throw error;
  }
  return { runId: jobId, adapter: leafName(outputDir) };
}

async function readRun(
  runId: string,
  signal: AbortSignal,
): Promise<TrainingRunDetailResponse> {
  const res = await authFetch(`/api/train/runs/${encodeURIComponent(runId)}`, {
    signal,
  });
  if (!res.ok) throw new Error(await readFastApiError(res));
  return (await res.json()) as TrainingRunDetailResponse;
}

/** Final eval loss the run recorded on the held-out rows. */
async function scoreEvalLoss(
  input: Parameters<LoopSteps["score"]>[0],
): Promise<number> {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const detail = await readRun(input.run.runId, input.signal);
    const history = detail.metrics.eval_loss_history;
    const last = history[history.length - 1];
    if (typeof last === "number" && Number.isFinite(last)) return last;
    await sleep(2000, input.signal);
  }
  throw new Error(
    "The training run recorded no eval loss. Keep held-out rows above 0, or generate at least 32 rows so training can split its own.",
  );
}

/** Train through Unsloth Studio's training API and score by eval loss; generation comes from the recipe. */
export function createStudioLoopSteps(
  generate: LoopSteps["generate"],
): LoopSteps {
  return {
    metric: EVAL_LOSS_METRIC,
    generate,
    train: trainThroughStudio,
    score: scoreEvalLoss,
  };
}
