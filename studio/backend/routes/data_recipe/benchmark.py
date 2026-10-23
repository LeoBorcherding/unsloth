# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark card endpoints: score a model on a held-out slice of a recipe run."""

from __future__ import annotations

import copy
import os
import threading
import time
import uuid
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from auth.authentication import get_current_credential
from auth.storage import CredentialRotated
from core.data_recipe.benchmark import MAX_BENCHMARK_ROWS, SCORERS, run_benchmark
from core.data_recipe.service import require_ui_session_for_host_env_credentials
from core.training.account_jobs import managed_account
from loggers import get_logger
from utils.utils import log_and_http_error, safe_curated_detail

from .jobs import _inject_local_providers, _revoke_internal_api_key_safe

logger = get_logger(__name__)
router = APIRouter()

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TIMEOUT_S = 120


class BenchmarkBaseline(BaseModel):
    mode: Literal["none", "adapter_off", "alias"] = "none"
    alias: Optional[str] = None


class BenchmarkRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length = 1, max_length = MAX_BENCHMARK_ROWS)
    prompt_column: str = Field(min_length = 1)
    reference_column: Optional[str] = None
    system_prompt: str = ""
    scorer: Literal[SCORERS]  # type: ignore[valid-type]
    rubric: str = ""
    model_alias: str = Field(min_length = 1)
    baseline: BenchmarkBaseline = BenchmarkBaseline()
    judge_alias: Optional[str] = None
    model_providers: list[dict[str, Any]] = Field(default_factory = list)
    model_configs: list[dict[str, Any]] = Field(default_factory = list)


_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_cancel: dict[str, threading.Event] = {}


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _resolve_target(
    alias: str, configs: dict[str, dict[str, Any]], providers: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    mc = configs.get(alias)
    if mc is None:
        raise HTTPException(status_code = 400, detail = f"Model {alias} is not in the recipe.")
    provider = providers.get(str(mc.get("provider") or ""))
    if provider is None:
        raise HTTPException(status_code = 400, detail = f"Model {alias} has no provider.")
    params = dict(mc.get("inference_parameters") or {})
    api_key = provider.get("api_key") or ""
    env_name = provider.get("api_key_env")
    if not api_key and isinstance(env_name, str) and env_name:
        api_key = os.environ.get(env_name, "")
    body: dict[str, Any] = {"model": mc.get("model") or "local"}
    body.update(provider.get("extra_body") or {})
    body.update(params.pop("extra_body", None) or {})
    timeout = params.pop("timeout", None) or _DEFAULT_TIMEOUT_S
    params.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
    params.setdefault("temperature", 0)
    body.update(params)
    headers = dict(provider.get("extra_headers") or {})
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return {
        "url": f"{str(provider.get('endpoint') or '').rstrip('/')}/chat/completions",
        "headers": headers,
        "body": body,
        "timeout": float(timeout),
    }


def _chat(client: httpx.Client, target: dict[str, Any], messages: list[dict[str, str]]) -> str:
    response = client.post(
        target["url"],
        json = {**target["body"], "messages": messages, "stream": False},
        headers = target["headers"],
        timeout = target["timeout"],
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail") or response.text
        except ValueError:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {str(detail)[:300]}")
    message = response.json()["choices"][0]["message"]
    return str(message.get("content") or "")


def _run(job_id: str, spec: dict[str, Any], targets: dict[str, dict[str, Any]], key_id: Optional[int]) -> None:
    cancel = _cancel[job_id]

    def progress(done: int, total: int) -> None:
        with _lock:
            _jobs[job_id]["progress"] = {"done": done, "total": total}

    try:
        with httpx.Client() as client:
            result = run_benchmark(
                spec,
                chat = lambda target, messages: _chat(client, targets[target], messages),
                on_progress = progress,
                is_cancelled = cancel.is_set,
            )
        with _lock:
            _jobs[job_id].update(
                status = "cancelled" if result["cancelled"] else "completed",
                result = result,
                finished_at = time.time(),
            )
    except Exception as exc:
        logger.exception("data_recipe.benchmark.failed")
        with _lock:
            _jobs[job_id].update(status = "error", error = str(exc)[:500], finished_at = time.time())
    finally:
        if key_id is not None:
            _revoke_internal_api_key_safe(key_id)


@router.post("/benchmark/jobs")
def start_benchmark(
    payload: BenchmarkRequest,
    request: Request,
    credential: tuple = Depends(get_current_credential),
):
    if payload.scorer == "judge" and not payload.judge_alias:
        raise HTTPException(status_code = 400, detail = "Pick a judge model for LLM judge scoring.")
    if payload.baseline.mode == "alias" and not payload.baseline.alias:
        raise HTTPException(status_code = 400, detail = "Pick a baseline model.")
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(status_code = 409, detail = "A benchmark is already running.")

    aliases = [payload.model_alias]
    if payload.baseline.mode == "alias":
        aliases.append(payload.baseline.alias)
    if payload.scorer == "judge":
        aliases.append(payload.judge_alias)
    recipe = {
        "model_providers": copy.deepcopy(payload.model_providers),
        "model_configs": copy.deepcopy(payload.model_configs),
        "columns": [
            {"name": f"benchmark_{i}", "column_type": "llm-text", "model_alias": alias}
            for i, alias in enumerate(aliases)
        ],
    }
    require_ui_session_for_host_env_credentials(recipe, credential[1])
    local_names = {
        p.get("name") for p in recipe["model_providers"] if isinstance(p, dict) and p.get("is_local") is True
    }
    configs = {mc.get("alias"): mc for mc in recipe["model_configs"] if isinstance(mc, dict)}
    if managed_account() and any(
        configs.get(alias, {}).get("provider") not in local_names for alias in aliases
    ):
        raise HTTPException(
            status_code = 403,
            detail = "Managed accounts can only benchmark with the local model.",
        )
    try:
        key_id = _inject_local_providers(recipe, request, credential[1])
    except CredentialRotated as exc:
        raise HTTPException(status_code = 401, detail = "Invalid or expired token") from exc
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_curated_detail(exc),
            event = "data_recipe.benchmark.inject_local_providers_failed",
            log = logger,
        ) from exc

    try:
        providers = {p.get("name"): p for p in recipe["model_providers"] if isinstance(p, dict)}
        candidate = _resolve_target(payload.model_alias, configs, providers)
        targets = {"candidate": candidate}
        if payload.baseline.mode == "adapter_off":
            # One loaded model serves both sides: Studio's /v1 toggles the LoRA per request.
            candidate["body"]["use_adapter"] = True
            targets["baseline"] = copy.deepcopy(candidate)
            targets["baseline"]["body"]["use_adapter"] = False
        elif payload.baseline.mode == "alias":
            targets["baseline"] = _resolve_target(payload.baseline.alias, configs, providers)
        if payload.scorer == "judge":
            targets["judge"] = _resolve_target(payload.judge_alias, configs, providers)
            judge_provider = configs[payload.judge_alias].get("provider")
            if payload.baseline.mode == "adapter_off" and judge_provider in local_names:
                # The local model can judge, but never with the adapter it is grading.
                targets["judge"]["body"]["use_adapter"] = False
    except Exception:
        if key_id is not None:
            _revoke_internal_api_key_safe(key_id)
        raise

    spec = {
        "rows": payload.rows,
        "prompt_column": payload.prompt_column,
        "reference_column": payload.reference_column,
        "system_prompt": payload.system_prompt,
        "scorer": payload.scorer,
        "rubric": payload.rubric,
        "with_baseline": "baseline" in targets,
    }
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs.clear()
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": {"done": 0, "total": len(payload.rows) * (2 if "baseline" in targets else 1)},
            "result": None,
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        }
        _cancel.clear()
        _cancel[job_id] = threading.Event()
    threading.Thread(
        target = _run, args = (job_id, spec, targets, key_id), daemon = True, name = "recipe-benchmark"
    ).start()
    return {"job_id": job_id}


@router.get("/benchmark/jobs/{job_id}")
def benchmark_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code = 404, detail = "benchmark not found")
        return _public(copy.deepcopy(job))


@router.post("/benchmark/jobs/{job_id}/cancel")
def cancel_benchmark(job_id: str):
    with _lock:
        event = _cancel.get(job_id)
    if event is None:
        raise HTTPException(status_code = 404, detail = "benchmark not found")
    event.set()
    return {"job_id": job_id}
