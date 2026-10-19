# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /jobs used to check only run_config, so a model timeout the worker's BuilderConfig
refuses still minted a workflow key and spawned a worker before failing."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from models.data_recipe import McpToolsListRequest, RecipePayload
from routes.data_recipe import jobs as route


def local_recipe(timeout):
    return {
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [
            {
                "alias": "writer",
                "model": "unsloth/Qwen3-0.6B",
                "provider": "local",
                "inference_parameters": {"timeout": timeout},
            }
        ],
        "columns": [
            {
                "column_type": "llm-text",
                "name": "answer",
                "model_alias": "writer",
                "prompt": "Say hi.",
            }
        ],
    }


@pytest.fixture
def calls(monkeypatch):
    seen = {"mint": 0, "start": 0}

    def inject(recipe, request, expect_gen = None):
        seen["mint"] += 1
        return 7

    class Manager:
        def start(self, **kwargs):
            seen["start"] += 1
            return "job-1"

    monkeypatch.setattr(route, "_inject_local_providers", inject)
    monkeypatch.setattr(route, "_revoke_internal_api_key_safe", lambda key_id: None)
    monkeypatch.setattr(route, "get_job_manager", lambda: Manager())
    return seen


def create(recipe):
    payload = RecipePayload(recipe = recipe, run = {"execution_type": "preview"})
    return route.create_job(payload, request = SimpleNamespace(), credential = ("u", None))


@pytest.mark.parametrize("timeout", [0, -5, 0.5])
def test_a_timeout_the_worker_refuses_is_400_before_a_key_is_minted(calls, timeout):
    with pytest.raises(HTTPException) as exc:
        create(local_recipe(timeout))
    assert exc.value.status_code == 400
    assert "timeout" in str(exc.value.detail)
    assert calls == {"mint": 0, "start": 0}


def test_a_valid_recipe_still_starts(calls):
    assert create(local_recipe(30)) == {"job_id": "job-1"}
    assert calls == {"mint": 1, "start": 1}


def test_load_tools_has_a_timeout_when_the_editor_sends_none():
    assert McpToolsListRequest().timeout_sec == 30
    assert McpToolsListRequest(timeout_sec = None).timeout_sec == 30
    assert McpToolsListRequest.model_validate({"timeout_sec": None}).timeout_sec == 30
