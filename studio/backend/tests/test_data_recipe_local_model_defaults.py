# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from types import SimpleNamespace

from auth import storage
from routes.data_recipe import jobs as route

MODEL = "unsloth/gemma-4-E2B-it-GGUF"


def recipe(*, local_params = None, remote_params = None):
    local_mc = {"alias": "local_model", "model": MODEL, "provider": "local"}
    if local_params is not None:
        local_mc["inference_parameters"] = local_params
    remote_mc = {"alias": "remote_model", "model": "gpt-4o-mini", "provider": "remote"}
    if remote_params is not None:
        remote_mc["inference_parameters"] = remote_params
    return {
        "model_providers": [
            {"name": "local", "is_local": True},
            {"name": "remote", "endpoint": "https://api.example.com/v1"},
        ],
        "model_configs": [local_mc, remote_mc],
        "columns": [
            {"column_type": "llm-text", "name": "question", "model_alias": "local_model"},
            {"column_type": "llm-text", "name": "answer", "model_alias": "remote_model"},
        ],
    }


def inject(monkeypatch, payload, *, gguf = True, slots = 1, can_batch = True):
    llama = SimpleNamespace(
        is_loaded = gguf, model_identifier = MODEL, hf_variant = "", effective_parallel_slots = slots
    )
    backend = SimpleNamespace(
        active_model_name = "" if gguf else MODEL,
        effective_parallel_slots = slots,
        models = {MODEL: {"can_batch": can_batch}},
    )
    monkeypatch.setitem(
        sys.modules, "routes.inference", SimpleNamespace(get_llama_cpp_backend = lambda: llama)
    )
    monkeypatch.setitem(
        sys.modules, "core.inference", SimpleNamespace(get_inference_backend = lambda: backend)
    )
    monkeypatch.setattr(
        route, "_resolve_local_v1_endpoint", lambda request: "http://127.0.0.1:8888/v1"
    )
    monkeypatch.setattr(route, "_inject_local_structured_response_format", lambda *a: None)
    monkeypatch.setattr(storage, "create_api_key", lambda **kwargs: ("sk-unsloth-test", {"id": 7}))
    route._inject_local_providers(payload, SimpleNamespace())
    return {mc["alias"]: mc.get("inference_parameters") or {} for mc in payload["model_configs"]}


def test_a_local_model_left_unset_gets_a_local_timeout_and_the_loaded_slot_count(monkeypatch):
    params = inject(monkeypatch, recipe(), slots = 1)

    assert params["local_model"]["timeout"] == 600
    assert params["local_model"]["max_parallel_requests"] == 1


def test_the_slot_count_follows_the_loaded_backend(monkeypatch):
    assert inject(monkeypatch, recipe(), slots = 3)["local_model"]["max_parallel_requests"] == 3
    assert (
        inject(monkeypatch, recipe(), gguf = False, slots = 2)["local_model"]["max_parallel_requests"]
        == 2
    )


def test_a_safetensors_model_that_cannot_batch_gets_one_request_at_a_time(monkeypatch):
    params = inject(monkeypatch, recipe(), gguf = False, slots = 4, can_batch = False)

    assert params["local_model"]["max_parallel_requests"] == 1


def test_values_the_user_set_on_a_local_model_are_kept(monkeypatch):
    params = inject(
        monkeypatch, recipe(local_params = {"timeout": 30, "max_parallel_requests": 8}), slots = 1
    )

    assert params["local_model"]["timeout"] == 30
    assert params["local_model"]["max_parallel_requests"] == 8


def test_a_remote_model_keeps_data_designers_defaults(monkeypatch):
    params = inject(monkeypatch, recipe(remote_params = {"temperature": 0.5}), slots = 1)

    assert params["remote_model"] == {"temperature": 0.5}


def test_the_injected_values_are_valid_data_designer_inference_parameters(monkeypatch):
    from data_designer.config.models import ChatCompletionInferenceParams

    params = inject(monkeypatch, recipe(), slots = 2)["local_model"]
    parsed = ChatCompletionInferenceParams.model_validate(params)

    assert (parsed.timeout, parsed.max_parallel_requests) == (600, 2)
