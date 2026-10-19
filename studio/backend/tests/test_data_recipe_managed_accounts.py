# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from core.training import account_jobs
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("alice-recipe-account", "alice")
MODEL = "unsloth/gemma-4-E2B-it-GGUF"


@pytest.fixture(autouse = True)
def multi(monkeypatch):
    from utils import managed_provider_url_settings

    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(account_jobs, "_retired", set())
    # The owner's "allow private provider URLs" switch, off: the case the audit hit.
    monkeypatch.setattr(
        managed_provider_url_settings, "get_managed_private_provider_urls_allowed", lambda: False
    )


def local_recipe(provider = None):
    return {
        "model_providers": [provider or {"name": "local", "is_local": True}],
        "model_configs": [{"alias": "local_model", "model": MODEL, "provider": "local"}],
        "columns": [
            {
                "column_type": "llm-text",
                "name": "question",
                "model_alias": "local_model",
                "prompt": "Write a question.",
            }
        ],
    }


def check(recipe, account = ALICE):
    from models.data_recipe import RecipePayload
    from routes.data_recipe import validate as route
    return run_as(account, route.validate, RecipePayload(recipe = recipe))


def test_check_accepts_the_local_provider_for_a_managed_account():
    pytest.importorskip("data_designer")
    owner = check(local_recipe(), OWNER)
    managed = check(local_recipe())

    assert owner.valid, owner
    assert managed.valid, managed


def test_check_still_refuses_a_typed_loopback_endpoint_for_a_managed_account():
    pytest.importorskip("data_designer")
    typed = {"name": "local", "endpoint": "http://127.0.0.1", "provider_type": "openai"}
    result = check(local_recipe(typed))

    assert not result.valid
    assert "HTTPS" in (result.raw_detail or "")


def test_a_public_hf_seed_needs_no_token_for_a_managed_account():
    source = {"seed_type": "hf", "path": "datasets/openai/gsm8k/main/train-*.parquet"}

    run_as(ALICE, account_jobs.validate_recipe_access, {"seed_config": {"source": source}})


def test_a_github_seed_still_needs_the_managed_accounts_own_token():
    source = {"seed_type": "github_repo", "repos": ["unslothai/unsloth"], "token": ""}

    with pytest.raises(HTTPException) as caught:
        run_as(ALICE, account_jobs.validate_recipe_access, {"seed_config": {"source": source}})
    assert caught.value.status_code == 403


@pytest.mark.parametrize(
    ("account", "token", "expected"),
    [(ALICE, None, False), (ALICE, "alice-token", "alice-token"), (OWNER, None, None)],
)
def test_a_managed_accounts_tokenless_hf_seed_never_uses_the_hosts_token(
    monkeypatch, account, token, expected
):
    pytest.importorskip("data_designer")
    from fsspec.implementations.memory import MemoryFileSystem
    from data_designer.engine.resources import seed_reader

    from core.data_recipe import service

    seen = []

    def fake_hffs(*, endpoint, token, skip_instance_cache):
        seen.append(token)
        return MemoryFileSystem()

    monkeypatch.setattr(seed_reader, "HfFileSystem", fake_hffs)
    monkeypatch.setattr(
        seed_reader.HuggingFaceSeedReader,
        "create_duckdb_connection",
        seed_reader.HuggingFaceSeedReader.create_duckdb_connection,
    )
    service._apply_data_designer_hf_anonymous_seed_patch()

    reader = object.__new__(seed_reader.HuggingFaceSeedReader)
    reader.source = SimpleNamespace(token = token, endpoint = "https://huggingface.co")
    reader.secret_resolver = SimpleNamespace(resolve = lambda value: value)
    run_as(account, reader.create_duckdb_connection).close()

    assert seen == [expected]


def resident_hidden_run(monkeypatch, *, model, hidden):
    from hub.services.models import account_access
    from routes.data_recipe import jobs as route

    monkeypatch.setattr(route, "_loaded_local_model_identity", lambda: (True, MODEL, ""))
    monkeypatch.setitem(
        sys.modules, "routes.inference", SimpleNamespace(_loaded_slot_ident = lambda: MODEL)
    )
    monkeypatch.setattr(account_access, "resident_hidden", lambda *a: hidden)
    recipe = local_recipe()
    recipe["model_configs"][0]["model"] = model
    return lambda account: run_as(
        account, route._ensure_selected_local_model_loaded, recipe, {"local"}
    )


def test_an_unnamed_local_model_says_why_when_the_loaded_one_is_not_shared(monkeypatch):
    run = resident_hidden_run(monkeypatch, model = "", hidden = True)

    with pytest.raises(ValueError, match = "not shared with this account.*local_model"):
        run(ALICE)


@pytest.mark.parametrize(
    ("account", "model", "hidden"),
    [(OWNER, "", True), (ALICE, "", False), (ALICE, MODEL, True)],
)
def test_an_unnamed_local_model_passes_when_it_would_be_served(monkeypatch, account, model, hidden):
    resident_hidden_run(monkeypatch, model = model, hidden = hidden)(account)
