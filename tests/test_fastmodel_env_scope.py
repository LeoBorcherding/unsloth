# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel's per-family compile and static-generation switches end with the load; the model keeps its own."""

import inspect
import os
import types

import pytest

import unsloth.models._block_compile as block_compile
import unsloth.models.vision as vision
from unsloth.models.loader_utils import _restore_load_scoped_env

SWITCHES = {"UNSLOTH_COMPILE_DISABLE": "partial", "UNSLOTH_DISABLE_STATIC_GENERATION": "1"}


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.delenv("UNSLOTH_BLOCK_COMPILE", raising = False)


def _model(load_env = None):
    config = types.SimpleNamespace(
        model_type = "qwen3_5", text_config = types.SimpleNamespace(model_type = "qwen3_5_text")
    )
    model = types.SimpleNamespace(config = config, generation_config = types.SimpleNamespace())
    if load_env is not None:
        model._unsloth_load_env = load_env
    return model


def test_the_next_load_does_not_inherit_a_familys_switches():
    @_restore_load_scoped_env
    def load():  # e.g. the CSM branch
        os.environ.update(SWITCHES)
        return "model"

    assert load() == "model"
    assert [name for name in SWITCHES if name in os.environ] == []


def test_a_users_own_setting_survives_the_load(monkeypatch):
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")

    @_restore_load_scoped_env
    def load():
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "partial"

    load()
    assert os.environ["UNSLOTH_COMPILE_DISABLE"] == "1"


@pytest.fixture
def decode_compile_available(monkeypatch):
    monkeypatch.setattr(vision, "unsloth_decode_compile", object())
    monkeypatch.setattr(vision, "_zoo_module_available", lambda name: True)
    monkeypatch.setenv("UNSLOTH_COMPILE_DECODE", "1")


def test_decode_compile_follows_the_models_own_load(decode_compile_available):
    assert not vision._compiles_decode(_model({"UNSLOTH_COMPILE_DISABLE": "partial"}))
    assert vision._compiles_decode(_model({"UNSLOTH_COMPILE_DISABLE": None}))
    assert vision._compiles_decode(_model())


def test_block_compile_follows_the_models_own_load():
    assert not block_compile._enabled(_model({"UNSLOTH_COMPILE_DISABLE": "partial"}))
    assert block_compile._enabled(_model({"UNSLOTH_COMPILE_DISABLE": None}))
    assert block_compile._enabled()
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    assert not block_compile._enabled(_model({"UNSLOTH_COMPILE_DISABLE": None}))


def test_static_generation_follows_the_models_own_load():
    from unsloth.models.loader_utils import _load_env_values

    name = "UNSLOTH_DISABLE_STATIC_GENERATION"
    assert _load_env_values(_model({name: "1"}), name) != {"0"}
    assert _load_env_values(_model({name: None}), name) == {"0"}
    src = inspect.getsource(vision.unsloth_base_fast_generate)
    assert f'_load_env_values(self, "{name}") == {{"0"}}' in src
    assert f'os.environ.get("{name}"' not in src


def test_the_load_stamps_both_switches_on_the_model():
    src = inspect.getsource(vision.FastBaseModel.from_pretrained)
    assert "model._unsloth_load_env = {name: os.environ.get(name) for name in MODEL_SCOPED_ENV_VARS}" in src
    from unsloth.models.loader_utils import LOAD_SCOPED_ENV_VARS, MODEL_SCOPED_ENV_VARS

    assert set(SWITCHES) == set(MODEL_SCOPED_ENV_VARS) <= set(LOAD_SCOPED_ENV_VARS)
