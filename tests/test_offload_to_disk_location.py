# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""offload_to_disk joined temporary_location with config._name_or_path; for a local model that path is
absolute, so the .pt landed in (and overwrote files in) the user's own model folder."""

import atexit
import gc
import os
import types

import pytest
import torch

from unsloth.models._utils import offload_to_disk


@pytest.fixture
def exit_hooks(monkeypatch):
    hooks = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: hooks.append((fn, a, k)))
    return hooks


def _model(name_or_path):
    return types.SimpleNamespace(config = types.SimpleNamespace(_name_or_path = name_or_path))


def test_local_model_folder_is_left_alone(tmp_path, exit_hooks):
    model_dir = tmp_path / "models" / "my-local-llama"
    model_dir.mkdir(parents = True)
    user_file = model_dir / "input_embeddings.pt"
    user_file.write_bytes(b"USER DATA")
    buffers = tmp_path / "buffers"

    W = offload_to_disk(torch.nn.Embedding(4, 2), _model(str(model_dir)), "input_embeddings", str(buffers))

    written = os.path.abspath(W._offloaded_file_location)
    assert user_file.read_bytes() == b"USER DATA"
    assert os.path.commonpath([written, str(model_dir)]) != str(model_dir)
    assert os.path.commonpath([written, str(buffers)]) == str(buffers)


@pytest.mark.parametrize("name", ["unsloth/llama-3.2-1b", "relative/local/model"])
def test_runs_on_the_same_model_do_not_share_a_file(tmp_path, exit_hooks, name):
    buffers = str(tmp_path / "buffers")
    first = offload_to_disk(torch.nn.Embedding(4, 2), _model(name), "input_embeddings", buffers)
    second = offload_to_disk(torch.nn.Embedding(4, 2), _model(name), "input_embeddings", buffers)
    assert first._offloaded_file_location != second._offloaded_file_location
    for W in (first, second):
        assert os.path.commonpath([os.path.abspath(W._offloaded_file_location), buffers]) == buffers


def test_buffer_dir_is_removed_at_exit(tmp_path, exit_hooks):
    W = offload_to_disk(torch.nn.Embedding(4, 2), _model("unsloth/x"), "input_embeddings", str(tmp_path))
    folder = os.path.dirname(os.path.abspath(W._offloaded_file_location))
    del W
    gc.collect()  # Windows can't delete a file that is still memory-mapped
    for fn, args, kwargs in exit_hooks:
        fn(*args, **kwargs)
    assert not os.path.exists(folder)
