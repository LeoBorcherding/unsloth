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


def test_a_dir_an_earlier_run_left_is_cleared(tmp_path, exit_hooks):
    # Windows keeps the .pt of a model still alive at exit, so the exit hook can't remove it.
    buffers = tmp_path / "buffers"
    stale = buffers / "my-model-abc123"
    stale.mkdir(parents = True)
    (stale / ".unsloth_offload").write_bytes(b"")
    (stale / "input_embeddings.pt").write_bytes(b"old")
    unmarked = buffers / "unsloth" / "llama-3.2-1b"
    unmarked.mkdir(parents = True)
    (unmarked / "input_embeddings.pt").write_bytes(b"not ours to judge")

    model = _model("unsloth/x")
    first = offload_to_disk(torch.nn.Embedding(4, 2), model, "input_embeddings", str(buffers))
    offload_to_disk(torch.nn.Embedding(4, 2), model, "output_embeddings", str(buffers))

    assert not stale.exists()
    assert (unmarked / "input_embeddings.pt").read_bytes() == b"not ours to judge"
    assert os.path.exists(first._offloaded_file_location)


def test_a_leftover_still_in_use_stays_marked(tmp_path, exit_hooks, monkeypatch):
    # Windows refuses to delete a .pt another process still maps; the dir must be retried later.
    from unsloth.models import _utils

    buffers = tmp_path / "buffers"
    stale = buffers / "my-model-abc123"
    stale.mkdir(parents = True)
    (stale / ".unsloth_offload").write_bytes(b"")
    (stale / "input_embeddings.pt").write_bytes(b"old")
    real_remove, real_unlink = os.remove, os.unlink

    def locked(remove):
        def wrapper(path, *args, **kwargs):
            if str(path).endswith("input_embeddings.pt"):
                raise PermissionError(32, "in use", str(path))
            return remove(path, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(os, "remove", locked(real_remove))
    monkeypatch.setattr(os, "unlink", locked(real_unlink))
    _utils._remove_stale_offload_dirs(str(buffers))
    assert (stale / ".unsloth_offload").exists()

    monkeypatch.setattr(os, "remove", real_remove)
    monkeypatch.setattr(os, "unlink", real_unlink)
    _utils._remove_stale_offload_dirs(str(buffers))
    assert not stale.exists()
