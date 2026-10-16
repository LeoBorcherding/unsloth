# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The patched push_to_hub retries without `tags` only when `tags` itself was rejected."""

import pytest

from unsloth.save import patch_saving_functions


def _model(push_to_hub):
    class _Model:
        pass

    _Model.push_to_hub = push_to_hub
    model = _Model()
    model.calls = []
    patch_saving_functions(model)
    return model


def test_ctrl_c_does_not_restart_the_upload():
    def push_to_hub(self, repo_id, tags = None):
        """stub"""
        self.calls.append(tags)
        raise KeyboardInterrupt

    model = _model(push_to_hub)
    with pytest.raises(KeyboardInterrupt):
        model.push_to_hub("u/m", tags = ["x"])
    assert len(model.calls) == 1


def test_real_error_is_not_masked_when_tags_is_not_a_parameter():
    def push_to_hub(self, repo_id):
        """stub"""
        self.calls.append(repo_id)
        raise ConnectionError("401 Unauthorized")

    model = _model(push_to_hub)
    with pytest.raises(ConnectionError):
        model.push_to_hub("u/m")
    assert len(model.calls) == 1


def test_unrelated_type_error_is_raised_once():
    def push_to_hub(self, repo_id, tags = None):
        """stub"""
        self.calls.append(tags)
        raise TypeError("expected str, got int")

    model = _model(push_to_hub)
    with pytest.raises(TypeError, match = "expected str"):
        model.push_to_hub("u/m", tags = ["x"])
    assert len(model.calls) == 1


def test_rejected_tags_are_dropped_and_retried():
    def push_to_hub(self, repo_id, tags = None):
        """stub"""
        self.calls.append(tags)
        if tags is not None:
            raise TypeError("got an unexpected keyword argument 'tags'")

    model = _model(push_to_hub)
    model.push_to_hub("u/m", tags = ["x"])
    assert model.calls == [["x", "unsloth"], None]
