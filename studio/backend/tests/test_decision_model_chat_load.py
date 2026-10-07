import json

import pytest
from fastapi import HTTPException

import routes.inference as routes


def _laya(folder):
    (folder / "encoder").mkdir(parents = True)
    (folder / "tokenizer").mkdir()
    (folder / "rl_agent_config.json").write_text(json.dumps({}))
    (folder / "model.safetensors").write_bytes(b"")
    return folder


def _clef(folder):
    folder.mkdir(parents = True)
    for name in ("config.json", "joint_head_config.json"):
        (folder / name).write_text(json.dumps({}))
    (folder / "joint_head.safetensors").write_bytes(b"")
    return folder


@pytest.mark.parametrize("make", [_laya, _clef])
def test_decision_checkpoint_is_refused_for_chat(tmp_path, make):
    folder = make(tmp_path / "decision")
    with pytest.raises(HTTPException) as err:
        routes._refuse_decision_model(str(folder))
    assert err.value.status_code == 400
    assert "Decision API" in err.value.detail


def test_cached_hub_laya_is_refused(tmp_path, monkeypatch):
    import utils.models.model_config as model_config
    import utils.utils as utils

    snapshot = tmp_path / "snapshot"
    _laya(snapshot / "multilingual")
    monkeypatch.setattr(utils, "hf_cache_snapshot_dir", lambda _repo: snapshot)
    monkeypatch.setattr(model_config, "cache_reads_authorized", lambda *a, **k: True)
    with pytest.raises(HTTPException):
        routes._refuse_decision_model("convaiinnovations/laya")


def test_plain_model_passes(tmp_path):
    folder = tmp_path / "llm"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (folder / "model.safetensors").write_bytes(b"")
    routes._refuse_decision_model(str(folder), None, "")
