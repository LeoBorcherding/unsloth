# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A video column must reach the vision collator, not train as text with the clip dropped."""

import pytest
from datasets import Dataset

from hub.utils.dataset_format import check_dataset_format
from utils.datasets import format_and_template_dataset

QUESTION = "What colour is the ball and which way does it move?"
ANSWERS = ["red moves left", "blue moves right"]
CLIPS = ["clips/clip00.mp4", "clips/clip01.mp4"]


def _messages_rows():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": clip},
                            {"type": "text", "text": QUESTION},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            }
            for clip, answer in zip(CLIPS, ANSWERS)
        ]
    )


def _flat_rows():
    return Dataset.from_dict({"video": CLIPS, "question": [QUESTION] * 2, "answer": ANSWERS})


def _format(ds):
    info = format_and_template_dataset(
        ds,
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
        tokenizer = None,
        is_vlm = True,
        dataset_name = "local",
    )
    assert info["success"], info["errors"]
    assert info["final_format"] == "vlm_messages"
    return list(info["dataset"])


def _video_parts(row):
    return [
        part
        for message in row["messages"]
        for part in message["content"]
        if part.get("type") == "video"
    ]


@pytest.mark.parametrize("rows", [_messages_rows, _flat_rows], ids = ["messages", "flat"])
def test_check_format_routes_video_to_the_vision_path(rows):
    result = check_dataset_format(rows(), is_vlm = True)
    # is_image is what the UI sends back as is_dataset_image, which selects the VLM trainer.
    assert result["is_image"] is True
    assert result["requires_manual_mapping"] is False
    assert result["detected_format"] in ("vlm_messages", "video_text")


def test_messages_rows_keep_their_video_parts():
    out = _format(_messages_rows())
    assert [_video_parts(row) for row in out] == [
        [{"type": "video", "video": clip}] for clip in CLIPS
    ]


def test_flat_video_rows_become_video_messages():
    out = _format(_flat_rows())
    for row, clip, answer in zip(out, CLIPS, ANSWERS):
        user, assistant = row["messages"]
        assert user["content"] == [
            {"type": "video", "video": clip},
            {"type": "text", "text": QUESTION},
        ]
        assert assistant["content"] == [{"type": "text", "text": answer}]


@pytest.mark.parametrize("clip", ["clips/a.webm", {"bytes": None, "path": "clips/a.webm"}], ids = ["str", "dict"])
def test_webm_in_a_video_column_is_video_not_audio(clip):
    ds = Dataset.from_list([{"video": clip, "answer": "red moves left"}] * 2)
    result = check_dataset_format(ds, is_vlm = True)
    assert result["is_image"] is True
    assert result["is_audio"] is False
    assert result["detected_format"] == "video_text"


def test_audio_in_a_column_named_video_stays_audio():
    ds = Dataset.from_list([{"video": {"bytes": None, "path": "a.wav"}, "text": "hello"}] * 2)
    result = check_dataset_format(ds, is_vlm = True)
    assert result["is_audio"] is True
    assert "video" not in (result["multimodal_columns"] or [])


def test_text_columns_named_like_video_stay_text():
    ds = Dataset.from_dict(
        {"video_id": ["abc", "def"], "video title": ["Cooking", "Hiking"], "text": ["a", "b"]}
    )
    result = check_dataset_format(ds, is_vlm = True)
    assert result["is_image"] is False


def _mlx_adapter():
    # worker.py pulls the whole CUDA/MLX import chain; the adapter itself is self-contained.
    import ast
    from pathlib import Path

    source = (Path(__file__).parents[1] / "core" / "training" / "worker.py").read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_adapt_for_mlx_vlm"
    )
    namespace = {"_resize_mlx_vlm_image": lambda image, *a, **k: image}
    exec(compile(ast.Module([node], []), "worker.py", "exec"), namespace)
    return namespace["_adapt_for_mlx_vlm"]


def test_mlx_refuses_video_rather_than_dropping_it():
    with pytest.raises(ValueError, match = "video"):
        _mlx_adapter()(_format(_messages_rows()))
