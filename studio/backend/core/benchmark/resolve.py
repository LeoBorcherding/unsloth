# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from pathlib import Path
from typing import Optional

from utils.paths import is_local_path, get_cache_path
from hub.utils.gguf import (
    list_gguf_variants_from_hf_cache,
    resolve_local_gguf_path,
    list_local_gguf_variants,
)
from loggers import get_logger

logger = get_logger(__name__)


def resolve_tokenizer(model_id: str) -> str:
    candidates = [model_id]
    gguf_suffixes = ["-GGUF", "_GGUF", "-gguf", "_gguf", ".gguf"]
    for suffix in gguf_suffixes:
        if model_id.endswith(suffix):
            candidates.append(model_id[: -len(suffix)])
            parts = model_id.rsplit("/", 1)
            if len(parts) == 2:
                base_part = parts[1]
                for s in gguf_suffixes:
                    if base_part.endswith(s):
                        candidates.append(f"{parts[0]}/{base_part[: -len(s)]}")
                        break
            break

    candidates.extend(["gpt2", "t5-small"])
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(c, trust_remote_code = True)
            return c
        except Exception:
            continue
    return "gpt2"


def build_local_completions_kwargs(
    identifier: str,
    tokenizer: str,
    base_url: str,
    task: str,
    batch_size: str,
    log_samples: bool,
    max_tokens: Optional[int] = None,
    num_fewshot: Optional[int] = None,
    output_path: Optional[str] = None,
) -> dict:
    completions_url = f"{base_url}/v1/completions"
    kwargs = {
        "model": "local-completions",
        "model_args": {
            "model": identifier,
            "base_url": completions_url,
            "tokenizer": tokenizer,
            "tokenizer_backend": "huggingface",
            "tokenized_requests": False,
            "max_gen_toks": max_tokens if max_tokens is not None else 32768,
            "num_concurrent": int(batch_size) if batch_size != "auto" else 1,
            "timeout": 1200,
            "max_retries": 5,
        },
        "tasks": [task],
        "batch_size": 1,
        "log_samples": log_samples,
    }
    if num_fewshot is not None:
        kwargs["num_fewshot"] = num_fewshot
    if output_path:
        kwargs["output_path"] = output_path
    return kwargs


def resolve_model_details(
    checkpoint_path: str,
    hf_token: Optional[str],
    gguf_variant: Optional[str] = None,
    task: str = "mmlu",
    server_url: Optional[str] = None,
    batch_size: str = "auto",
    log_samples: bool = True,
    num_fewshot: Optional[int] = None,
    output_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> tuple[list[str], dict | None]:
    """Resolve model details and return (empty_log_lines, lm_eval_kwargs or None)."""
    identifier = checkpoint_path.strip()
    is_gguf_model = False
    model_variant: Optional[str] = None
    gguf_file_on_disk: Optional[str] = None
    user_variant = gguf_variant

    local = is_local_path(identifier)

    # Imported lazily: ModelConfig transitively imports transformers, which is
    # heavy. Benchmarks never load a model in-process (they use the
    # local-completions HTTP backend), so we avoid pinning transformers into the
    # web process at startup.
    try:
        from utils.models.model_config import ModelConfig
        cfg = ModelConfig.from_identifier(identifier, hf_token = hf_token, gguf_variant = user_variant)
        if cfg is not None:
            is_gguf_model = cfg.is_gguf
            model_variant = cfg.gguf_variant
            if cfg.gguf_file and os.path.isfile(cfg.gguf_file):
                gguf_file_on_disk = cfg.gguf_file
    except Exception as e:
        logger.debug("ModelConfig.from_identifier failed for %s: %s", identifier, e)

    if not local:
        try:
            cache_path = get_cache_path(identifier)
            if cache_path and cache_path.exists():
                try:
                    vfc = list_gguf_variants_from_hf_cache(identifier)
                    if vfc is not None:
                        is_gguf_model = True
                except Exception as e:
                    logger.debug("list_gguf_variants_from_hf_cache failed for %s: %s", identifier, e)

                if is_gguf_model:
                    try:
                        resolved = resolve_local_gguf_path(identifier, model_variant)
                        if not resolved:
                            resolved = resolve_local_gguf_path(identifier, None)
                        if resolved:
                            gguf_file_on_disk = str(Path(resolved).resolve())
                    except Exception as e:
                        logger.debug("resolve_local_gguf_path failed for %s: %s", identifier, e)
        except Exception as e:
            logger.debug("get_cache_path failed for %s: %s", identifier, e)

    if local and os.path.isdir(identifier):
        try:
            vl = list_local_gguf_variants(identifier)
            if vl is not None:
                is_gguf_model = True
                if vl[0]:
                    first = Path(identifier) / vl[0][0].filename
                    gguf_file_on_disk = str(first.resolve())
        except Exception as e:
            logger.debug("list_local_gguf_variants failed for %s: %s", identifier, e)

    lm_eval_kwargs: dict | None = None

    base_url = (server_url or os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")).rstrip("/")

    # Benchmarks always talk to a local inference server via lm_eval's
    # `local-completions` backend (HTTP API calls); we never load the model
    # in-process via the `hf` backend (which would need transformers to
    # instantiate the architecture and is unsupported on macOS).
    tokenizer_source = gguf_file_on_disk if (is_gguf_model and gguf_file_on_disk) else identifier
    tokenizer = resolve_tokenizer(tokenizer_source)
    lm_eval_kwargs = build_local_completions_kwargs(
        identifier, tokenizer, base_url, task, batch_size, log_samples,
        max_tokens = max_tokens, num_fewshot = num_fewshot, output_path = output_path,
    )

    return [], lm_eval_kwargs
