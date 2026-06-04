# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for efficiency benchmarks (bench.py, profile_breakdown.py)."""

import json
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
RULER_SCRIPT_DIR = SCRIPT_DIR.parent / "RULER" / "scripts"
RULER_PREPARE_PY = RULER_SCRIPT_DIR / "data" / "prepare.py"
RULER_DATA_ROOT = RULER_SCRIPT_DIR / "results" / "data"
RULER_DEFAULT_TASK = "niah_single_3"
RULER_TEMPLATE_TYPE = "minicpm4"


def parse_seq_len(s: str) -> int:
    """Parse '16K', '128K', '131072', etc. into an integer."""
    s = s.strip().upper()
    if s.endswith("K"):
        return int(s[:-1]) * 1024
    return int(s)


def prepare_input_batches(
    dataset_name: str,
    dataset_split: str,
    tokenizer,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    model_path: str = None,
) -> list:
    """Load dataset, select qualifying samples, and build input batches.

    Both bench.py and profile_breakdown.py call this to guarantee identical
    inputs for the same (dataset, tokenizer, B, L, n_batches) arguments.

    Args:
        dataset_name: HuggingFace dataset name (e.g. ``"emozilla/pg19"``) or
            the special source ``"ruler"`` / ``"ruler_<task>"`` for exact
            RULER prompt text.
        dataset_split: Dataset split (e.g. ``"test"``).
        tokenizer: A HuggingFace tokenizer.
        batch_size: Number of samples per batch (B).
        seq_len: Target input length (L); only samples with >= L tokens qualify.
        n_batches: Total number of batches to produce (typically 1 warmup + N timed).
        model_path: Model/tokenizer path. Required for the special ``"ruler"`` source
            so data preparation uses the same tokenizer setup as RULER.

    Returns:
        List of ``n_batches`` tensors, each of shape ``(batch_size, seq_len)``
        on CUDA.
    """
    source_name = dataset_name.strip().lower()
    if _is_ruler_dataset(source_name):
        if not model_path:
            raise ValueError("model_path is required when dataset_name is 'ruler' or 'ruler_<task>'")
        return _prepare_ruler_input_batches(
            dataset_name=source_name,
            tokenizer=tokenizer,
            model_path=model_path,
            batch_size=batch_size,
            seq_len=seq_len,
            n_batches=n_batches,
        )
    dataset = load_dataset(dataset_name)[dataset_split]["text"]

    qualifying: list[torch.Tensor] = []
    for text in dataset:
        ids = tokenizer(text, return_tensors="pt").to("cuda").input_ids
        if ids.shape[1] >= seq_len:
            qualifying.append(ids[:, :seq_len])  # (1, L)
        if len(qualifying) >= n_batches * batch_size:
            break

    if not qualifying:
        raise RuntimeError(
            f"No sample in dataset {dataset_name!r} [{dataset_split}] "
            f"with length >= {seq_len}"
        )

    input_batches: list[torch.Tensor] = []
    for b_idx in range(n_batches):
        batch_ids = [
            qualifying[(b_idx * batch_size + s) % len(qualifying)]
            for s in range(batch_size)
        ]
        input_batches.append(torch.cat(batch_ids, dim=0))  # (B, L)

    return input_batches


def _resolve_tokenizer_args(model_path: str) -> tuple[str, str]:
    return model_path, "hf"


def _is_ruler_dataset(dataset_name: str) -> bool:
    return dataset_name == "ruler" or dataset_name.startswith("ruler_")


def _resolve_ruler_task_name(dataset_name: str) -> str:
    if dataset_name == "ruler":
        return RULER_DEFAULT_TASK
    task_name = dataset_name[len("ruler_"):]
    if not task_name:
        raise ValueError("dataset_name='ruler_' is invalid; expected ruler_<task>")
    return task_name


def _ruler_data_dir(seq_len: int, num_samples: int) -> Path:
    return RULER_DATA_ROOT / f"{seq_len}_n{num_samples}"


def _ruler_validation_file(task_name: str, seq_len: int, num_samples: int) -> Path:
    return _ruler_data_dir(seq_len, num_samples) / task_name / "validation.jsonl"


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _ensure_ruler_inputs(task_name: str, model_path: str, seq_len: int, num_samples: int) -> Path:
    validation_file = _ruler_validation_file(task_name, seq_len, num_samples)
    if _count_jsonl_lines(validation_file) >= num_samples:
        return validation_file

    tokenizer_path, tokenizer_type = _resolve_tokenizer_args(model_path)
    cmd = [
        sys.executable,
        str(RULER_PREPARE_PY),
        "--save_dir", str(_ruler_data_dir(seq_len, num_samples)),
        "--benchmark", "synthetic",
        "--task", task_name,
        "--tokenizer_path", tokenizer_path,
        "--tokenizer_type", tokenizer_type,
        "--max_seq_length", str(seq_len),
        "--model_template_type", RULER_TEMPLATE_TYPE,
        "--num_samples", str(num_samples),
    ]
    subprocess.run(cmd, check=True)

    if _count_jsonl_lines(validation_file) < num_samples:
        raise RuntimeError(
            f"RULER data preparation did not produce {num_samples} samples at {validation_file}"
    )
    return validation_file


def _load_ruler_inputs(
    model_path: str,
    dataset_name: str,
    seq_len: int,
    num_samples: int,
) -> list[str]:
    task_name = _resolve_ruler_task_name(dataset_name)
    validation_file = _ensure_ruler_inputs(task_name, model_path, seq_len, num_samples)
    prompts: list[str] = []
    with validation_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = payload.get("input")
            if prompt:
                prompt += payload.get("answer_prefix", "")
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= num_samples:
                break
    if len(prompts) < num_samples:
        raise RuntimeError(
            f"Expected {num_samples} RULER prompts in {validation_file}, found {len(prompts)}"
        )
    return prompts


def _prepare_ruler_input_batches(
    dataset_name: str,
    tokenizer,
    model_path: str,
    batch_size: int,
    seq_len: int,
    n_batches: int,
) -> list:
    num_needed = n_batches * batch_size
    prompts = _load_ruler_inputs(
        model_path=model_path,
        dataset_name=dataset_name,
        seq_len=seq_len,
        num_samples=num_needed,
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError(
            "RULER efficiency input padding requires tokenizer.pad_token_id or tokenizer.eos_token_id"
        )

    padded_inputs: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
        prompt_len = int(ids.shape[1])
        prompt_lengths.append(prompt_len)
        if prompt_len > seq_len:
            raise RuntimeError(
                f"RULER prompt length {prompt_len} exceeds requested seq_len={seq_len} "
                f"for dataset {dataset_name!r}"
            )
        if prompt_len < seq_len:
            pad = torch.full(
                (1, seq_len - prompt_len),
                pad_id,
                dtype=ids.dtype,
                device=ids.device,
            )
            ids = torch.cat([pad, ids], dim=1)
        padded_inputs.append(ids)

    unique_lengths = sorted(set(prompt_lengths))
    if unique_lengths != [seq_len]:
        print(
            f"[Efficiency] Left-padding RULER prompts from lengths {unique_lengths} "
            f"to requested seq_len={seq_len}"
        )

    input_batches: list[torch.Tensor] = []
    for b_idx in range(n_batches):
        batch_ids = [
            padded_inputs[b_idx * batch_size + s]
            for s in range(batch_size)
        ]
        input_batches.append(torch.cat(batch_ids, dim=0))
    return input_batches
