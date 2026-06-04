#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RULER benchmark runner for NOSA / MiniCPM / dense baselines.

Thinking mode is OFF: greedy decoding (temperature=0) with no <think> prompting.

Dataset reuse:
    Synthetic datasets only depend on the tokenizer, sequence length, and
    number of samples -- they are independent of the model mode (dense /
    SparDA / NOSA) or indexer checkpoint.  Data is therefore stored in a
    shared directory (results/data/<seq_len>_n<num_samples>/) and reused
    across runs automatically.  Predictions are always regenerated and
    stored under a mode-specific directory (results/<model>/<mode>/...).

Usage:
    # NOSA 8B (auto-detects modeling_llama_nosa)
    python run.py --model-path openbmb/NOSA-8B --sparda --indexer-path /path/to/indexer_weights.pt

    # MiniCPM 8B with SparDA indexer
    python run.py --model-path openbmb/MiniCPM4.1-8B --sparda --indexer-path /path/to/indexer_weights.pt

    # MiniCPM 8B with long-context (128K)
    python run.py --model-path openbmb/MiniCPM4.1-8B --sparda --indexer-path /path/to/indexer_weights.pt --long-context --seq-len 131072

    # Dense baseline (reuses data from any previous run at same seq-len)
    python run.py --model-path openbmb/MiniCPM4.1-8B

    # Multi-GPU (auto-detected -- runs 13 tasks across all visible GPUs)
    python run.py --model-path openbmb/NOSA-8B --sparda --indexer-path /path/to/indexer_weights.pt

    # Explicit GPU selection (4 GPUs)
    python run.py --model-path openbmb/NOSA-8B --sparda --indexer-path /path/to/indexer_weights.pt --gpus 0,1,2,3

Multi-GPU parallelization:
    When multiple GPUs are available (auto-detected via CUDA_VISIBLE_DEVICES
    or nvidia-smi), one long-lived prediction worker is launched per GPU.
    Large RULER tasks are split into sample chunks and workers pull those
    chunks from a shared queue, so each worker can reuse one loaded model
    while still achieving better load balance on 4-8 GPUs. Per-task logs are
    saved to <pred_dir>/logs/<task>.log. Use --gpus to override GPU selection.

All arguments are Python-style (--flag / --key value). The shell wrapper run.sh
sets up the environment and forwards arguments here.
"""

import argparse
import fcntl
import importlib
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import Counter, deque
from pathlib import Path

import yaml

os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_{os.getpid()}")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent.parent
JSON_DIR = SCRIPT_DIR / "data" / "synthetic" / "json"

benchmarks_dir_str = str(BENCHMARKS_DIR)
if benchmarks_dir_str not in sys.path:
    sys.path.insert(0, benchmarks_dir_str)


def _print_block_hit_rate_summary(prefix: str, label: str, stats: dict):
    if not stats or stats.get("total_blocks", 0) <= 0:
        return
    hit_rate = float(stats.get("hit_rate", 0.0)) * 100.0
    miss_blocks = int(stats.get("miss_blocks", 0))
    total_blocks = int(stats.get("total_blocks", 0))
    print(f"[{prefix}] {label}: block hit-rate {hit_rate:.1f}% ({total_blocks - miss_blocks}/{total_blocks})")


def _print_saved_block_hit_rate_summaries(prefix: str, pred_dir: Path, task_names: list):
    missing = []
    for task_name in task_names:
        stats_path = pred_dir / f"{task_name}_block_stats.json"
        if not stats_path.exists():
            missing.append(task_name)
            continue
        try:
            stats = json.loads(stats_path.read_text())
        except json.JSONDecodeError:
            missing.append(task_name)
            continue
        if stats and stats.get("total_blocks", 0) > 0:
            _print_block_hit_rate_summary(prefix, task_name, stats)
        else:
            missing.append(task_name)
    if missing:
        print(f"[{prefix}] WARNING: missing block hit-rate stats for {missing}")

# Maps haystack type / dataset name -> (filename, download function name)
_DATA_FILE_URLS = {
    "squad.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "hotpotqa.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
}

ALL_SYNTHETIC_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
    "qa_1", "qa_2",
]

_TASK_CONFIG_CACHE = {}


def _detect_gpus(gpu_arg: str) -> list:
    """Return a list of GPU ID strings for CUDA_VISIBLE_DEVICES.

    - ``"auto"`` (default): respects CUDA_VISIBLE_DEVICES if already set,
      otherwise queries ``nvidia-smi``, falling back to ``["0"]``.
    - Comma-separated IDs (e.g. ``"0,2,5"``): uses those specific GPUs.
    """
    if gpu_arg != "auto":
        return [g.strip() for g in gpu_arg.split(",") if g.strip()]

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip():
        return [g.strip() for g in cuda_visible.split(",") if g.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [g.strip() for g in result.stdout.strip().split("\n") if g.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return ["0"]


def _download_url(url: str, dest: Path):
    """Download a single file via urllib."""
    print(f"[RULER] Downloading {dest.name} from {url} ...")
    temp_dest = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    try:
        urllib.request.urlretrieve(url, str(temp_dest))
        temp_dest.replace(dest)
    finally:
        try:
            temp_dest.unlink()
        except FileNotFoundError:
            pass
    print(f"[RULER] Saved {dest}")


def _download_paul_graham_essays():
    """Generate PaulGrahamEssays.json using the full download script.

    Requires ``html2text`` and ``beautifulsoup4``.  Raises an error if the
    script fails rather than silently falling back to a partial download.
    """
    dest = JSON_DIR / "PaulGrahamEssays.json"
    script = JSON_DIR / "download_paulgraham_essay.py"

    if not script.exists():
        raise FileNotFoundError(
            f"Cannot find {script}. Please download PaulGrahamEssays.json manually."
        )

    # Auto-install missing dependencies into the current Python environment
    missing_pkgs = []
    for pkg_import, pkg_pip in [("html2text", "html2text"), ("bs4", "beautifulsoup4")]:
        try:
            __import__(pkg_import)
        except ImportError:
            missing_pkgs.append(pkg_pip)
    if missing_pkgs:
        import site
        target = site.getsitepackages()[0]
        print(f"[RULER] Installing missing dependencies: {missing_pkgs} into {target} ...")
        subprocess.run(
            ["pip", "install", f"--target={target}"] + missing_pkgs,
            check=True,
        )

    print(f"[RULER] Running {script.name} ...")
    result = subprocess.run(
        [sys.executable, str(script)], cwd=str(JSON_DIR),
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not dest.exists():
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"Failed to generate PaulGrahamEssays.json.\n"
            f"  Script: {script}\n"
            f"  Exit code: {result.returncode}\n"
            f"  Error: {stderr}\n\n"
            f"This usually means 'html2text' or 'beautifulsoup4' could not be installed.\n"
            f"Fix:  python -m ensurepip && python -m pip install html2text beautifulsoup4"
        )
    print(f"[RULER] Saved {dest}")


def _obtain_english_words():
    """Get english_words.json by generating from nltk."""
    dest = JSON_DIR / "english_words.json"
    temp_dest = dest.with_name(f"{dest.name}.{os.getpid()}.part")

    print("[RULER] Generating english_words.json from nltk corpus ...")
    try:
        import nltk
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
        word_list = sorted(set(nltk_words.words()))
        mapping = {str(i): w for i, w in enumerate(word_list)}
        with open(temp_dest, "w") as f:
            json.dump(mapping, f)
        temp_dest.replace(dest)
        print(f"[RULER] Saved {dest} ({len(mapping)} words)")
    except Exception as e:
        raise RuntimeError(
            f"Cannot obtain english_words.json: {e}\n"
            f"Install nltk and download the 'words' corpus, or place "
            f"english_words.json at: {dest}"
        )


def ensure_data_files(tasks: list):
    """Detect and download any missing data files required by *tasks*."""
    with open(SCRIPT_DIR / "synthetic.yaml") as f:
        task_configs = yaml.safe_load(f)

    needed = set()
    for task in tasks:
        cfg = task_configs.get(task, {})
        task_type = cfg.get("task")
        task_args = cfg.get("args", {})

        if task_type in ("niah", "variable_tracking") and task_args.get("type_haystack") == "essay":
            needed.add("PaulGrahamEssays.json")
        elif task_type == "common_words_extraction":
            needed.add("english_words.json")
        elif task_type == "qa":
            ds = task_args.get("dataset")
            if ds == "squad":
                needed.add("squad.json")
            elif ds == "hotpotqa":
                needed.add("hotpotqa.json")

    missing = [f for f in needed if not (JSON_DIR / f).exists()]
    if not missing:
        return

    print(f"[RULER] Missing data files: {missing}")
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = JSON_DIR / ".data_files.lock"

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        missing = [f for f in needed if not (JSON_DIR / f).exists()]
        if not missing:
            print("[RULER] All data files ready.\n")
            return

        for filename in missing:
            if filename == "PaulGrahamEssays.json":
                _download_paul_graham_essays()
            elif filename == "english_words.json":
                _obtain_english_words()
            elif filename in _DATA_FILE_URLS:
                _download_url(_DATA_FILE_URLS[filename], JSON_DIR / filename)
            else:
                raise FileNotFoundError(f"No download source for {filename}")

    still_missing = [f for f in missing if not (JSON_DIR / f).exists()]
    if still_missing:
        raise RuntimeError(f"Failed to obtain data files: {still_missing}")
    print("[RULER] All data files ready.\n")


def _load_task_configs(benchmark: str):
    if benchmark in _TASK_CONFIG_CACHE:
        return _TASK_CONFIG_CACHE[benchmark]

    script_dir_str = str(SCRIPT_DIR)
    if script_dir_str not in sys.path:
        sys.path.insert(0, script_dir_str)

    module = importlib.import_module(f"data.{benchmark}.constants")
    with open(SCRIPT_DIR / f"{benchmark}.yaml") as f:
        tasks_customized = yaml.safe_load(f)

    merged = {}
    for task_name, task_cfg in tasks_customized.items():
        config = dict(task_cfg)
        config.update(module.TASKS[config["task"]])
        merged[task_name] = config

    _TASK_CONFIG_CACHE[benchmark] = merged
    return merged


def _task_weight(task_configs, task_name: str, num_samples: int) -> int:
    config = task_configs.get(task_name, {})
    tokens = int(config.get("tokens_to_generate", 128) or 128)
    return max(1, tokens) * max(1, num_samples)


def _count_nonempty_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except FileNotFoundError:
        return 0


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_lines_atomic(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line if line.endswith("\n") else line + "\n")
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    temp_path.replace(path)


def _split_ranges(total_size: int, num_parts: int):
    if total_size <= 0 or num_parts <= 1:
        return [(0, total_size)]

    ranges = []
    base = total_size // num_parts
    extra = total_size % num_parts
    start = 0
    for idx in range(num_parts):
        size = base + (1 if idx < extra else 0)
        if size <= 0:
            continue
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges or [(0, total_size)]


def _task_data_file(data_dir: Path, task_name: str) -> Path:
    return data_dir / task_name / "validation.jsonl"


def _task_chunk_count(task_configs, task_name: str, total_samples: int,
                      target_chunk_weight: int, num_gpus: int) -> int:
    if num_gpus < 2 or total_samples <= 1:
        return 1

    if target_chunk_weight <= 0:
        return 1

    desired = (_task_weight(task_configs, task_name, total_samples) + target_chunk_weight - 1) // target_chunk_weight
    return max(1, min(total_samples, desired))


def resolve_tokenizer(model_path: str):
    """Auto-detect tokenizer path and type from model directory."""
    return model_path, "hf"


def _data_ready(data_dir: Path, task: str, num_samples: int) -> bool:
    """True if the data jsonl for *task* already has *num_samples* lines."""
    f = data_dir / task / "validation.jsonl"
    if not f.exists():
        return False
    with open(f) as fh:
        return sum(1 for _ in fh) >= num_samples



def run_prepare(args, data_dir: Path, task: str, max_seq_length: int):
    tokenizer_path, tokenizer_type = resolve_tokenizer(args.model_path)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "data" / "prepare.py"),
        "--save_dir", str(data_dir),
        "--benchmark", "synthetic",
        "--task", task,
        "--tokenizer_path", tokenizer_path,
        "--tokenizer_type", tokenizer_type,
        "--max_seq_length", str(max_seq_length),
        "--model_template_type", args.template_type,
        "--num_samples", str(args.num_samples),
    ]
    subprocess.run(cmd, check=True)


def ensure_task_data(args, data_dir: Path, task: str, max_seq_length: int):
    lock_path = data_dir / f".{task}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        if not args.force and _data_ready(data_dir, task, args.num_samples):
            print(f"--- {task} @ {max_seq_length} --- (data cached)")
            return

        existing = data_dir / task / "validation.jsonl"
        if existing.exists():
            existing.unlink()
        print(f"--- {task} @ {max_seq_length} --- (generating data)")
        run_prepare(args, data_dir, task, max_seq_length)

        if not _data_ready(data_dir, task, args.num_samples):
            raise RuntimeError(
                f"[RULER] Data preparation failed for task={task}, seq_length={max_seq_length}"
            )


def _build_predict_cmd(args, data_dir: Path, pred_dir: Path, tasks,
                       max_seq_length: int, progress_file: Path = None,
                       results_file: Path = None, log_dir: Path = None):
    """Build the command list for one prediction worker subprocess."""
    task_list = [tasks] if isinstance(tasks, str) else list(tasks)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "pred" / "call_api.py"),
        "--data_dir", str(data_dir),
        "--save_dir", str(pred_dir),
        "--benchmark", "synthetic",
        "--server_type", args.server_type,
        "--model_name_or_path", args.model_path,
        "--temperature", str(args.temperature),
        "--top_k", str(args.top_k),
        "--top_p", str(args.top_p),
        "--batch_size", str(args.batch_size),
        "--max_seq_length", str(max_seq_length),
    ]
    if len(task_list) == 1:
        cmd.extend(["--task", task_list[0]])
    else:
        cmd.append("--tasks")
        cmd.extend(task_list)
    if progress_file:
        cmd.extend(["--progress_file", str(progress_file)])
    if results_file:
        cmd.extend(["--results_file", str(results_file)])
    if log_dir:
        cmd.extend(["--log_dir", str(log_dir)])
    _is_nosa = "nosa" in args.model_path.lower()
    if not args.dense and not getattr(args, 'infinigen', False):
        if args.sparda:
            cmd.append("--sparda")
        elif not _is_nosa:
            cmd.append("--enable_sparse")
    if args.indexer_path:
        cmd.extend(["--indexer_path", args.indexer_path])
    if args.long_context:
        cmd.append("--long_context")
    if getattr(args, 'yarn', False):
        cmd.append("--yarn")
    if getattr(args, 'block_hit_rate', False):
        cmd.append("--collect_block_hit_rate")
    if getattr(args, 'infinigen', False):
        cmd.append("--enable_infinigen")
    if args.dense:
        cmd.append("--dense")
    return cmd


def run_predict(args, data_dir: Path, pred_dir: Path, task: str, max_seq_length: int):
    cmd = _build_predict_cmd(args, data_dir, pred_dir, task, max_seq_length)
    subprocess.run(cmd, check=True)


def _read_progress(progress_file: Path):
    """Read (task, current, total) from a progress file."""
    try:
        text = progress_file.read_text().strip()
        if text:
            if text.startswith("{"):
                payload = json.loads(text)
                return (
                    payload.get("task", ""),
                    int(payload.get("current", 0)),
                    int(payload.get("total", 0)),
                )
            parts = text.split()
            if len(parts) >= 3:
                return parts[0], int(parts[1]), int(parts[2])
            if len(parts) >= 2:
                return "", int(parts[0]), int(parts[1])
    except (OSError, ValueError, IndexError, json.JSONDecodeError):
        pass
    return "", 0, 0


def _tail_log(log_file: Path, max_lines: int):
    """Return (omitted_count, tail_lines) for a log file."""
    if max_lines <= 0:
        return 0, []
    total = 0
    tail = deque(maxlen=max_lines)
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                total += 1
                tail.append(line.rstrip("\n"))
    except OSError as e:
        return 0, [f"[RULER] Failed to read log '{log_file}': {e}"]
    omitted = max(0, total - len(tail))
    return omitted, list(tail)


def _seed_resume_chunks(final_output: Path, chunk_specs):
    if not final_output.exists():
        return
    if any(Path(spec["output_path"]).exists() for spec in chunk_specs):
        return

    lines_by_index = {}
    with final_output.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            lines_by_index[payload["index"]] = raw_line if raw_line.endswith("\n") else raw_line + "\n"

    if not lines_by_index:
        return

    print(f"[RULER] Resume: seeding {len(chunk_specs)} chunk file(s) from {final_output.name}")
    for spec in chunk_specs:
        chunk_lines = [
            lines_by_index[idx]
            for idx in spec["sample_indices"]
            if idx in lines_by_index
        ]
        if chunk_lines:
            _write_lines_atomic(Path(spec["output_path"]), chunk_lines)


def _prepare_prediction_chunks(args, data_dir: Path, pred_dir: Path,
                               tasks: list, num_gpus: int):
    task_configs = _load_task_configs("synthetic")
    log_dir = pred_dir / "logs"
    chunk_dir = pred_dir / "chunks"
    log_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    total_weight = sum(
        _task_weight(task_configs, task, args.num_samples)
        for task in tasks
    )
    target_chunks = max(len(tasks), num_gpus * 4)
    target_chunk_weight = max(1, (total_weight + target_chunks - 1) // target_chunks)

    work_items = []
    chunk_groups = {}
    skipped = []

    for task in tasks:
        task_file = _task_data_file(data_dir, task)
        samples = _read_jsonl(task_file)
        total_samples = len(samples)
        chunk_count = _task_chunk_count(
            task_configs, task, total_samples, target_chunk_weight, num_gpus
        )
        ranges = _split_ranges(total_samples, chunk_count)

        if len(ranges) <= 1:
            spec = {
                "task": task,
                "label": task,
                "task_key": task,
                "chunk_idx": 0,
                "chunk_amount": 1,
                "sample_indices": [sample["index"] for sample in samples],
                "expected_lines": total_samples,
                "expected_weight": _task_weight(task_configs, task, total_samples),
                "save_dir": str(pred_dir),
                "log_dir": str(log_dir),
                "output_path": str(pred_dir / f"{task}.jsonl"),
                "block_stats_path": str(pred_dir / f"{task}_block_stats.json"),
                "progress_file": str(log_dir / f"{task}.progress"),
                "log_file": str(log_dir / f"{task}.log"),
                "skipped": False,
            }
            if args.resume and _count_nonempty_lines(Path(spec["output_path"])) >= spec["expected_lines"]:
                spec["skipped"] = True
                skipped.append(spec["label"])
            else:
                work_items.append(spec)
            continue

        final_output = pred_dir / f"{task}.jsonl"
        final_block_stats = pred_dir / f"{task}_block_stats.json"
        chunk_specs = []
        for chunk_idx, (start_idx, end_idx) in enumerate(ranges):
            chunk_key = f"{task}.chunk{chunk_idx + 1}of{len(ranges)}"
            chunk_samples = samples[start_idx:end_idx]
            chunk_specs.append({
                "task": task,
                "label": f"{task} [{chunk_idx + 1}/{len(ranges)}]",
                "task_key": chunk_key,
                "chunk_idx": chunk_idx,
                "chunk_amount": len(ranges),
                "sample_indices": [sample["index"] for sample in chunk_samples],
                "expected_lines": len(chunk_samples),
                "expected_weight": _task_weight(task_configs, task, len(chunk_samples)),
                "save_dir": str(chunk_dir),
                "log_dir": str(log_dir),
                "output_path": str(chunk_dir / f"{chunk_key}.jsonl"),
                "block_stats_path": str(chunk_dir / f"{chunk_key}_block_stats.json"),
                "progress_file": str(log_dir / f"{chunk_key}.progress"),
                "log_file": str(log_dir / f"{chunk_key}.log"),
                "skipped": False,
            })

        if args.resume:
            _seed_resume_chunks(final_output, chunk_specs)

        chunk_groups[task] = {
            "final_output": str(final_output),
            "final_block_stats": str(final_block_stats),
            "chunks": chunk_specs,
        }

        for spec in chunk_specs:
            if args.resume and _count_nonempty_lines(Path(spec["output_path"])) >= spec["expected_lines"]:
                spec["skipped"] = True
                skipped.append(spec["label"])
                continue
            work_items.append(spec)

    work_items.sort(key=lambda item: item["expected_weight"], reverse=True)
    return work_items, chunk_groups, skipped


def _merge_chunked_outputs(chunk_groups, collect_block_hit_rate):
    if not chunk_groups:
        return

    merge_block_stats = None
    if collect_block_hit_rate:
        from block_hit_rate import merge_block_stats as _merge_block_stats
        merge_block_stats = _merge_block_stats

    for task_name, group in chunk_groups.items():
        final_output = Path(group["final_output"])
        final_block_stats = Path(group["final_block_stats"])
        chunk_specs = group["chunks"]
        merged_lines = []
        stats_list = []
        missing_stats = []
        incomplete = False

        for spec in sorted(chunk_specs, key=lambda item: item["chunk_idx"]):
            chunk_path = Path(spec["output_path"])
            if _count_nonempty_lines(chunk_path) < spec["expected_lines"]:
                incomplete = True
                break

            with chunk_path.open("r", encoding="utf-8") as fh:
                merged_lines.extend(
                    line if line.endswith("\n") else line + "\n"
                    for line in fh if line.strip()
                )

            if collect_block_hit_rate and merge_block_stats is not None:
                stats_path = Path(spec["block_stats_path"])
                if stats_path.exists():
                    try:
                        stats_list.append(json.loads(stats_path.read_text()))
                    except json.JSONDecodeError:
                        missing_stats.append(spec)
                else:
                    missing_stats.append(spec)

        if incomplete:
            if final_output.exists():
                try:
                    final_output.unlink()
                except OSError as exc:
                    print(
                        f"[RULER] WARNING: incomplete chunk merge for {task_name}; "
                        f"failed to remove stale {final_output.name}: {exc}"
                    )
                else:
                    print(
                        f"[RULER] WARNING: incomplete chunk merge for {task_name}; "
                        f"removed stale {final_output.name}."
                    )
            else:
                print(f"[RULER] WARNING: incomplete chunk merge for {task_name}; final JSONL not updated.")

            if final_block_stats.exists():
                try:
                    final_block_stats.unlink()
                except OSError as exc:
                    print(
                        f"[RULER] WARNING: incomplete chunk merge for {task_name}; "
                        f"failed to remove stale {final_block_stats.name}: {exc}"
                    )
            continue

        _write_lines_atomic(final_output, merged_lines)

        if collect_block_hit_rate and merge_block_stats is not None:
            if len(stats_list) == len(chunk_specs):
                merged_stats = merge_block_stats(stats_list)
                if merged_stats and merged_stats.get("total_blocks", 0) > 0:
                    _write_json_atomic(final_block_stats, merged_stats)
                    _print_block_hit_rate_summary("RULER", task_name, merged_stats)
                else:
                    try:
                        final_block_stats.unlink()
                    except FileNotFoundError:
                        pass
            elif final_block_stats.exists():
                if not stats_list and all(spec.get("skipped", False) for spec in missing_stats):
                    print(f"[RULER] Resume: preserving existing block stats for {task_name}.")
                    try:
                        existing_stats = json.loads(final_block_stats.read_text())
                    except json.JSONDecodeError:
                        existing_stats = None
                    _print_block_hit_rate_summary("RULER", task_name, existing_stats)
                else:
                    print(
                        f"[RULER] WARNING: incomplete chunk block stats for {task_name}; "
                        f"preserving existing {final_block_stats.name}."
                    )
            elif missing_stats:
                print(
                    f"[RULER] WARNING: incomplete chunk block stats for {task_name}; "
                    "final block stats not written."
                )


def _ruler_worker(gpu_id, base_args, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    pred_dir = SCRIPT_DIR / "pred"
    pred_dir_str = str(pred_dir)
    if pred_dir_str not in sys.path:
        sys.path.insert(0, pred_dir_str)

    llm = None
    collector = None
    call_api_mod = None
    tasks_base = None
    tasks_customized = None

    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            if call_api_mod is None:
                import call_api as call_api_mod  # pylint: disable=import-outside-toplevel
                tasks_base, tasks_customized = call_api_mod._load_task_catalog("synthetic")

            worker_args = argparse.Namespace(
                data_dir=Path(base_args["data_dir"]),
                save_dir=Path(task["save_dir"]),
                benchmark="synthetic",
                task=task["task"],
                tasks=[task["task"]],
                subset="validation",
                chunk_idx=task["chunk_idx"],
                chunk_amount=task["chunk_amount"],
                task_key=task["task_key"],
                server_type=base_args["server_type"],
                model_name_or_path=base_args["model_name_or_path"],
                temperature=base_args["temperature"],
                top_k=base_args["top_k"],
                top_p=base_args["top_p"],
                random_seed=base_args["random_seed"],
                stop_words=list(base_args["stop_words"]),
                threads=1,
                batch_size=base_args["batch_size"],
                max_seq_length=base_args["max_seq_length"],
                enable_sparse=base_args["enable_sparse"],
                enable_sparda=base_args["enable_sparda"],
                dense=base_args["dense"],
                indexer_path=base_args["indexer_path"],
                long_context=base_args["long_context"],
                yarn=base_args["yarn"],
                enable_infinigen=base_args["enable_infinigen"],
                progress_file=task["progress_file"],
                results_file=None,
                log_dir=Path(task["log_dir"]),
                collect_block_hit_rate=base_args["collect_block_hit_rate"],
            )
            task_config = call_api_mod._build_task_config(task["task"], tasks_base, tasks_customized)

            if llm is None:
                llm = call_api_mod.get_llm(worker_args, task_config["tokens_to_generate"])
                collector = call_api_mod._create_collector(worker_args, llm)

            result_queue.put({
                "event": "start",
                "gpu": gpu_id,
                "label": task["label"],
                "progress_file": task["progress_file"],
            })

            task_result = call_api_mod._run_single_task(
                worker_args, llm, collector, task["task"], task_config
            )
            result_queue.put({
                "event": "done",
                "gpu": gpu_id,
                "label": task["label"],
                "elapsed": float(task_result["elapsed"]),
                "returncode": int(task_result["returncode"]),
                "log_file": task_result["log_file"] or task["log_file"],
            })
        except Exception:
            try:
                with open(task["log_file"], "a", encoding="utf-8") as fh:
                    fh.write("\n[RULER worker] Unhandled exception:\n")
                    traceback.print_exc(file=fh)
            except OSError:
                pass
            result_queue.put({
                "event": "done",
                "gpu": gpu_id,
                "label": task["label"],
                "elapsed": 0.0,
                "returncode": 1,
                "log_file": task["log_file"],
            })

    if collector is not None:
        collector.uninstall()


def _run_predictions_parallel(args, data_dir: Path, pred_dir: Path,
                              tasks: list, seq_length: int, gpus: list):
    """Run chunked RULER tasks via a dynamic per-GPU worker queue."""
    work_items, chunk_groups, skipped = _prepare_prediction_chunks(
        args, data_dir, pred_dir, tasks, len(gpus)
    )
    unchunked_tasks = [task for task in tasks if task not in chunk_groups]
    if skipped:
        print(f"[RULER] Resume: skipping {len(skipped)} completed chunk(s): {skipped}")

    log_dir = pred_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RULER] Parallel prediction: {len(work_items)} chunk(s) on "
          f"{len(gpus)} GPUs {list(gpus)}")
    if chunk_groups:
        print("[RULER] Chunked tasks: " + ", ".join(
            f"{task} x{len(group['chunks'])}"
            for task, group in chunk_groups.items()
        ))

    if not work_items:
        _merge_chunked_outputs(chunk_groups, getattr(args, "block_hit_rate", False))
        if getattr(args, "block_hit_rate", False) and unchunked_tasks:
            _print_saved_block_hit_rate_summaries("RULER", pred_dir, unchunked_tasks)
        return {}

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task in work_items:
        task_queue.put(task)
    for _ in gpus:
        task_queue.put(None)

    base_args = {
        "data_dir": str(data_dir),
        "server_type": args.server_type,
        "model_name_or_path": args.model_path,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "random_seed": 0,
        "stop_words": [],
        "batch_size": args.batch_size,
        "max_seq_length": seq_length,
        "enable_sparse": (not args.dense and not getattr(args, "infinigen", False)
                           and not args.sparda and "nosa" not in args.model_path.lower()),
        "enable_sparda": args.sparda,
        "dense": args.dense,
        "indexer_path": args.indexer_path,
        "long_context": args.long_context,
        "yarn": getattr(args, "yarn", False),
        "enable_infinigen": getattr(args, "infinigen", False),
        "collect_block_hit_rate": getattr(args, "block_hit_rate", False),
    }

    workers = []
    for gpu in gpus:
        proc = ctx.Process(
            target=_ruler_worker,
            args=(gpu, base_args, task_queue, result_queue),
        )
        proc.start()
        workers.append(proc)

    is_tty = sys.stdout.isatty()
    show_event_messages = not is_tty
    progress_lines = 0
    rendered_progress = []
    gpu_info = {g: {"status": "idle"} for g in gpus}
    results = {}
    failed = []

    def _clear_progress():
        nonlocal progress_lines, rendered_progress
        if is_tty and progress_lines > 0:
            sys.stdout.write(f"\033[{progress_lines}A\033[J")
            sys.stdout.flush()
            progress_lines = 0
            rendered_progress = []

    def _show_progress():
        nonlocal progress_lines, rendered_progress
        if not is_tty:
            return
        lines = []
        for gpu in gpus:
            info = gpu_info[gpu]
            st = info["status"]
            if st == "idle":
                lines.append(f"  GPU {gpu:>2}: {'idle':22s}")
                continue

            task_label = info.get("task", "loading")
            progress_file = info.get("progress_file")
            _, cur, tot = _read_progress(Path(progress_file)) if progress_file else ("", 0, 0)
            if tot > 0:
                pct = cur / tot
                bar_w = 40
                filled = int(bar_w * pct)
                bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
                lines.append(f"  GPU {gpu:>2}: {task_label:22s} {bar} {cur:>4d}/{tot}")
            else:
                lines.append(f"  GPU {gpu:>2}: {task_label:22s} loading model...")
        if lines == rendered_progress:
            return
        if progress_lines > 0:
            sys.stdout.write(f"\033[{progress_lines}A\033[J")
        for line in lines:
            sys.stdout.write(f"\033[K{line}\n")
        sys.stdout.flush()
        progress_lines = len(lines)
        rendered_progress = list(lines)

    completed = 0
    while completed < len(work_items):
        messages = []
        pending = []
        try:
            pending.append(result_queue.get(timeout=0.5))
            while True:
                pending.append(result_queue.get_nowait())
        except queue.Empty:
            pass

        for event in pending:
            if event["event"] == "start":
                gpu_info[event["gpu"]] = {
                    "status": "running",
                    "task": event["label"],
                    "progress_file": event["progress_file"],
                }
                if show_event_messages:
                    messages.append(f"    [{event['label']}] started on GPU {event['gpu']}")
            elif event["event"] == "done":
                completed += 1
                rc = int(event["returncode"])
                results[event["label"]] = (float(event["elapsed"]), rc)
                gpu_info[event["gpu"]] = {"status": "idle"}
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                if show_event_messages:
                    messages.append(
                        f"    [{event['label']}] done on GPU {event['gpu']}: "
                        f"{float(event['elapsed']):.1f}s  [{status}]"
                    )
                if rc != 0:
                    failed.append((event["label"], rc, Path(event["log_file"])))
                    if show_event_messages:
                        messages.append(f"        see log: {event['log_file']}")

        if messages:
            _clear_progress()
            for msg in messages:
                print(msg)
        if completed < len(work_items):
            _show_progress()
        if completed < len(work_items) and not any(proc.is_alive() for proc in workers):
            raise RuntimeError(
                f"All RULER workers exited with {len(work_items) - completed} chunk(s) remaining."
            )

    _clear_progress()
    for proc in workers:
        proc.join(timeout=5)

    _merge_chunked_outputs(chunk_groups, getattr(args, "block_hit_rate", False))
    if getattr(args, "block_hit_rate", False) and unchunked_tasks:
        _print_saved_block_hit_rate_summaries("RULER", pred_dir, unchunked_tasks)

    if failed:
        tail_lines = int(getattr(args, "failed_log_tail_lines", 0) or 0)
        if tail_lines > 0:
            print(f"\n[RULER] Showing last {tail_lines} line(s) of failed task logs:")
            for task, ret, log_file in failed:
                omitted, lines = _tail_log(log_file, tail_lines)
                print(f"\n--- {task} (rc={ret}) :: {log_file.name} ---")
                if omitted > 0:
                    print(f"... ({omitted} earlier line(s) omitted) ...")
                if lines:
                    for line in lines:
                        print(line)
                else:
                    print("[RULER] Log file is empty.")
                print(f"--- end {task} ---")

        fail_msg = ", ".join(
            f"{task}(rc={ret}, log={log_file.name})"
            for task, ret, log_file in failed
        )
        raise RuntimeError(
            f"{len(failed)} prediction task(s) failed: {fail_msg}. "
            f"Check logs in {log_dir}/"
        )

    return results


def run_evaluate(pred_dir: Path):
    cmd = [
        sys.executable, str(SCRIPT_DIR / "eval" / "evaluate.py"),
        "--data_dir", str(pred_dir),
        "--benchmark", "synthetic",
    ]
    subprocess.run(cmd, check=True)


def run_analyze(pred_dir: Path):
    """Print per-category averages and overall score from summary.csv."""
    summary = pred_dir / "summary.csv"
    if not summary.exists():
        return
    cmd = [
        sys.executable, str(SCRIPT_DIR / "eval" / "analyze_results.py"),
        str(summary),
    ]
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(
        description="RULER benchmark runner for NOSA / MiniCPM / dense baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-path", required=True,
        help="HuggingFace model path or local directory.",
    )
    parser.add_argument(
        "--server-type", default="hf", choices=["hf", "shadowkv"],
        help="Inference backend (default: hf).",
    )
    parser.add_argument(
        "--template-type", default="minicpm4",
        help="Model template type for data preparation (default: minicpm4).",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Force dense (full) attention. By default MiniCPM models use sparse attention.",
    )
    parser.add_argument(
        "--sparda", action="store_true",
        help="Enable SparDA sparse attention with trained q_future indexer. "
             "Requires --indexer-path.",
    )
    parser.add_argument(
        "--indexer-path", default=None,
        help="Path to .pt checkpoint with q_future_proj/q_curr_proj indexer weights. "
             "Required when --sparda is set.",
    )
    parser.add_argument(
        "--long-context", action="store_true",
        help="Enable 128K long-context extension. "
             "For NOSA: increases rope_theta to 40000 (default). "
             "For MiniCPM: applies LongRoPE.",
    )
    parser.add_argument(
        "--yarn", action="store_true",
        help="Use YaRN (factor=4.0) instead of rope_theta increase for NOSA long-context. "
             "Only effective with --long-context on NOSA models.",
    )
    parser.add_argument(
        "--infinigen", action="store_true",
        help="Enable InfiniGen decode (InfLLMv2 sparse prefill + InfiniGen masked decode).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Root output directory for predictions/scores. "
             "Defaults to RULER/scripts/results/<model-basename>/.",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Shared directory for generated synthetic datasets. "
             "Defaults to <results_root>/data/<seq_len>_n<num_samples>/ "
             "(same root as --output-dir). Data is tokenizer-dependent but "
             "model/mode-independent, so it is reused across runs automatically.",
    )
    parser.add_argument(
        "--seq-len", type=int, nargs="+", default=None,
        help="Sequence lengths to evaluate. Defaults to the model's native max "
             "(65536 for MiniCPM, 32768 for NOSA). "
             "Accepts multiple values, e.g. --seq-len 16384 32768 65536.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Tasks to run. Defaults to all 13 synthetic RULER tasks.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for prediction (default: 1).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of samples per task (default: 50).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 = greedy).",
    )
    parser.add_argument(
        "--top-k", type=int, default=32,
        help="Top-k sampling (default: 32).",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Top-p / nucleus sampling (default: 1.0).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regeneration of synthetic data (ignore cached data).",
    )
    parser.add_argument(
        "--gpus", default="auto",
        help="Comma-separated GPU IDs to use (default: auto). "
             "'auto' detects all available GPUs via CUDA_VISIBLE_DEVICES or "
             "nvidia-smi. When multiple GPUs are available, tasks are "
             "balanced across persistent per-GPU workers automatically.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip data preparation and prediction; evaluate existing outputs only.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run: keep existing predictions and only "
             "run missing tasks. Without this flag, old predictions, logs, "
             "and summaries are cleared while reusable synthetic data is kept.",
    )
    parser.add_argument(
        "--block-hit-rate", action="store_true",
        help="Collect and print block hit-rate stats per task "
             "(NOSA/MiniCPM sparse models only).",
    )
    parser.add_argument(
        "--failed-log-tail-lines", type=int, default=40,
        help="On parallel prediction failure, print the last N log lines per failed task "
             "(default: 40, set 0 to disable).",
    )

    args = parser.parse_args()

    # Resolve default --seq-len from model's native max context length
    is_nosa = "nosa" in args.model_path.lower()
    native_max = 32768 if is_nosa else 65536
    if args.seq_len is None:
        args.seq_len = [native_max]
    max_requested = max(args.seq_len)
    if not args.long_context and max_requested > native_max:
        args.long_context = True
        print(f"[Auto] --long-context enabled: seq_len {max_requested} > "
              f"native max {native_max}")

    # Validate flag combinations
    if args.dense and args.sparda:
        parser.error("--dense and --sparda are mutually exclusive.")
    if getattr(args, 'infinigen', False) and args.dense:
        parser.error("--dense and --infinigen are mutually exclusive.")
    if getattr(args, 'infinigen', False) and args.sparda:
        print("WARNING: --infinigen overrides --sparda. Ignoring --sparda.")
        args.sparda = False
    # InfiniGen now supports both MiniCPM and NOSA models
    if args.sparda and args.indexer_path is None:
        parser.error("--indexer-path is required when --sparda is set. "
                     "Provide a .pt checkpoint with q_future_proj/q_curr_proj weights.")
    if args.indexer_path and not os.path.exists(args.indexer_path):
        parser.error(f"--indexer-path does not exist: {args.indexer_path}")
    if args.failed_log_tail_lines < 0:
        parser.error("--failed-log-tail-lines must be >= 0.")

    tasks = args.tasks if args.tasks else ALL_SYNTHETIC_TASKS

    duplicate_tasks = [t for t, c in Counter(tasks).items() if c > 1]
    if duplicate_tasks:
        parser.error(
            "--tasks contains duplicates, which can cause parallel workers "
            f"to write to the same output file: {duplicate_tasks}"
        )

    if not args.eval_only:
        ensure_data_files(tasks)

    # --- GPU detection ---
    gpus = _detect_gpus(args.gpus)
    num_gpus = len(gpus)
    parallel = num_gpus > 1

    model_basename = Path(args.model_path).name
    if args.output_dir:
        results_root = Path(args.output_dir)
        root_dir = results_root
    else:
        results_root = SCRIPT_DIR / "results"
        root_dir = results_root / model_basename

    if is_nosa:
        if args.dense:
            mode = "nosa_dense"
        elif getattr(args, 'infinigen', False):
            mode = "nosa_infinigen"
        elif args.sparda:
            mode = "nosa_sparda"
        else:
            mode = "nosa"
    elif args.dense:
        mode = "dense"
    elif getattr(args, 'infinigen', False):
        mode = "infinigen"
    elif args.sparda:
        mode = "minicpm_sparda"
    else:
        mode = "minicpm"

    if is_nosa:
        if args.long_context and args.yarn:
            rope_mode = "yarn"
        elif args.long_context:
            rope_mode = "rope_theta=40000"
        else:
            rope_mode = "native"
    else:
        rope_mode = "longrope" if args.long_context else "native"

    print(f"[RULER] model={args.model_path}")
    print(f"[RULER] mode={mode}, server={args.server_type}, "
          f"long_context={args.long_context}, rope_mode={rope_mode}")
    print(f"[RULER] seq_lengths={args.seq_len}, tasks={len(tasks)}, "
          f"samples={args.num_samples}, batch={args.batch_size}")
    print(f"[RULER] GPUs={gpus} "
          f"({'parallel, ' + str(num_gpus) + ' workers' if parallel else 'sequential'})")
    print(f"[RULER] output_dir={root_dir}")
    print(f"[RULER] data_dir={args.data_dir or results_root / 'data' / '<seq_len>_n<samples>'}")
    print()

    total_time = 0

    for seq_length in args.seq_len:
        # Data is shared across all models/modes (only depends on
        # tokenizer, seq_length, and num_samples).
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            data_dir = results_root / "data" / f"{seq_length}_n{args.num_samples}"

        # Predictions are model- and mode-specific.
        pred_dir = root_dir / mode / "synthetic" / str(seq_length) / "pred"
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        if not args.eval_only:
            # Clear old predictions unless resuming. Shared synthetic data lives
            # under data_dir and is intentionally preserved.
            if not args.resume:
                old_preds = list(pred_dir.glob("*.jsonl"))
                old_preds += list(pred_dir.glob("*_block_stats.json"))
                old_preds += list(pred_dir.glob("summary.csv"))
                old_preds += list((pred_dir / "chunks").glob("*.jsonl"))
                old_preds += list((pred_dir / "chunks").glob("*_block_stats.json"))
                old_preds += list((pred_dir / "logs").glob("*.log"))
                old_preds += list((pred_dir / "logs").glob("*.progress"))
                if old_preds:
                    for f in old_preds:
                        f.unlink()
                    print(f"[RULER] Cleared {len(old_preds)} old file(s) in {pred_dir}")

            # ---- Phase 1: data preparation (sequential, CPU-only) ----
            for task in tasks:
                ensure_task_data(args, data_dir, task, seq_length)

            # ---- Phase 2: predictions ----
            if parallel:
                t0 = time.time()
                results = _run_predictions_parallel(
                    args, data_dir, pred_dir, tasks, seq_length, gpus)
                wall_time = time.time() - t0
                sum_time = sum(e for e, _ in results.values())
                total_time += wall_time
                print(f"    parallel wall-time: {wall_time:.1f}s  "
                      f"(sum across GPUs: {sum_time:.1f}s)")
            else:
                for task in tasks:
                    t0 = time.time()
                    run_predict(args, data_dir, pred_dir, task, seq_length)
                    elapsed = time.time() - t0
                    total_time += elapsed
                    print(f"    [{task}] predict: {elapsed:.1f}s")
        else:
            if not pred_dir.exists():
                raise FileNotFoundError(
                    f"[RULER] --eval-only requires existing predictions at {pred_dir}"
                )
            print(f"[RULER] Eval-only: reusing predictions in {pred_dir}")

        # ---- Phase 3: evaluation (after all tasks complete) ----
        print(f"\n=== Evaluating seq_length={seq_length} ===")
        run_evaluate(pred_dir)
        run_analyze(pred_dir)

    print(f"\nTotal prediction time: {total_time:.1f}s "
          f"({total_time / 60:.1f} min)")
    if parallel:
        print(f"  ({num_gpus} GPUs, chunk-queue parallel execution)")


if __name__ == "__main__":
    main()
