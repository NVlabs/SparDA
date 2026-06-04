#!/usr/bin/env python3
# This file is copied/adapted from LongBench (https://github.com/THUDM/LongBench).
# Copyright (c) 2023 THU-KEG & Zhipu AI.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
LongBench benchmark runner: predict and evaluate.

Thinking mode is OFF: greedy decoding (do_sample=False, temperature=0) with
no <think> prompting.

Data is auto-downloaded on first run if not present.

Usage:
    # NOSA 8B (model key)
    python run.py --model 8b_nosa_sft

    # MiniCPM 8B
    python run.py --model 8b_minicpm_sft

    # Direct model path (auto-registers as model key)
    python run.py --model-path openbmb/NOSA-8B
    python run.py --model-path openbmb/MiniCPM4.1-8B --sparda --indexer-path /path/to/indexer_weights.pt

    # Long-context (auto-enabled when max_length > native max)
    python run.py --model 8b_nosa_sft --long-context
    python run.py --model-path openbmb/NOSA-8B --max-length 131072

    # Multi-GPU (auto-detected -- runs all LongBench datasets across visible GPUs)
    python run.py --model-path openbmb/NOSA-8B

    # Explicit GPU selection
    python run.py --model-path openbmb/NOSA-8B --gpus 0,1,2,3

    # Evaluate only
    python run.py --model 8b_nosa_sft --eval-only

    # LongBench-E (length-stratified)
    python run.py --model 8b_nosa_sft --longbench-e

Multi-GPU parallelization:
    When multiple GPUs are available (auto-detected via CUDA_VISIBLE_DEVICES
    or nvidia-smi), persistent GPU workers load the model once and pull queued
    prediction chunks. Datasets are split into multiple sample ranges when
    needed so late-stage work stays balanced on 4-8 GPUs. Per-task logs are
    saved under the configured output root, e.g.
    <output-dir>/pred/<model>/logs/<task>.log. Use --gpus to override GPU
    selection.
"""

import argparse
import contextlib
import fcntl
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_{os.getpid()}")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_URL = "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"

benchmarks_dir_str = str(BENCHMARKS_DIR)
if benchmarks_dir_str not in sys.path:
    sys.path.insert(0, benchmarks_dir_str)

NATIVE_MAX = {"nosa": 32768, "minicpm": 65536}

DATASETS = [
    "gov_report", "triviaqa", "narrativeqa", "qasper", "qmsum", "musique",
    "2wikimqa", "multifieldqa_en", "repobench-p", "hotpotqa",
    "trec", "passage_retrieval_en", "passage_count", "samsum",
]

_DATASET_SIZE_CACHE = {}
_DATASET_MAX_GEN_CACHE = None


def _print_block_hit_rate_summary(prefix: str, label: str, stats: dict):
    if not stats or stats.get("total_blocks", 0) <= 0:
        return
    hit_rate = float(stats.get("hit_rate", 0.0)) * 100.0
    miss_blocks = int(stats.get("miss_blocks", 0))
    total_blocks = int(stats.get("total_blocks", 0))
    print(f"[{prefix}] {label}: block hit-rate {hit_rate:.1f}% ({total_blocks - miss_blocks}/{total_blocks})")


def _print_saved_block_hit_rate_summaries(prefix: str, pred_dir: Path, dataset_names: list):
    missing = []
    for dataset in dataset_names:
        stats_path = pred_dir / f"{dataset}_block_stats.json"
        if not stats_path.exists():
            missing.append(dataset)
            continue
        try:
            stats = json.loads(stats_path.read_text())
        except json.JSONDecodeError:
            missing.append(dataset)
            continue
        if stats and stats.get("total_blocks", 0) > 0:
            _print_block_hit_rate_summary(prefix, dataset, stats)
        else:
            missing.append(dataset)
    if missing:
        print(f"[{prefix}] WARNING: missing block hit-rate stats for {missing}")


def _missing_data_files(longbench_e: bool):
    suffix = "_e" if longbench_e else ""
    return [
        f"{dataset}{suffix}.jsonl"
        for dataset in DATASETS
        if not (DATA_DIR / f"{dataset}{suffix}.jsonl").is_file()
    ]


def _download_data_archive(zip_path: Path):
    temp_zip_path = zip_path.with_suffix(zip_path.suffix + f".{os.getpid()}.part")
    try:
        subprocess.run(
            ["curl", "-L", DATA_URL, "-o", str(temp_zip_path)],
            check=True,
        )
        temp_zip_path.replace(zip_path)
    finally:
        try:
            temp_zip_path.unlink()
        except FileNotFoundError:
            pass


def ensure_data(longbench_e: bool = False):
    missing = _missing_data_files(longbench_e)
    if not missing:
        return

    print(f"[LongBench] Data incomplete, missing: {missing}")
    zip_path = SCRIPT_DIR / "data.zip"
    lock_path = SCRIPT_DIR / ".data.lock"

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        missing = _missing_data_files(longbench_e)
        if not missing:
            print("[LongBench] Data ready.")
            return

        for archive_attempt in range(2):
            if not zip_path.exists() or zip_path.stat().st_size <= 1_000_000:
                print("[LongBench] Downloading data archive...")
                _download_data_archive(zip_path)
            else:
                print(f"[LongBench] Reusing existing {zip_path.name} for extraction...")

            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(SCRIPT_DIR)
            except (zipfile.BadZipFile, OSError, EOFError) as exc:
                try:
                    zip_path.unlink()
                except FileNotFoundError:
                    pass
                if archive_attempt == 0:
                    print(f"[LongBench] Archive extraction failed ({exc}); re-downloading once ...")
                    continue
                raise RuntimeError(f"Failed to extract LongBench archive: {zip_path}") from exc
            break

        try:
            zip_path.unlink()
        except FileNotFoundError:
            pass

    missing = _missing_data_files(longbench_e)
    if missing:
        raise RuntimeError(f"[LongBench] Missing data files after extraction: {missing}")
    print("[LongBench] Data ready.")


def detect_method(model_key: str, model_path: str) -> str:
    """Detect method (nosa / minicpm / other) for auto long-context."""
    combined = (model_key + model_path).lower()
    if "nosa" in combined:
        return "nosa"
    if "minicpm" in combined or "infllmv2" in combined:
        return "minicpm"
    return "other"


def normalize_local_path(path_str):
    """Resolve local filesystem paths while leaving HF repo IDs untouched."""
    if not path_str:
        return path_str
    path = Path(path_str).expanduser()
    if path.exists() or path_str.startswith(("/", "./", "../", "~")):
        return str(path.resolve())
    return path_str


def _write_json_atomic(cfg_path: Path, payload: dict):
    temp_path = cfg_path.with_name(f"{cfg_path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=4) + "\n")
    temp_path.replace(cfg_path)


def register_model(model_key: str, model_path: str, max_length: int):
    """Ensure model_key exists in model2path.json and model2maxlen.json."""
    config_dir = SCRIPT_DIR / "config"
    lock_path = config_dir / ".model_registry.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        for cfg_name, value in [("model2path.json", model_path),
                                ("model2maxlen.json", max_length)]:
            cfg_path = config_dir / cfg_name
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            if model_key not in cfg or cfg[model_key] != value:
                cfg[model_key] = value
                _write_json_atomic(cfg_path, cfg)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpus(gpu_arg: str) -> list:
    """Return a list of GPU ID strings.

    - ``"auto"`` (default): respects CUDA_VISIBLE_DEVICES if already set,
      otherwise queries ``nvidia-smi``, falling back to ``["0"]``.
    - Comma-separated IDs (e.g. ``"0,2,5"``): uses those specific GPUs.
    """
    if gpu_arg != "auto":
        return [g.strip() for g in gpu_arg.split(",") if g.strip()]

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
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


# ---------------------------------------------------------------------------
# Task-parallel prediction
# ---------------------------------------------------------------------------

def _build_pred_cmd(model_key, dataset, max_length, output_dir,
                    sparda, infinigen, indexer_path, longbench_e,
                    progress_file=None, resume=False,
                    collect_block_hit_rate=False):
    """Build the command list for a single prediction subprocess."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "pred.py"),
        "--model", model_key,
        "--dataset", dataset,
        "--max-length", str(max_length),
        "--output-dir", str(output_dir),
    ]
    if progress_file:
        cmd.extend(["--progress-file", str(progress_file)])
    if sparda:
        cmd.append("--sparda")
    if infinigen:
        cmd.append("--infinigen")
    if indexer_path:
        cmd.extend(["--indexer_path", indexer_path])
    if longbench_e:
        cmd.append("--e")
    if resume:
        cmd.append("--resume")
    if collect_block_hit_rate:
        cmd.append("--collect_block_hit_rate")
    return cmd


def _read_progress(progress_file: Path):
    """Read (current, total) from a progress file. Returns (0, 0) on failure."""
    try:
        text = progress_file.read_text().strip()
        if text:
            parts = text.split()
            return int(parts[0]), int(parts[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0, 0


def _dataset_data_file(dataset: str, longbench_e: bool) -> Path:
    suffix = "_e" if longbench_e else ""
    return DATA_DIR / f"{dataset}{suffix}.jsonl"


def _count_nonempty_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def _dataset_sample_count(dataset: str, longbench_e: bool) -> int:
    key = (dataset, longbench_e)
    if key not in _DATASET_SIZE_CACHE:
        _DATASET_SIZE_CACHE[key] = _count_nonempty_lines(
            _dataset_data_file(dataset, longbench_e)
        )
    return _DATASET_SIZE_CACHE[key]


def _dataset_max_gen(dataset: str) -> int:
    global _DATASET_MAX_GEN_CACHE
    if _DATASET_MAX_GEN_CACHE is None:
        _DATASET_MAX_GEN_CACHE = json.loads(
            (SCRIPT_DIR / "config" / "dataset2maxlen.json").read_text()
        )
    return int(_DATASET_MAX_GEN_CACHE.get(dataset, 64) or 64)


def _dataset_task_weight(dataset: str, longbench_e: bool) -> int:
    return max(1, _dataset_sample_count(dataset, longbench_e)) * max(1, _dataset_max_gen(dataset))


def _split_ranges(total: int, num_parts: int):
    if total <= 0 or num_parts <= 1:
        return [(0, total)]

    ranges = []
    base = total // num_parts
    extra = total % num_parts
    start = 0
    for idx in range(num_parts):
        size = base + (1 if idx < extra else 0)
        if size <= 0:
            continue
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges or [(0, total)]


def _task_stem(dataset: str, shard_index=None, num_shards: int = 1) -> str:
    if shard_index is None or num_shards <= 1:
        return dataset
    return f"{dataset}.chunk{shard_index + 1}of{num_shards}"


def _write_lines_atomic(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    temp_path.replace(path)


def _desired_shard_count(dataset: str, num_gpus: int, longbench_e: bool,
                         target_chunk_weight: int) -> int:
    if longbench_e or num_gpus < 2:
        return 1

    total_samples = _dataset_sample_count(dataset, longbench_e)
    if total_samples <= 1 or target_chunk_weight <= 0:
        return 1

    desired = (_dataset_task_weight(dataset, longbench_e) + target_chunk_weight - 1) // target_chunk_weight
    return max(1, min(total_samples, desired))


def _seed_resume_shards(final_output: Path, shard_specs):
    if not final_output.exists():
        return
    if any(Path(spec["output_path"]).exists() for spec in shard_specs):
        return

    with final_output.open("r", encoding="utf-8") as f:
        lines = [line if line.endswith("\n") else line + "\n" for line in f if line.strip()]
    if not lines:
        return

    print(f"[LongBench] Resume: seeding {len(shard_specs)} shard file(s) from {final_output.name}")
    for spec in shard_specs:
        chunk = lines[spec["start_index"]:spec["end_index"]]
        if chunk:
            _write_lines_atomic(Path(spec["output_path"]), chunk)


def _prepare_prediction_tasks(model_key, datasets, output_dir, num_gpus,
                              longbench_e, resume):
    pred_subdir = "pred_e" if longbench_e else "pred"
    pred_dir = Path(output_dir) / pred_subdir / model_key
    log_dir = pred_dir / "logs"
    shard_dir = pred_dir / "shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    shard_groups = {}
    skipped = []
    total_weight = sum(_dataset_task_weight(dataset, longbench_e) for dataset in datasets)
    target_chunks = max(len(datasets), num_gpus * 4)
    target_chunk_weight = max(1, (total_weight + target_chunks - 1) // target_chunks)

    for dataset in datasets:
        total_samples = _dataset_sample_count(dataset, longbench_e)
        shard_count = _desired_shard_count(dataset, num_gpus, longbench_e, target_chunk_weight)
        ranges = _split_ranges(total_samples, shard_count)

        if len(ranges) > 1:
            shard_specs = []
            final_output = pred_dir / f"{dataset}.jsonl"
            final_block_stats = pred_dir / f"{dataset}_block_stats.json"
            for shard_index, (start_idx, end_idx) in enumerate(ranges):
                stem = _task_stem(dataset, shard_index, len(ranges))
                shard_specs.append({
                    "dataset": dataset,
                    "label": f"{dataset} [{shard_index + 1}/{len(ranges)}]",
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "expected_lines": end_idx - start_idx,
                    "output_path": str(shard_dir / f"{stem}.jsonl"),
                    "block_stats_path": str(shard_dir / f"{stem}_block_stats.json"),
                    "progress_file": str(log_dir / f"{stem}.progress"),
                    "log_file": str(log_dir / f"{stem}.log"),
                    "shard_index": shard_index,
                    "num_shards": len(ranges),
                    "expected_weight": max(1, end_idx - start_idx) * max(1, _dataset_max_gen(dataset)),
                    "skipped": False,
                })

            if resume:
                _seed_resume_shards(final_output, shard_specs)

            shard_groups[dataset] = {
                "final_output": str(final_output),
                "final_block_stats": str(final_block_stats),
                "shards": shard_specs,
            }

            for spec in shard_specs:
                if resume and _count_nonempty_lines(Path(spec["output_path"])) >= spec["expected_lines"]:
                    spec["skipped"] = True
                    skipped.append(spec["label"])
                    continue
                tasks.append(spec)
            continue

        spec = {
            "dataset": dataset,
            "label": dataset,
            "start_index": 0,
            "end_index": total_samples,
            "expected_lines": total_samples,
            "output_path": str(pred_dir / f"{dataset}.jsonl"),
            "block_stats_path": str(pred_dir / f"{dataset}_block_stats.json"),
            "progress_file": str(log_dir / f"{dataset}.progress"),
            "log_file": str(log_dir / f"{dataset}.log"),
            "shard_index": None,
            "num_shards": 1,
            "expected_weight": _dataset_task_weight(dataset, longbench_e),
        }
        if resume and _count_nonempty_lines(Path(spec["output_path"])) >= total_samples:
            skipped.append(spec["label"])
            continue
        tasks.append(spec)

    tasks.sort(key=lambda item: item["expected_weight"], reverse=True)
    return tasks, shard_groups, skipped


def _merge_sharded_outputs(model_key, output_dir, longbench_e, shard_groups,
                           collect_block_hit_rate):
    if not shard_groups:
        return

    merge_block_stats = None
    if collect_block_hit_rate:
        from block_hit_rate import merge_block_stats as _merge_block_stats
        merge_block_stats = _merge_block_stats

    for dataset, group in shard_groups.items():
        final_output = Path(group["final_output"])
        final_block_stats = Path(group["final_block_stats"])
        shard_specs = group["shards"]
        merged_lines = []
        stats_list = []
        missing_stats = []
        incomplete = False

        for spec in sorted(shard_specs, key=lambda item: item["shard_index"]):
            shard_path = Path(spec["output_path"])
            if _count_nonempty_lines(shard_path) < spec["expected_lines"]:
                incomplete = True
                break
            with shard_path.open("r", encoding="utf-8") as f:
                merged_lines.extend(
                    line if line.endswith("\n") else line + "\n"
                    for line in f if line.strip()
                )
            if collect_block_hit_rate:
                stats_path = Path(spec["block_stats_path"])
                if stats_path.exists():
                    try:
                        stats_list.append(json.loads(stats_path.read_text()))
                    except json.JSONDecodeError:
                        missing_stats.append(spec)
                else:
                    missing_stats.append(spec)

        if incomplete:
            try:
                final_output.unlink()
            except FileNotFoundError:
                pass
            try:
                final_block_stats.unlink()
            except FileNotFoundError:
                pass
            print(f"[LongBench] WARNING: shard merge incomplete for {dataset}; final JSONL not updated.")
            continue

        _write_lines_atomic(final_output, merged_lines)

        if collect_block_hit_rate and merge_block_stats is not None:
            if len(stats_list) == len(shard_specs):
                merged_stats = merge_block_stats(stats_list)
                if merged_stats and merged_stats.get("total_blocks", 0) > 0:
                    _write_json_atomic(final_block_stats, merged_stats)
                    _print_block_hit_rate_summary("LongBench", dataset, merged_stats)
                else:
                    try:
                        final_block_stats.unlink()
                    except FileNotFoundError:
                        pass
            elif final_block_stats.exists():
                if not stats_list and all(spec.get("skipped", False) for spec in missing_stats):
                    print(f"[LongBench] Resume: preserving existing block stats for {dataset}.")
                    try:
                        existing_stats = json.loads(final_block_stats.read_text())
                    except json.JSONDecodeError:
                        existing_stats = None
                    _print_block_hit_rate_summary("LongBench", dataset, existing_stats)
                else:
                    print(
                        f"[LongBench] WARNING: incomplete shard block stats for {dataset}; "
                        f"preserving existing {final_block_stats.name}."
                    )
            elif missing_stats:
                print(
                    f"[LongBench] WARNING: incomplete shard block stats for {dataset}; "
                    "final block stats not written."
                )


def _longbench_worker(gpu_id, base_args, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    collector = None
    data_cache = {}

    try:
        import torch
        import pred as pred_mod

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, tokenizer = pred_mod.load_model_and_tokenizer(
            base_args["model_path"],
            base_args["model_key"],
            device,
            enable_sparda=base_args["sparda"],
            enable_infinigen=base_args["infinigen"],
            indexer_path=base_args["indexer_path"],
        )
        dataset2prompt = json.loads((SCRIPT_DIR / "config" / "dataset2prompt.json").read_text())
        dataset2maxlen = json.loads((SCRIPT_DIR / "config" / "dataset2maxlen.json").read_text())

        if base_args["collect_block_hit_rate"]:
            from block_hit_rate import BlockHitRateCollector
            collector = BlockHitRateCollector(model)
            if collector.available:
                collector.install()
            else:
                collector = None

    except Exception:
        result_queue.put({
            "event": "worker_init_failed",
            "gpu": gpu_id,
            "error": traceback.format_exc(),
        })
        return

    while True:
        task = task_queue.get()
        if task is None:
            break

        progress_file = Path(task["progress_file"])
        try:
            progress_file.unlink()
        except FileNotFoundError:
            pass

        result_queue.put({
            "event": "start",
            "gpu": gpu_id,
            "label": task["label"],
            "progress_file": task["progress_file"],
        })

        start_time = time.time()
        rc = 0
        log_file = Path(task["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with log_file.open("w", encoding="utf-8") as log_fh, \
                contextlib.redirect_stdout(log_fh), \
                contextlib.redirect_stderr(log_fh):
            try:
                print(f"[LongBench worker] gpu={gpu_id} task={task['label']}")
                dataset_key = (task["dataset"], base_args["longbench_e"])
                if dataset_key not in data_cache:
                    data_path = _dataset_data_file(task["dataset"], base_args["longbench_e"])
                    dataset = pred_mod.load_dataset("json", data_files=str(data_path))["train"]
                    data_cache[dataset_key] = list(dataset)
                data_all = data_cache[dataset_key][task["start_index"]:task["end_index"]]
                out_path = Path(task["output_path"])
                out_path.parent.mkdir(parents=True, exist_ok=True)

                pred_mod.predict_dataset(
                    model,
                    tokenizer,
                    base_args["model_key"],
                    task["dataset"],
                    data_all,
                    base_args["max_length"],
                    dataset2maxlen[task["dataset"]],
                    dataset2prompt[task["dataset"]],
                    str(out_path),
                    progress_file=task["progress_file"],
                    resume=base_args["resume"],
                    collector=collector,
                )

                if collector is not None:
                    stats = collector.get_stats()
                    if stats and stats.get("total_blocks", 0) > 0:
                        stats_path = Path(task["block_stats_path"])
                        stats_path.parent.mkdir(parents=True, exist_ok=True)
                        _write_json_atomic(stats_path, stats)
                        print(f"[LongBench worker] block hit-rate: {stats['hit_rate'] * 100:.1f}%")
                    collector._tracker.reset()

            except Exception:
                traceback.print_exc()
                rc = 1
                if collector is not None:
                    collector._tracker.reset()

        try:
            progress_file.unlink()
        except FileNotFoundError:
            pass

        result_queue.put({
            "event": "done",
            "gpu": gpu_id,
            "label": task["label"],
            "elapsed": time.time() - start_time,
            "returncode": rc,
            "log_file": task["log_file"],
        })

    if collector is not None:
        collector.uninstall()


def _run_predictions_parallel(model_key, datasets, max_length, output_dir, gpus,
                              sparda, infinigen, indexer_path, longbench_e,
                              resume=False, collect_block_hit_rate=False):
    """Run LongBench with persistent per-GPU workers and dynamic chunk queue."""
    tasks, shard_groups, skipped = _prepare_prediction_tasks(
        model_key, datasets, output_dir, len(gpus), longbench_e, resume
    )
    unsharded_datasets = [dataset for dataset in datasets if dataset not in shard_groups]
    if skipped:
        print(f"[LongBench] Resume: skipping {len(skipped)} completed task(s): {skipped}")

    print(f"[LongBench] Parallel prediction: {len(tasks)} task(s) on "
          f"{len(gpus)} GPUs {list(gpus)}")
    if shard_groups:
        print("[LongBench] Chunked datasets: " + ", ".join(
            f"{dataset} x{len(group['shards'])}"
            for dataset, group in shard_groups.items()
        ))

    if not tasks:
        _merge_sharded_outputs(
            model_key, output_dir, longbench_e, shard_groups, collect_block_hit_rate
        )
        if collect_block_hit_rate and unsharded_datasets:
            pred_dir = Path(output_dir) / ("pred_e" if longbench_e else "pred") / model_key
            _print_saved_block_hit_rate_summaries("LongBench", pred_dir, unsharded_datasets)
        return {}

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in gpus:
        task_queue.put(None)

    model_registry = json.loads((SCRIPT_DIR / "config" / "model2path.json").read_text())
    base_args = {
        "model_key": model_key,
        "model_path": model_registry[model_key],
        "max_length": max_length,
        "sparda": sparda,
        "infinigen": infinigen,
        "indexer_path": indexer_path,
        "longbench_e": longbench_e,
        "resume": resume,
        "collect_block_hit_rate": collect_block_hit_rate,
    }

    workers = []
    for gpu in gpus:
        proc = ctx.Process(
            target=_longbench_worker,
            args=(gpu, base_args, task_queue, result_queue),
        )
        proc.start()
        workers.append(proc)

    is_tty = sys.stdout.isatty()
    show_event_messages = not is_tty
    progress_lines = 0
    rendered_progress = []
    gpu_info = {gpu: {"status": "idle"} for gpu in gpus}
    results = {}

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
            if info["status"] == "idle":
                lines.append(f"  GPU {gpu:>2}: {'idle':22s}")
                continue

            progress_file = info.get("progress_file")
            cur, tot = _read_progress(Path(progress_file)) if progress_file else (0, 0)
            if tot > 0:
                pct = cur / tot
                bar_w = 40
                filled = int(bar_w * pct)
                bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
                lines.append(f"  GPU {gpu:>2}: {info['label']:22s} {bar} {cur:>4d}/{tot}")
            else:
                lines.append(f"  GPU {gpu:>2}: {info['label']:22s} loading model...")

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
    while completed < len(tasks):
        messages = []
        pending = []
        try:
            pending.append(result_queue.get(timeout=0.5))
            while True:
                pending.append(result_queue.get_nowait())
        except queue.Empty:
            pass

        for event in pending:
            if event["event"] == "worker_init_failed":
                raise RuntimeError(
                    f"LongBench worker failed to initialize on GPU {event['gpu']}:\n"
                    f"{event['error']}"
                )
            if event["event"] == "start":
                gpu_info[event["gpu"]] = {
                    "status": "running",
                    "label": event["label"],
                    "progress_file": event["progress_file"],
                }
                if show_event_messages:
                    messages.append(f"    [{event['label']}] started on GPU {event['gpu']}")
            elif event["event"] == "done":
                completed += 1
                rc = event["returncode"]
                results[event["label"]] = (event["elapsed"], rc)
                gpu_info[event["gpu"]] = {"status": "idle"}
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                if show_event_messages:
                    messages.append(
                        f"    [{event['label']}] done on GPU {event['gpu']}: "
                        f"{event['elapsed']:.1f}s  [{status}]"
                    )
                if rc != 0:
                    if show_event_messages:
                        messages.append(f"        see log: {event['log_file']}")

        if messages:
            _clear_progress()
            for msg in messages:
                print(msg)
        if completed < len(tasks):
            _show_progress()

        if completed < len(tasks) and not any(proc.is_alive() for proc in workers):
            raise RuntimeError(
                f"All LongBench workers exited with {len(tasks) - completed} task(s) remaining."
            )

    _clear_progress()
    for proc in workers:
        proc.join(timeout=5)

    _merge_sharded_outputs(
        model_key, output_dir, longbench_e, shard_groups, collect_block_hit_rate
    )
    if collect_block_hit_rate and unsharded_datasets:
        pred_dir = Path(output_dir) / ("pred_e" if longbench_e else "pred") / model_key
        _print_saved_block_hit_rate_summaries("LongBench", pred_dir, unsharded_datasets)
    return results


def _run_predictions_sequential(model_key, datasets, max_length, output_dir,
                                sparda, infinigen, indexer_path, longbench_e,
                                resume=False, collect_block_hit_rate=False):
    """Run all datasets sequentially in a single pred.py invocation."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "pred.py"),
        "--model", model_key,
        "--max-length", str(max_length),
        "--output-dir", str(output_dir),
    ]
    if sparda:
        cmd.append("--sparda")
    if infinigen:
        cmd.append("--infinigen")
    if indexer_path:
        cmd.extend(["--indexer_path", indexer_path])
    if longbench_e:
        cmd.append("--e")
    if resume:
        cmd.append("--resume")
    if collect_block_hit_rate:
        cmd.append("--collect_block_hit_rate")
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Single-Doc QA":  ["narrativeqa", "multifieldqa_en", "multifieldqa_zh", "qasper"],
    "Multi-Doc QA":   ["hotpotqa", "2wikimqa", "musique"],
    "Summarization":  ["gov_report", "qmsum", "multi_news", "samsum", "vcsum", "dureader"],
    "Few-shot":       ["trec", "triviaqa", "lsht"],
    "Synthetic":      ["passage_retrieval_en", "passage_count", "passage_retrieval_zh"],
    "Code":           ["lcc", "repobench-p"],
}


def _print_results(result_path: Path, longbench_e: bool = False):
    """Print per-dataset scores, category averages, and overall average."""
    if not result_path.exists():
        print(f"[LongBench] WARNING: result file not found: {result_path}")
        return

    scores = json.loads(result_path.read_text())
    if not scores:
        return

    sep = "=" * 55
    thin = "-" * 55
    print(f"\n{sep}")
    print("LongBench Results" + (" (E)" if longbench_e else ""))
    print(f"{sep}\n")

    if longbench_e:
        buckets = ["0-4k", "4-8k", "8k+"]
        header = f"{'Dataset':<28} | " + " | ".join(f"{b:>6}" for b in buckets)
        print(header)
        print(thin)
        all_vals = {b: [] for b in buckets}
        for dataset in sorted(scores):
            row = scores[dataset]
            parts = " | ".join(f"{row.get(b, 0):>6.2f}" for b in buckets)
            print(f"{dataset:<28} | {parts}")
            for b in buckets:
                if b in row:
                    all_vals[b].append(row[b])
        print(thin)
        avg_parts = " | ".join(
            f"{(sum(v) / len(v)):>6.2f}" if v else f"{'N/A':>6}"
            for v in (all_vals[b] for b in buckets)
        )
        print(f"{'AVERAGE':<28} | {avg_parts}")
    else:
        print(f"{'Dataset':<28} | {'Score':>8}")
        print(thin)
        for dataset in sorted(scores):
            print(f"{dataset:<28} | {scores[dataset]:>8.2f}")
        print(thin)

        cat_avgs = {}
        for cat, members in CATEGORIES.items():
            vals = [scores[d] for d in members if d in scores]
            if vals:
                cat_avgs[cat] = sum(vals) / len(vals)

        if cat_avgs:
            overall = sum(scores.values()) / len(scores)
            print(f"{'OVERALL AVERAGE':<28} | {overall:>8.2f}")
            print(f"\n{'Category Averages':}")
            print("-" * 38)
            for cat, avg in cat_avgs.items():
                print(f"  {cat:<22}: {avg:>8.2f}")

    print(f"\n{sep}")


# ---------------------------------------------------------------------------
# Reclean existing predictions
# ---------------------------------------------------------------------------

def _reclean_predictions(model_key: str, method: str, longbench_e: bool,
                         output_root: Path):
    """Re-apply post_process to saved .jsonl prediction files in-place."""
    from pred import post_process

    pred_subdir = "pred_e" if longbench_e else "pred"
    pred_dir = output_root / pred_subdir / model_key
    if not pred_dir.exists():
        print(f"[LongBench] No predictions found at {pred_dir}")
        return

    jsonl_files = sorted(pred_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[LongBench] No .jsonl files in {pred_dir}")
        return

    total_changed = 0
    for fpath in jsonl_files:
        lines = fpath.read_text().strip().split("\n")
        new_lines = []
        changed = 0
        for line in lines:
            obj = json.loads(line)
            original = obj["pred"]
            cleaned = post_process(original, model_key)
            if cleaned != original:
                obj["pred"] = cleaned
                changed += 1
            new_lines.append(json.dumps(obj, ensure_ascii=False))
        if changed:
            fpath.write_text("\n".join(new_lines) + "\n")
            total_changed += changed
            print(f"  [{fpath.stem}] cleaned {changed}/{len(lines)} predictions")

    print(f"[LongBench] Reclean: {total_changed} predictions updated across "
          f"{len(jsonl_files)} files")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LongBench: predict and evaluate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default=None,
        help="Model key (e.g. 8b_nosa_sft). See config/model2path.json.",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Direct HuggingFace model path. When given, --model is auto-derived.",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Force dense (full) attention. By default MiniCPM models use sparse.",
    )
    parser.add_argument(
        "--sparda", action="store_true",
        help="Enable SparDA sparse attention with trained q_future indexer. "
             "Requires --indexer-path.",
    )
    parser.add_argument(
        "--infinigen", action="store_true",
        help="Enable InfiniGen decode (InfLLMv2 sparse prefill + InfiniGen masked decode). "
             "Only for MiniCPM/NOSA models. Overrides --sparda.",
    )
    parser.add_argument(
        "--indexer-path", default=None,
        help="Path to .pt indexer checkpoint. "
             "Required when --sparda is set.",
    )
    parser.add_argument(
        "--long-context", action="store_true",
        help="Enable long-context extension (LongRoPE / YaRN).",
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Max input length for truncation. "
             "Defaults to model native max (nosa=32768, minicpm=65536).",
    )
    parser.add_argument(
        "--gpus", default="auto",
        help="Comma-separated GPU IDs to use (default: auto). "
        "'auto' detects all available GPUs via CUDA_VISIBLE_DEVICES or "
             "nvidia-smi. When multiple GPUs are available, persistent workers "
             "load the model once and heavy datasets may be sharded automatically.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip prediction; evaluate existing outputs only.",
    )
    parser.add_argument(
        "--reclean", action="store_true",
        help="Re-apply post_process to existing prediction files before "
             "evaluation (no re-inference needed). Implies --eval-only.",
    )
    parser.add_argument(
        "--longbench-e", action="store_true",
        help="Evaluate on LongBench-E (length-stratified scoring).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Root directory for LongBench outputs. "
             "Predictions are stored under <output-dir>/pred[/_e]/<model>/ "
             "(default: benchmark script directory).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run: keep existing predictions and only "
             "run missing samples. Without this flag, all old predictions, "
             "logs, and summaries are cleared.",
    )
    parser.add_argument(
        "--block-hit-rate", action="store_true",
        help="Collect and print block hit-rate stats per dataset "
             "(NOSA/MiniCPM sparse models only).",
    )
    args = parser.parse_args()

    if args.reclean:
        args.eval_only = True

    if args.model is None and args.model_path is None:
        parser.error("Provide --model or --model-path.")

    # Validate flag combinations
    if args.dense and args.sparda:
        parser.error("--dense and --sparda are mutually exclusive.")
    if getattr(args, 'infinigen', False) and args.dense:
        parser.error("--dense and --infinigen are mutually exclusive.")
    if getattr(args, 'infinigen', False) and args.sparda:
        print("WARNING: --infinigen overrides --sparda. Ignoring --sparda.")
        args.sparda = False
    if args.sparda and args.indexer_path is None:
        parser.error("--indexer-path is required when --sparda is set.")
    if args.indexer_path and not os.path.exists(args.indexer_path):
        parser.error(f"--indexer-path does not exist: {args.indexer_path}")

    # Resolve model key and path
    if args.model_path:
        model_key = args.model or Path(args.model_path).name.lower().replace(".", "-")
        model_path = normalize_local_path(args.model_path)
    else:
        model_key = args.model
        cfg = json.loads((SCRIPT_DIR / "config" / "model2path.json").read_text())
        model_path = normalize_local_path(cfg.get(model_key, ""))
    args.indexer_path = normalize_local_path(args.indexer_path)

    # Auto long-context
    method = detect_method(model_key, model_path)

    # Compute mode and incorporate into model_key for output isolation
    if method == "nosa":
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
    elif method == "minicpm":
        mode = "minicpm"
    else:
        mode = "dense"
    if not model_key.endswith(mode):
        model_key = f"{model_key}_{mode}"
    native_max = NATIVE_MAX.get(method, 0)
    if args.max_length is None:
        args.max_length = native_max or 16400
    if not args.long_context and native_max and args.max_length > native_max:
        args.long_context = True
        print(f"[Auto] --long-context enabled: max_length {args.max_length} > "
              f"native max {native_max} for {method}")

    # Register model path so pred.py can find it
    if args.model_path:
        register_model(model_key, model_path, args.max_length)

    # GPU detection
    gpus = _detect_gpus(args.gpus)
    num_gpus = len(gpus)
    parallel = num_gpus > 1

    print(f"[LongBench] model_key={model_key}, path={model_path}")
    print(f"[LongBench] mode={mode}, method={method}, sparda={args.sparda}, "
          f"long_context={args.long_context}, max_length={args.max_length}")
    print(f"[LongBench] GPUs={gpus} "
          f"({'parallel, ' + str(num_gpus) + ' workers' if parallel else 'sequential'})")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else SCRIPT_DIR.resolve()
    )
    print(f"[LongBench] output_root={output_root}")

    if not args.eval_only:
        pred_subdir = "pred_e" if args.longbench_e else "pred"
        pred_dir = output_root / pred_subdir / model_key
        if pred_dir.exists():
            if args.resume:
                # --resume: keep prediction files and block stats, clear only result.json
                old = list(pred_dir.glob("result.json"))
            else:
                # Default: clear everything for a fresh run
                old = list(pred_dir.glob("*.jsonl")) + list(pred_dir.glob("result.json"))
                old += list(pred_dir.glob("*_block_stats.json"))
                old += list((pred_dir / "logs").glob("*.log"))
                old += list((pred_dir / "logs").glob("*.progress"))
                shard_dir = pred_dir / "shards"
                if shard_dir.exists():
                    shutil.rmtree(shard_dir)
                    print(f"[LongBench] Cleared shard cache in {shard_dir}")
            if old:
                for f in old:
                    f.unlink()
                mode = "result files only" if args.resume else "all results"
                print(f"[LongBench] Cleared {len(old)} file(s) ({mode}) in {pred_dir}")

    ensure_data(args.longbench_e)

    if not args.eval_only:
        print(f"\n[LongBench] Predicting: {model_key}")
        t0 = time.time()

        if parallel:
            results = _run_predictions_parallel(
                model_key, DATASETS, args.max_length, output_root, gpus,
                args.sparda, getattr(args, 'infinigen', False),
                args.indexer_path, args.longbench_e,
                resume=args.resume,
                collect_block_hit_rate=args.block_hit_rate)
            wall_time = time.time() - t0
            sum_time = sum(e for e, _ in results.values())
            failed = {task: rc for task, (_, rc) in results.items() if rc != 0}
            print(f"\n[LongBench] Prediction complete: wall-time {wall_time:.1f}s  "
                  f"(sum across GPUs: {sum_time:.1f}s)")
            if failed:
                raise RuntimeError(
                    f"[LongBench] {len(failed)} prediction task(s) failed: "
                    f"{list(failed.keys())}"
                )
        else:
            _run_predictions_sequential(
                model_key, DATASETS, args.max_length, output_root,
                args.sparda, getattr(args, 'infinigen', False),
                args.indexer_path, args.longbench_e,
                resume=args.resume,
                collect_block_hit_rate=args.block_hit_rate)
            wall_time = time.time() - t0
            print(f"\n[LongBench] Prediction complete: {wall_time:.1f}s")

    if args.reclean:
        _reclean_predictions(model_key, method, args.longbench_e, output_root)

    print(f"\n[LongBench] Evaluating: {model_key}")
    eval_cmd = [sys.executable, str(SCRIPT_DIR / "eval.py"),
                "--model", model_key,
                "--output-dir", str(output_root)]
    if args.longbench_e:
        eval_cmd.append("--e")
    subprocess.run(eval_cmd, cwd=str(SCRIPT_DIR), check=True)

    pred_subdir = "pred_e" if args.longbench_e else "pred"
    result_path = output_root / pred_subdir / model_key / "result.json"
    _print_results(result_path, longbench_e=args.longbench_e)

    print("\n[LongBench] Done.")


if __name__ == "__main__":
    main()
