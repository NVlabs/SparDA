#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reasoning benchmark runner: download data, predict, and evaluate.

Thinking mode is ON by default. NOSA uses the legacy two-shot `<think>` prompt,
while MiniCPM uses a lightweight official-style prompt with native thinking
enabled through its chat template. Default sampling is:
- NOSA: temperature=1.0, top_p=1.0
- MiniCPM: temperature=0.9, top_p=0.95

Data is auto-downloaded on first run if not present.

Usage:
    # NOSA 8B
    python run.py --model 8b_nosa --model-path openbmb/NOSA-8B

    # MiniCPM 8B
    python run.py --model 8b_minicpm --model-path openbmb/MiniCPM4.1-8B

    # Custom settings
    python run.py --model 8b_nosa --model-path /path/to/model \\
                  --dataset math-500 --gen-len 8192 --batch-size 1

    # Evaluate only (predictions already exist)
    python run.py --model 8b_nosa --eval-only

    # Predict only (skip evaluation)
    python run.py --model 8b_nosa --model-path openbmb/NOSA-8B --predict-only
"""

import argparse
import fcntl
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_{os.getpid()}")

SCRIPT_DIR = Path(__file__).resolve().parent

KNOWN_MODELS = {
    "8b_nosa": "openbmb/NOSA-8B",
    "8b_minicpm": "openbmb/MiniCPM4.1-8B",
    "8b_fullattn": "",
    "8b_dma": "",
}

DATASET_SOURCES = {
    "math-500": {
        "file": "math-500.jsonl",
        "url": "https://media.githubusercontent.com/media/openai/prm800k/refs/heads/main/prm800k/math_splits/test.jsonl?download=true",
    },
    "aime24": {"file": "aime24.jsonl", "hf_repo": "math-ai/aime24", "hf_split": "test"},
    "aime25": {"file": "aime25.jsonl", "hf_repo": "math-ai/aime25", "hf_split": "test"},
}


def _download_hf_math(hf_repo: str, hf_split: str, out_path: Path):
    """Download a math dataset from HuggingFace and save as JSONL."""
    import json
    from datasets import load_dataset

    temp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.part")
    ds = load_dataset(hf_repo, split=hf_split)
    with open(temp_path, "w") as f:
        for idx, row in enumerate(ds):
            problem = row.get("problem") or row.get("question") or ""
            answer = row.get("answer") or row.get("final_answer") or ""
            solution = row.get("solution") or row.get("rationale") or ""
            rec = {"problem": str(problem), "answer": str(answer)}
            if solution:
                rec["solution"] = str(solution)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    temp_path.replace(out_path)
    print(f"[Reasoning] Saved {len(ds)} samples to {out_path}")


def _canonical_answer(row: dict) -> str:
    answer = row.get("answer") or row.get("final_answer")
    if answer:
        return str(answer)

    solution = row.get("solution") or row.get("rationale")
    if solution:
        return str(solution)

    return ""


def _normalize_dataset_answers(out_path: Path):
    temp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.normalize")
    updated = False
    with open(out_path, "r", encoding="utf-8") as src_fp, open(temp_path, "w", encoding="utf-8") as dst_fp:
        for line in src_fp:
            record = json.loads(line)
            canonical_answer = _canonical_answer(record)
            if record.get("answer", "") != canonical_answer:
                record["answer"] = canonical_answer
                updated = True
            dst_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    if updated:
        temp_path.replace(out_path)
        print(f"[Reasoning] Normalized answer fields in {out_path.name}.")
    else:
        temp_path.unlink()


def _download_url_to_path(url: str, out_path: Path):
    temp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.part")
    try:
        subprocess.run(["curl", "-L", url, "-o", str(temp_path)], check=True)
        temp_path.replace(out_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def ensure_data(dataset: str):
    src = DATASET_SOURCES.get(dataset)
    if src is None or src.get("file") is None:
        return
    out_path = SCRIPT_DIR / src["file"]
    if out_path.exists() and out_path.stat().st_size > 0:
        _normalize_dataset_answers(out_path)
        return

    lock_path = SCRIPT_DIR / f".{src['file']}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[Reasoning] {dataset} ready.")
            return
        if "url" in src:
            print(f"[Reasoning] {dataset} not found, downloading...")
            _download_url_to_path(src["url"], out_path)
        elif "hf_repo" in src:
            print(f"[Reasoning] {dataset} not found, downloading from {src['hf_repo']}...")
            _download_hf_math(src["hf_repo"], src["hf_split"], out_path)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"[Reasoning] Failed to prepare dataset: {out_path}")
    print(f"[Reasoning] {dataset} ready.")
    _normalize_dataset_answers(out_path)


def detect_method(model_key: str, model_path: str) -> str:
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


def _clear_fresh_outputs(results_root: Path, logs_root: Path, datasets, model_mode_key: str):
    cleared = []

    for ds in datasets:
        dataset_dir = results_root / ds / model_mode_key
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            cleared.append(dataset_dir)

        dataset_log_dir = logs_root / ds / model_mode_key
        if dataset_log_dir.exists():
            shutil.rmtree(dataset_log_dir)
            cleared.append(dataset_log_dir)

    multi_log_dir = logs_root / "multi" / model_mode_key
    if multi_log_dir.exists():
        shutil.rmtree(multi_log_dir)
        cleared.append(multi_log_dir)

    summary_dir = results_root / "_summary" / model_mode_key
    if summary_dir.exists():
        shutil.rmtree(summary_dir)
        cleared.append(summary_dir)

    if cleared:
        print(
            f"[Reasoning] Cleared {len(cleared)} stale path(s) for fresh run of {model_mode_key}."
        )
        for path in cleared:
            print(f"[Reasoning]   removed: {path}")


def _expected_dataset_total(dataset: str, data_path_override: str | None = None) -> int:
    dataset_path = Path(data_path_override) if data_path_override else (SCRIPT_DIR / DATASET_SOURCES[dataset]["file"])
    with dataset_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _prediction_completion_stats(results_root: Path, dataset: str, model_mode_key: str, expected_total: int):
    dataset_dir = results_root / dataset / model_mode_key
    files = sorted(glob.glob(str(dataset_dir / "*.jsonl")))
    total_rows = 0
    completed_sample_ids = set()

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_idx = record.get("_sample_idx")
                if isinstance(sample_idx, int) and 0 <= sample_idx < expected_total:
                    completed_sample_ids.add(sample_idx)

    completed = len(completed_sample_ids) if completed_sample_ids else total_rows
    is_complete = bool(files) and completed == expected_total and total_rows == expected_total
    return {
        "dataset_dir": dataset_dir,
        "files": files,
        "completed": completed,
        "expected_total": expected_total,
        "total_rows": total_rows,
        "has_sample_ids": bool(completed_sample_ids),
        "is_complete": is_complete,
    }


def _ensure_prediction_complete(results_root: Path, datasets, model_mode_key: str, data_path_override: str | None = None):
    problems = []

    for dataset in datasets:
        expected_total = _expected_dataset_total(
            dataset,
            data_path_override=data_path_override if len(datasets) == 1 else None,
        )
        stats = _prediction_completion_stats(results_root, dataset, model_mode_key, expected_total)
        if stats["is_complete"]:
            continue

        files_desc = ", ".join(Path(path).name for path in stats["files"]) if stats["files"] else "no files"
        problems.append(
            f"{dataset}: completed={stats['completed']}/{stats['expected_total']}, "
            f"rows={stats['total_rows']}, files={files_desc}"
        )

    if problems:
        joined = "\n".join(f"  - {problem}" for problem in problems)
        raise RuntimeError(
            "[Reasoning] Prediction outputs are incomplete or inconsistent; refusing evaluation.\n"
            f"{joined}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning benchmark: predict and evaluate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help="Model key (e.g. 8b_nosa, 8b_minicpm).",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="HuggingFace model path. Auto-resolved from model key if known.",
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
        help="Path to .pt checkpoint with indexer weights. "
             "Required when --sparda is set.",
    )
    parser.add_argument(
        "--dataset", default="all",
        help="Comma-separated datasets or 'all' (default: all). "
             "Choices: math-500, aime24, aime25, all.",
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Override dataset file path (single-dataset mode only).",
    )
    parser.add_argument(
        "--gen-len", type=int, default=None,
        help="Max generation length. Defaults to 8192 for NOSA, 65536 for MiniCPM.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size. Reasoning currently forces 1 for all modes.",
    )
    parser.add_argument(
        "--gpus", default="0,1,2,3,4,5,6,7",
        help="CUDA devices (default: 0,1,2,3,4,5,6,7).",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix for results/logs directories when --output-dir is not set.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Explicit root directory for reasoning outputs. "
             "Results are stored under <output-dir>/<dataset>/<model>/ and logs "
             "under <output-dir>/_logs/.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip prediction; run exact-match and judge evaluation only.",
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Run prediction only; skip exact-match and judge evaluation.",
    )
    parser.add_argument(
        "--skip-judge-eval", action="store_true",
        help="Skip GPT-5.2 judge evaluation and only report deterministic exact-match scores.",
    )
    parser.add_argument(
        "--judge-model", default=os.environ.get("REASONING_JUDGE_MODEL", "gpt-5.2"),
        help="Judge model for answer-presence evaluation "
             "(default: REASONING_JUDGE_MODEL or gpt-5.2; gpt-5.2 auto-resolves like HELMET).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run: continue from existing prediction "
             "JSONL files when possible and skip datasets that are already complete. "
             "Without this flag, stale outputs for the selected datasets/model are "
             "cleared before prediction starts.",
    )
    parser.add_argument(
        "--block-hit-rate", action="store_true",
        help="Collect and print block hit-rate stats "
             "(NOSA/MiniCPM sparse models only).",
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model_path is None:
        args.model_path = KNOWN_MODELS.get(args.model, "")
    args.model_path = normalize_local_path(args.model_path)
    args.data_path = normalize_local_path(args.data_path)
    args.indexer_path = normalize_local_path(args.indexer_path)
    if not args.model_path and not args.eval_only:
        parser.error(f"--model-path is required (no known path for '{args.model}').")

    # Validate flag combinations
    if args.dense and args.sparda:
        parser.error("--dense and --sparda are mutually exclusive.")
    if getattr(args, 'infinigen', False) and args.dense:
        parser.error("--dense and --infinigen are mutually exclusive.")
    if getattr(args, 'infinigen', False) and args.sparda:
        print("WARNING: --infinigen overrides --sparda. Ignoring --sparda.")
        args.sparda = False
    method = detect_method(args.model, args.model_path or "")
    is_nosa = method == "nosa"
    # InfiniGen now supports both MiniCPM and NOSA models
    if args.sparda and args.indexer_path is None:
        parser.error("--indexer-path is required when --sparda is set. "
                     "Provide a .pt checkpoint with q_future_proj/q_curr_proj weights.")
    if args.indexer_path and not os.path.exists(args.indexer_path):
        parser.error(f"--indexer-path does not exist: {args.indexer_path}")

    # Default gen-len: 8K for NOSA, 64K for MiniCPM
    if args.gen_len is None:
        args.gen_len = 8192 if is_nosa else 65536

    if args.batch_size != 1:
        print(f"[Reasoning] Overriding batch_size {args.batch_size} -> 1 for all modes")
        args.batch_size = 1

    results_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else (SCRIPT_DIR / f"results{args.output_suffix}").resolve()
    )
    logs_root = (
        (results_root / "_logs")
        if args.output_dir else (SCRIPT_DIR / f"logs{args.output_suffix}").resolve()
    )

    # Default temperature/top_p by method
    if is_nosa:
        temperature = 1.0
        top_p = 1.0
    else:
        temperature = 0.9
        top_p = 0.95

    # Compute mode for output isolation (matches RULER/LongBench convention)
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
    elif method == "minicpm":
        mode = "minicpm"
    else:
        mode = "dense"
    model_mode_key = f"{args.model}_{mode}" if not args.model.endswith(mode) else args.model

    # Parse dataset list
    ALL_DATASETS = ["math-500", "aime24", "aime25"]
    if args.dataset.strip().lower() == "all":
        datasets = list(ALL_DATASETS)
    else:
        datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    for ds in datasets:
        if ds not in DATASET_SOURCES:
            parser.error(f"Unknown dataset: {ds}. Choose from: {', '.join(ALL_DATASETS)}, all.")

    print(f"[Reasoning] model={args.model}, path={args.model_path}")
    print(f"[Reasoning] mode={mode}, method={method}, sparda={args.sparda}")
    print(f"[Reasoning] datasets={datasets}, gen_len={args.gen_len}, "
          f"temperature={temperature}, top_p={top_p}, batch_size={args.batch_size}")
    print(f"[Reasoning] judge_eval={'off' if args.skip_judge_eval else 'on'}, judge_model={args.judge_model}")

    if not args.resume and not args.eval_only:
        _clear_fresh_outputs(results_root, logs_root, datasets, model_mode_key)

    # Download data for all datasets
    if not args.eval_only:
        for ds in datasets:
            ensure_data(ds)

    # Predict
    if not args.eval_only:
        predict_datasets = list(datasets)
        if args.resume:
            print("[Reasoning] Resume enabled: existing prediction JSONL files "
                  "will be reused per dataset where possible.")

        if not predict_datasets:
            print("[Reasoning] All datasets already predicted, skipping prediction.")
        elif len(predict_datasets) == 1 and args.data_path is not None:
            # Single-dataset mode with explicit --data-path override
            ds = predict_datasets[0]
            save_dir = results_root / ds / model_mode_key
            log_dir = logs_root / ds / model_mode_key
            save_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{ds}-{model_mode_key}-rank_0.log"

            cmd = [
                sys.executable, "-u", str(SCRIPT_DIR / "test.py"),
                "--model_type", model_mode_key,
                "--data_type", ds,
                "--model_path", args.model_path,
                "--data_path", args.data_path,
                "--save_path", str(save_dir),
                "--gen_len", str(args.gen_len),
                "--temperature", str(temperature),
                "--top_p", str(top_p),
                "--batch_size", str(args.batch_size),
                "--cuda_devices", args.gpus,
            ]
            if args.dense:
                cmd.append("--dense")
            if args.sparda:
                cmd.append("--sparda")
            if getattr(args, 'infinigen', False):
                cmd.append("--infinigen")
            if args.indexer_path:
                cmd.extend(["--indexer_path", args.indexer_path])
            if args.block_hit_rate:
                cmd.append("--collect_block_hit_rate")
            if args.resume:
                cmd.append("--resume")

            print(f"\n[Reasoning] Predicting: {model_mode_key} on {ds}")
            print(f"[Reasoning] Log: {log_file}")
            with open(log_file, "w") as f_log:
                subprocess.run(cmd, stdout=f_log, stderr=f_log, cwd=str(SCRIPT_DIR), check=True)
            print(f"[Reasoning] Prediction done.")
        else:
            # Multi-dataset mode: single test.py invocation with --data_paths
            # Build data_paths arg and create per-dataset output dirs
            data_paths_parts = []
            for ds in predict_datasets:
                src = DATASET_SOURCES.get(ds, {})
                dp = str(SCRIPT_DIR / src["file"]) if src.get("file") else "DO_NOT_NEED"
                data_paths_parts.append(f"{ds}:{dp}")

                # Create per-dataset save/log dirs
                ds_save_dir = results_root / ds / model_mode_key
                ds_save_dir.mkdir(parents=True, exist_ok=True)

            data_paths_str = ",".join(data_paths_parts)

            # Use the first dataset's dir for logs
            log_dir = logs_root / "multi" / model_mode_key
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{'_'.join(predict_datasets)}-{model_mode_key}-rank_0.log"

            cmd = [
                sys.executable, "-u", str(SCRIPT_DIR / "test.py"),
                "--model_type", model_mode_key,
                "--model_path", args.model_path,
                "--data_paths", data_paths_str,
                "--save_path", str(results_root),
                "--gen_len", str(args.gen_len),
                "--temperature", str(temperature),
                "--top_p", str(top_p),
                "--batch_size", str(args.batch_size),
                "--cuda_devices", args.gpus,
            ]
            if args.dense:
                cmd.append("--dense")
            if args.sparda:
                cmd.append("--sparda")
            if getattr(args, 'infinigen', False):
                cmd.append("--infinigen")
            if args.indexer_path:
                cmd.extend(["--indexer_path", args.indexer_path])
            if args.block_hit_rate:
                cmd.append("--collect_block_hit_rate")
            if args.resume:
                cmd.append("--resume")

            print(f"\n[Reasoning] Predicting: {model_mode_key} on {predict_datasets}")
            print(f"[Reasoning] Log: {log_file}")
            with open(log_file, "w") as f_log:
                subprocess.run(cmd, stdout=f_log, stderr=f_log, cwd=str(SCRIPT_DIR), check=True)
            print(f"[Reasoning] Prediction done.")

        _ensure_prediction_complete(
            results_root,
            predict_datasets,
            model_mode_key,
            data_path_override=args.data_path,
        )

    # Evaluate
    if not args.predict_only:
        print(f"\n[Reasoning] Evaluating: {model_mode_key} on {datasets}")
        eval_cmd = [
            sys.executable, str(SCRIPT_DIR / "check.py"),
            "--model", model_mode_key,
            "--dataset", *datasets,
            "--output-dir", str(results_root),
        ]
        if args.skip_judge_eval:
            eval_cmd.append("--skip-judge-eval")
        else:
            eval_cmd.extend(["--judge-model", args.judge_model])
        if args.data_path is not None and len(datasets) == 1:
            eval_cmd.append("--allow-partial")
        subprocess.run(eval_cmd, cwd=str(SCRIPT_DIR), check=True)

    print("\n[Reasoning] Done.")


if __name__ == "__main__":
    main()
