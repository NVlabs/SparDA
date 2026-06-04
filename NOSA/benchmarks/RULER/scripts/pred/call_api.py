# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl files with field `pred`.

This script supports both:
- single-task mode via `--task`
- multi-task worker mode via `--tasks`, which reuses one loaded model
  across multiple RULER tasks
"""

import argparse
import contextlib
import importlib
import json
import json as _json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from tqdm import tqdm
import yaml


def read_manifest(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(_json.loads(line))
    return data


def _write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(path)


def _write_progress(progress_file, task_name, current, total):
    if not progress_file:
        return
    _write_json_atomic(
        Path(progress_file),
        {"task": task_name, "current": int(current), "total": int(total)},
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="path to load the dataset jsonl files")
    parser.add_argument("--save_dir", type=Path, required=True,
                        help="path to save the prediction jsonl files")
    parser.add_argument("--benchmark", type=str, default="synthetic",
                        help="Options: [synthetic]")
    parser.add_argument("--task", type=str, default=None,
                        help="Single task to run.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Multiple tasks to run sequentially with one model load.")
    parser.add_argument("--subset", type=str, default="validation",
                        help="Options: validation or test")
    parser.add_argument("--chunk_idx", type=int, default=0,
                        help="index of current split chunk")
    parser.add_argument("--chunk_amount", type=int, default=1,
                        help="size of split chunk")
    parser.add_argument("--task_key", type=str, default=None,
                        help="Optional unique output key for this task/chunk.")

    parser.add_argument("--server_type", default="hf", choices=["hf", "shadowkv"],
                        help="Inference backend (default: hf). RULER does not expose the nosi backend.")
    parser.add_argument("--model_name_or_path", type=str, default="gpt-3.5-turbo",
                        help="HF model repo ID or local checkpoint path.")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--stop_words", type=str, default="")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=32768,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--enable_sparse", action="store_true",
                        help="Enable infllmv2 sparse attention with q_current for top-k (loads modeling_minicpm.py)")
    parser.add_argument("--sparda", dest="enable_sparda", action="store_true",
                        help="Enable SparDA sparse attention with q_future for top-k (loads modeling_minicpm.py); use with --indexer_path")
    parser.add_argument("--dense", action="store_true",
                        help="Force dense inference mode when supported by the backend/model wrapper.")
    parser.add_argument("--indexer_path", type=str, default=None,
                        help="Path to .pt checkpoint with indexer weights for --sparda (q_future_proj/q_curr_proj)")
    parser.add_argument("--long_context", action="store_true",
                        help="Enable 128K long-context extension (LongRoPE for InfLLMv2, rope_theta increase for NOSA)")
    parser.add_argument("--yarn", action="store_true",
                        help="Use YaRN instead of rope_theta increase for NOSA long-context")
    parser.add_argument("--enable_infinigen", action="store_true",
                        help="Enable InfiniGen decode (InfLLMv2 sparse prefill + InfiniGen masked decode)")
    parser.add_argument("--progress_file", type=str, default=None,
                        help="File to write current task progress for external monitoring.")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Optional JSON file to record per-task elapsed time and exit status.")
    parser.add_argument("--log_dir", type=Path, default=None,
                        help="Optional directory for per-task logs.")
    parser.add_argument("--collect_block_hit_rate", action="store_true",
                        help="Collect block hit-rate stats (sparse models only).")

    args = parser.parse_args(argv)
    if (args.task is None) == (args.tasks is None):
        parser.error("Provide exactly one of --task or --tasks.")

    args.tasks = [args.task] if args.tasks is None else list(args.tasks)
    args.stop_words = list(filter(None, args.stop_words.split(",")))
    # Both HF and ShadowKV use a single model instance; keep one worker thread.
    args.threads = 1
    return args


def _load_task_catalog(benchmark):
    curr_folder = Path(__file__).resolve().parent
    parent_dir = str(curr_folder.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    module = importlib.import_module(f"data.{benchmark}.constants")
    with open(curr_folder.parent / f"{benchmark}.yaml", "r") as f:
        tasks_customized = yaml.safe_load(f)
    return module.TASKS, tasks_customized


def _build_task_config(task_name, tasks_base, tasks_customized):
    if task_name not in tasks_customized:
        raise ValueError(f"{task_name} is not found in config_tasks.yaml")
    config = dict(tasks_customized.get(task_name))
    config.update(tasks_base[config["task"]])
    return config


def _split_ranges(total_size, num_parts):
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


def _task_output_stem(args, task_name):
    if args.task_key:
        return args.task_key
    if args.chunk_amount > 1:
        return f"{task_name}.chunk{args.chunk_idx + 1}of{args.chunk_amount}"
    return task_name


def _task_paths(args, task_name):
    task_file = args.data_dir / task_name / f"{args.subset}.jsonl"
    stem = _task_output_stem(args, task_name)
    pred_file = args.save_dir / f"{stem}.jsonl"
    stats_path = args.save_dir / f"{stem}_block_stats.json"
    log_path = args.log_dir / f"{stem}.log" if args.log_dir else None
    return task_file, pred_file, stats_path, log_path


def _load_pending_samples(args, task_file, pred_file):
    data = read_manifest(task_file)
    if args.chunk_amount > 1:
        ranges = _split_ranges(len(data), args.chunk_amount)
        if args.chunk_idx < 0 or args.chunk_idx >= len(ranges):
            raise ValueError(
                f"chunk_idx {args.chunk_idx} out of range for chunk_amount={args.chunk_amount}"
            )
        start, end = ranges[args.chunk_idx]
        data = data[start:end]

    if pred_file.exists():
        pred_index = [sample["index"] for sample in read_manifest(pred_file)]
        data = [sample for sample in data if sample["index"] not in pred_index]
    return data


def _configure_llm_for_task(llm, args, tokens_to_generate):
    if args.server_type == "shadowkv":
        llm.tokens_to_generate = tokens_to_generate
        return

    if hasattr(llm, "generation_kwargs"):
        llm.generation_kwargs["max_new_tokens"] = tokens_to_generate
    if hasattr(llm, "max_genlen"):
        llm.max_genlen = tokens_to_generate


def get_llm(args, tokens_to_generate):
    if args.server_type == "shadowkv":
        shadowkv_root = os.environ.get("SHADOWKV_ROOT")
        if not shadowkv_root:
            raise ValueError("ShadowKV is not bundled. Set SHADOWKV_ROOT to an external ShadowKV checkout.")
        sys.path.insert(0, shadowkv_root)
        from models import Llama

        llm = Llama(
            model_name=args.model_name_or_path,
            sparse_budget=3072,
            attn_mode="shadowkv",
            rank=40,
            chunk_size=8,
            minference=True,
        )
        llm.tokens_to_generate = tokens_to_generate
        return llm

    from model_wrappers import HuggingFaceModel

    return HuggingFaceModel(
        name_or_path=args.model_name_or_path,
        do_sample=args.temperature > 0,
        repetition_penalty=1,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop=args.stop_words,
        max_new_tokens=tokens_to_generate,
        enable_sparse=args.enable_sparse,
        enable_sparda=args.enable_sparda,
        enable_dense=args.dense,
        enable_infinigen=getattr(args, "enable_infinigen", False),
        indexer_path=args.indexer_path,
        long_context=args.long_context,
        yarn=getattr(args, "yarn", False),
        max_seq_length=args.max_seq_length,
    )


def _create_collector(args, llm):
    if not args.collect_block_hit_rate or not hasattr(llm, "model"):
        return None

    try:
        bench_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        from block_hit_rate import BlockHitRateCollector

        collector = BlockHitRateCollector(llm.model)
        if not collector.available:
            return None
        collector.install()
        return collector
    except ImportError:
        return None


def _reset_collector(collector):
    if collector is not None and getattr(collector, "_tracker", None) is not None:
        collector._tracker.reset()


def _save_collector_stats(task_name, collector, stats_path):
    if collector is None:
        return
    stats = collector.get_stats()
    if stats and stats.get("total_blocks", 0) > 0:
        _write_json_atomic(stats_path, stats)
        print(f"[{task_name}] Block hit-rate: {stats['hit_rate'] * 100:.1f}%  "
              f"(saved to {stats_path})")
    _reset_collector(collector)


def _predict_dataset_batches(args, llm, collector, task_name, data, pred_file):
    thread_error = [None]
    outputs_parallel = [{} for _ in range(len(data))]

    def get_output(idx_list, index_list, input_list, outputs_list, others_list,
                   truncation_list, length_list):
        if collector is not None:
            collector.begin_sample()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if args.server_type == "shadowkv":
                    input_ids = llm.tokenizer(input_list, return_tensors="pt").input_ids.to(llm.device)
                    output = llm.generate(
                        input_ids,
                        gen_len=llm.tokens_to_generate,
                        temperature=args.temperature,
                    )
                    pred_list = [{"text": output_str} for output_str in output]
                else:
                    pred_list = llm.process_batch(prompts=input_list)
                break
            except OSError:
                traceback.print_exc()
                thread_error[0] = sys.exc_info()
                raise
            except Exception as exc:
                traceback.print_exc()
                if attempt == max_retries - 1:
                    thread_error[0] = sys.exc_info()
                    raise RuntimeError(
                        f"Inference failed after {max_retries} attempts"
                    ) from exc

        if collector is not None:
            collector.end_sample()

        zipped_iter = zip(
            pred_list,
            idx_list,
            index_list,
            input_list,
            outputs_list,
            others_list,
            truncation_list,
            length_list,
        )
        for pred, idx, index, input_text, outputs, others, truncation, length in zipped_iter:
            if isinstance(pred["text"], str):
                pred_text = pred["text"]
            elif len(pred["text"]) > 0:
                pred_text = pred["text"][0]
            else:
                pred_text = ""

            outputs_parallel[idx] = {
                "index": index,
                "pred": pred_text,
                "input": input_text,
                "outputs": outputs,
                "others": others,
                "truncation": truncation,
                "length": length,
            }

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        data_point["idx"] = idx
        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []
        batch.append(data_point)
    if batch:
        batched_data.append(batch)

    with open(pred_file, "at", encoding="utf-8", buffering=1) as fout:
        start_idx = 0
        threads = []

        for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
            idx_list = [data_point["idx"] for data_point in batch]
            end_idx = idx_list[-1]

            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    idx_list=idx_list,
                    index_list=[data_point["index"] for data_point in batch],
                    input_list=[data_point["input"] + data_point.get("answer_prefix", "")
                                for data_point in batch],
                    outputs_list=[data_point["outputs"] for data_point in batch],
                    others_list=[data_point.get("others", {}) for data_point in batch],
                    truncation_list=[data_point.get("truncation", -1) for data_point in batch],
                    length_list=[data_point.get("length", -1) for data_point in batch],
                ),
            )
            thread.start()
            threads.append(thread)

            is_last_batch = (batch_idx == len(batched_data) - 1)
            if len(threads) == args.threads or is_last_batch:
                for worker in threads:
                    worker.join()
                threads = []

                if thread_error[0] is not None:
                    raise thread_error[0][1].with_traceback(thread_error[0][2])

                for idx in range(start_idx, end_idx + 1):
                    if outputs_parallel[idx]:
                        fout.write(json.dumps(outputs_parallel[idx]) + "\n")
                start_idx = end_idx + 1
                _write_progress(args.progress_file, task_name, end_idx + 1, len(data))


def _run_single_task(args, llm, collector, task_name, task_config):
    task_file, pred_file, stats_path, log_path = _task_paths(args, task_name)
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    _configure_llm_for_task(llm, args, task_config["tokens_to_generate"])
    data = _load_pending_samples(args, task_file, pred_file)
    total_samples = len(data)
    task_start = time.time()

    stream_ctx = contextlib.nullcontext()
    log_handle = None
    if log_path is not None:
        log_handle = open(log_path, "w", encoding="utf-8")
        stream_ctx = contextlib.ExitStack()
        stream_ctx.enter_context(contextlib.redirect_stdout(log_handle))
        stream_ctx.enter_context(contextlib.redirect_stderr(log_handle))

    try:
        with stream_ctx:
            print(f"Predict {task_name}\nfrom {task_file}\nto {pred_file}")
            if collector is not None:
                print(f"[{task_name}] Block hit-rate collection enabled")

            if total_samples == 0:
                print(f"[{task_name}] Nothing to do: predictions already exist.")
                _write_progress(args.progress_file, task_name, 0, 0)
            else:
                _write_progress(args.progress_file, task_name, 0, total_samples)
                _predict_dataset_batches(args, llm, collector, task_name, data, pred_file)

            _save_collector_stats(task_name, collector, stats_path)
            print(f"[{task_name}] Used time: {round((time.time() - task_start) / 60, 1)} minutes")
    finally:
        if log_handle is not None:
            log_handle.close()

    return {
        "elapsed": time.time() - task_start,
        "returncode": 0,
        "log_file": str(log_path) if log_path is not None else "",
    }


def main(argv=None):
    args = parse_args(argv)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    tasks_base, tasks_customized = _load_task_catalog(args.benchmark)
    task_names = list(args.tasks)

    llm = None
    collector = None
    task_results = {}

    try:
        for task_name in task_names:
            task_config = _build_task_config(task_name, tasks_base, tasks_customized)
            if llm is None:
                llm = get_llm(args, task_config["tokens_to_generate"])
                collector = _create_collector(args, llm)

            try:
                task_results[task_name] = _run_single_task(
                    args, llm, collector, task_name, task_config
                )
            except Exception:
                task_file, _, _, log_path = _task_paths(args, task_name)
                task_results[task_name] = {
                    "elapsed": 0.0,
                    "returncode": 1,
                    "log_file": str(log_path) if log_path is not None else "",
                    "task_file": str(task_file),
                }
                if args.results_file:
                    _write_json_atomic(Path(args.results_file), task_results)
                raise

            if args.results_file:
                _write_json_atomic(Path(args.results_file), task_results)

    finally:
        if collector is not None:
            collector.uninstall()

    if args.results_file:
        _write_json_atomic(Path(args.results_file), task_results)


if __name__ == "__main__":
    main()
