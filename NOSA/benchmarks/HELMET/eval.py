# This file is copied/adapted from HELMET (https://github.com/princeton-nlp/HELMET).
# Copyright (c) 2024 Princeton Natural Language Processing.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import os
import warnings

# Default quiet mode for cleaner benchmark logs.
# Set HELMET_QUIET=0 to keep verbose library warnings/debug output.
HELMET_QUIET = os.environ.get("HELMET_QUIET", "1") != "0"
if HELMET_QUIET:
    warnings.filterwarnings("ignore", message=r"Using `TRANSFORMERS_CACHE` is deprecated.*")
    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")
    warnings.filterwarnings("ignore", message=r"`torch_dtype` is deprecated! Use `dtype` instead!")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")

import gc
import sys
import time
import json
import re
import fcntl
import random
import copy
import traceback
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
import logging
import yaml

BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)

MODEL_CONFIGS = {
    "8b_nosa_sft": "openbmb/NOSA-8B",
    "8b_fullattn_sft": "",
    "8b_minicpm_sft": "openbmb/MiniCPM4.1-8B",
    "8b_nosa_sft_pref": "",
    "8b_dma": "",
    "8b_dma_pref": "",
    "8b_nosa_sft_sb=8": "openbmb/NOSA-8B",
    "8b_nosa_sft_sb=16": "openbmb/NOSA-8B",
    "8b_nosa_sft_sb=24": "openbmb/NOSA-8B",
    "8b_nosa_sft_sb=32": "openbmb/NOSA-8B",
    "8b_nosa_sparda": "openbmb/NOSA-8B",
}

CATEGORIES = ["recall", "rag", "rerank", "cite", "longqa", "summ", "icl"]
LENGTH_SUFFIXES = {16384: "16k", 32768: "32k", 65536: "64k", 131072: "128k"}


def _resolve_length_suffix(input_max_length: int) -> str:
    """Map an input_max_length to the best matching config suffix."""
    for threshold in sorted(LENGTH_SUFFIXES.keys(), reverse=True):
        if input_max_length >= threshold:
            return LENGTH_SUFFIXES[threshold]
    return LENGTH_SUFFIXES[16384]


def _build_dataset_names(input_max_length: int) -> list:
    suffix = _resolve_length_suffix(input_max_length)
    return [f"{cat}_{suffix}" for cat in CATEGORIES]

DATASET_CONFIG_DIR = "configs"
SEEDS = [42]


def _detect_gpus(gpu_arg: str) -> list:
    """Return a list of GPU ID ints."""
    if gpu_arg != "auto":
        return [int(g.strip()) for g in gpu_arg.split(",") if g.strip()]

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip():
        return list(range(len(cuda_visible.split(","))))

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(g.strip()) for g in result.stdout.strip().split("\n") if g.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return [0]

from arguments import parse_arguments
from model_utils import load_LLM, OpenAIModel, AnthropicModel, TgiVllmModel, HFModel
from data import load_data

# Block hit-rate collection (shared across all benchmarks)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from block_hit_rate import (
    BlockHitRateCollector as _BlockHitRateCollector,
    merge_block_stats as _merge_block_stats_impl,
)

logging.basicConfig(
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.WARNING if HELMET_QUIET else logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING if HELMET_QUIET else logging.INFO)
if HELMET_QUIET:
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("eval_alce").setLevel(logging.WARNING)

def _progress_inc(counter, n=1):
    """Increment a shared mp.Value counter.  No-op when *counter* is None."""
    if counter is not None:
        with counter.get_lock():
            counter.value += n


def _merge_block_stats(stats_list):
    """Merge ``BlockStatsTracker.get_stats()`` dicts from multiple shards."""
    return _merge_block_stats_impl(stats_list)


def _print_block_hit_rate_summary(block_hr_data):
    """Print a per-category block hit-rate summary table.

    Args:
        block_hr_data: dict mapping (model_key, category) -> [(dataset, hit_rate%)].
    """
    if not block_hr_data:
        return

    w = 80
    print(f"\n{'=' * w}")
    print(f"  Block Hit-Rate Summary (per task)")
    print(f"{'=' * w}")
    print(f"  {'Category':<14} {'Dataset':<28} {'Hit-Rate':>10}")
    print(f"  {'-' * 14} {'-' * 28} {'-' * 10}")

    all_rates = []
    model_count = len({model_key for model_key, _ in block_hr_data})

    model_groups = {}
    for (model_key, category), entries in block_hr_data.items():
        model_groups.setdefault(model_key, {})[category] = entries

    for model_key, category_map in sorted(model_groups.items()):
        if model_count > 1:
            print(f"  Model: {model_key}")

        model_rates = []
        for category, entries in sorted(category_map.items()):
            cat_rates = []
            for dataset, hr in sorted(entries, key=lambda x: x[0]):
                print(f"  {category:<14} {dataset:<28} {hr:>9.1f}%")
                cat_rates.append(hr)
                model_rates.append(hr)
                all_rates.append(hr)

            if len(cat_rates) > 1:
                avg = np.mean(cat_rates)
                print(f"  {'':<14} {category + ' (avg)':<28} {avg:>9.1f}%")
            print(f"  {'-' * 14} {'-' * 28} {'-' * 10}")

        if model_count > 1 and model_rates:
            model_overall = np.mean(model_rates)
            print(f"  {'':<14} {(model_key + ' OVERALL'):<28} {model_overall:>9.1f}%")
            print(f"  {'-' * 14} {'-' * 28} {'-' * 10}")

    if all_rates:
        overall = np.mean(all_rates)
        print(f"  {'':<14} {'OVERALL':<28} {overall:>9.1f}%")
    print(f"{'=' * w}\n")


def _resolve_model_configs(args):
    if args.model_key and args.model_name_or_path:
        return {args.model_key: args.model_name_or_path}

    model_configs = {k: v for k, v in MODEL_CONFIGS.items() if v}
    if not model_configs:
        logger.error("No models with non-empty paths in MODEL_CONFIGS. "
                     "Use --model_key and --model_name_or_path, or fill in MODEL_CONFIGS.")
        return None
    return model_configs


def _build_tasks(args, model_configs, dataset_names):
    tasks = []

    for model_key, model_path in model_configs.items():
        os.makedirs(f"{args.output_dir}/{model_key}", exist_ok=True)

        for dataset_name in dataset_names:
            os.makedirs(f"{args.output_dir}/{model_key}/{dataset_name}", exist_ok=True)

            yaml_path = os.path.join(DATASET_CONFIG_DIR, f"{dataset_name}.yaml")
            if not os.path.exists(yaml_path):
                logger.warning(f"Config file not found for {dataset_name}: {yaml_path}, skipping...")
                continue

            with open(yaml_path, "r", encoding="utf-8") as f:
                ds_config = yaml.safe_load(f)

            datasets = ds_config["datasets"].split(",")
            test_files = ds_config["test_files"].split(",")
            demo_files = ds_config["demo_files"].split(",")

            max_lengths = ([int(ds_config["input_max_length"])] * len(datasets)) \
                if isinstance(ds_config["input_max_length"], int) or len(ds_config["input_max_length"].split(",")) == 1 \
                else [int(l) for l in ds_config["input_max_length"].split(",")]
            gen_lengths = ([int(ds_config["generation_max_length"])] * len(datasets)) \
                if isinstance(ds_config["generation_max_length"], int) or len(ds_config["generation_max_length"].split(",")) == 1 \
                else [int(l) for l in ds_config["generation_max_length"].split(",")]

            use_chat_template = ds_config["use_chat_template"]
            max_test_samples = args.max_test_samples if args.max_test_samples is not None else ds_config["max_test_samples"]
            shots = ds_config["shots"]
            stop_new_line = ds_config["stop_new_line"]

            for seed in SEEDS:
                tasks.extend([
                    (
                        model_key,
                        model_path,
                        dataset,
                        test_file,
                        demo_file,
                        seed,
                        int(max_length),
                        int(gen_length),
                        dataset_name,
                        use_chat_template,
                        max_test_samples,
                        shots,
                        stop_new_line,
                    )
                    for dataset, test_file, demo_file, max_length, gen_length
                    in zip(datasets, test_files, demo_files, max_lengths, gen_lengths)
                ])

    return tasks


def _build_task_args(base_args, task):
    (
        model_key,
        model_path,
        dataset,
        test_file,
        demo_file,
        seed,
        max_length,
        gen_length,
        _dataset_name,
        use_chat_template,
        max_test_samples,
        shots,
        stop_new_line,
    ) = task

    task_args = copy.deepcopy(base_args)
    task_args.model_key = model_key
    task_args.model_name_or_path = model_path
    task_args.model_alias = model_key
    task_args.datasets = dataset
    task_args.test_files = test_file
    task_args.demo_files = demo_file
    task_args.seed = seed
    task_args.input_max_length = max_length
    task_args.generation_max_length = gen_length
    task_args.use_chat_template = use_chat_template
    task_args.max_test_samples = max_test_samples
    task_args.shots = shots
    task_args.stop_new_line = stop_new_line
    task_args.tag = getattr(base_args, "tag", "")
    return task_args


def _task_output_info(task_args, dataset_name):
    model_short_name = task_args.model_alias
    dataset = task_args.datasets
    test_file = task_args.test_files
    tag = task_args.tag

    if dataset == "popqa":
        tag += f"_pop{task_args.popularity_threshold}"

    test_name = os.path.splitext(os.path.basename(test_file))[0]
    base_filename = (
        f"{model_short_name}/{dataset_name}/"
        f"{dataset}_{tag}_{test_name}_in{task_args.input_max_length}_size{task_args.max_test_samples}"
        f"_shots{task_args.shots}_samp{task_args.do_sample}max{task_args.generation_max_length}"
        f"min{task_args.generation_min_length}t{task_args.temperature}p{task_args.top_p}"
        f"_chat{task_args.use_chat_template}.json"
    )
    output_path = os.path.join(task_args.output_dir, f"{base_filename}_seed{task_args.seed}.json")
    return output_path, base_filename


def _load_completed_metrics(output_path, dataset, overwrite):
    if overwrite or not os.path.exists(output_path):
        return None

    try:
        with open(output_path, "r") as f:
            output_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    metrics = output_data.get("averaged_metrics")
    if not metrics or not isinstance(metrics, dict):
        return None

    if "alce" in dataset:
        score_path = output_path + ".score"
        if not os.path.exists(score_path):
            return None
        try:
            with open(score_path, "r") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(loaded, dict):
            return None
        metrics = loaded

    return metrics


def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_batch_execution(task_seed, sample_pairs):
    if task_seed is None or not sample_pairs:
        return
    _set_random_seeds(int(task_seed) + int(sample_pairs[0][0]))


def _task_batch_size(dataset_name):
    if dataset_name.startswith("longqa_"):
        return 1
    if dataset_name.startswith("summ_"):
        return 1
    if dataset_name.startswith("icl_"):
        return 4
    if dataset_name.startswith("cite_"):
        return 2
    return 8


def _write_json_atomic(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp_path, path)


def _update_json_with_lock(path, updater):
    lock_path = f"{path}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        current_payload = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    current_payload = loaded
            except (json.JSONDecodeError, OSError):
                current_payload = {}

        updated_payload = updater(current_payload)
        _write_json_atomic(path, updated_payload)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _run_alce_eval_if_needed(output_path, dataset, overwrite):
    if "alce" not in dataset:
        return None

    score_path = output_path + ".score"
    try:
        if not os.path.exists(score_path) or overwrite:
            import eval_alce
            logger.info(f"Running eval_alce on {output_path} ...")
            cli_args = ["--f", output_path]
            if "nocite" not in dataset:
                cli_args.append("--citations")
            eval_alce.main(cli_args)

        with open(score_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"eval_alce failed for {output_path}: {exc}")
        return None


def _record_metrics(aggregator, filename_map, block_hr_data, output_dir, model_key, dataset, dataset_name, base_filename, metrics):
    key = (model_key, dataset)
    aggregator[key].append(metrics)
    filename_map[key] = base_filename

    hr_val = metrics.get("block_hit_rate", 0)
    if hr_val > 0:
        category = dataset_name.rsplit("_", 1)[0] if dataset_name else "unknown"
        block_hr_data[(model_key, category)].append((dataset, hr_val))

    if len(aggregator[key]) != len(SEEDS):
        return

    metrics_list = aggregator[key]
    stat_output = {}
    all_keys = metrics_list[0].keys()

    for metric_k in all_keys:
        values = [m.get(metric_k, 0) for m in metrics_list]
        try:
            stat_output[metric_k] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "raw_values": values,
            }
        except (TypeError, ValueError):
            stat_output[metric_k] = {"raw_values": values}

    stat_path = os.path.join(output_dir, f"{base_filename}.score.stat")
    _write_json_atomic(stat_path, stat_output)
    logger.info(f"Aggregated stats saved to {stat_path}")


def _prepare_batch_inputs(model, data_bundle, sample_pairs):
    batch_inputs = []
    input_texts = []

    for _, sample in sample_pairs:
        inputs = model.prepare_inputs(sample, data_bundle)
        original_text = None
        if hasattr(inputs, "__contains__") and "input_ids" in inputs:
            original_text = model.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        batch_inputs.append(inputs)
        input_texts.append(original_text)

    return batch_inputs, input_texts


def _generate_batch_outputs(model, sample_pairs, batch_inputs, input_texts, model_key, progress_counter=None, collect_block_hit_rate=False):
    all_outputs = []
    quiet = progress_counter is not None
    collector = None

    if collect_block_hit_rate and isinstance(model, HFModel):
        hf_inner = getattr(model, "model", None)
        if hf_inner is not None and hasattr(hf_inner, "config"):
            collector = _BlockHitRateCollector(hf_inner)
            if collector.available:
                collector.install()
            else:
                collector = None

    start_time = time.time()
    try:
        if (isinstance(model, OpenAIModel) or isinstance(model, AnthropicModel)) and (not isinstance(model, TgiVllmModel)):
            all_outputs = model.generate_batch(batch_inputs)
            _progress_inc(progress_counter, len(batch_inputs))
        elif collector is not None:
            for item in tqdm(batch_inputs, desc="Inference + block hit-rate", leave=False, disable=quiet):
                collector.begin_sample()
                output = model.generate(inputs=item)
                collector.end_sample()
                all_outputs.append(output)
                _progress_inc(progress_counter)
        elif quiet:
            for item in batch_inputs:
                output = model.generate(inputs=item)
                all_outputs.append(output)
                _progress_inc(progress_counter)
        else:
            all_outputs = model.generate_batch(batch_inputs)
            _progress_inc(progress_counter, len(batch_inputs))
    finally:
        block_stats = None
        if collector is not None:
            block_stats = collector.get_stats()
            collector.uninstall()

    elapsed_time = time.time() - start_time
    records = []
    for (sample_idx, _sample), output, input_text in zip(sample_pairs, all_outputs, input_texts):
        records.append({
            "sample_idx": sample_idx,
            "output": output,
            "input_text": input_text,
        })

    return records, block_stats, elapsed_time


def batch_worker_process(gpu_id, task_queue, result_queue, base_args, progress_counter=None):
    import torch

    torch.cuda.set_device(gpu_id)
    device_str = f"cuda:{gpu_id}"
    assert torch.cuda.current_device() == gpu_id, (torch.cuda.current_device(), gpu_id)

    if progress_counter is not None:
        logging.disable(logging.WARNING)
        os.environ["TQDM_DISABLE"] = "1"
        try:
            import datasets as _hf_datasets
            _hf_datasets.disable_progress_bar()
        except (ImportError, AttributeError):
            pass

    current_process = mp.current_process()
    current_process.name = f"BatchWorker-GPU{gpu_id}"
    logger.info(f"Batch worker started on physical GPU {gpu_id}")

    current_model_key = None
    model = None

    while True:
        batch_task = task_queue.get()
        if batch_task is None:
            break

        model_key = batch_task["model_key"]
        model_path = batch_task["model_path"]
        task_args = batch_task["task_args"]

        try:
            if model_key != current_model_key:
                if model is not None:
                    logger.info(f"Unloading previous model {current_model_key}...")
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

                logger.info(f"Loading new model: {model_key} from {model_path}")
                load_args = copy.deepcopy(base_args)
                load_args.model_key = model_key
                load_args.model_name_or_path = model_path
                load_args.input_max_length = task_args.input_max_length
                load_args.generation_max_length = task_args.generation_max_length
                load_args.stop_new_line = task_args.stop_new_line
                load_args.use_chat_template = task_args.use_chat_template
                load_args.shots = task_args.shots
                load_args.seed = task_args.seed
                model = load_LLM(load_args, _device=device_str)
                current_model_key = model_key

            model.max_length = task_args.input_max_length
            model.generation_max_length = task_args.generation_max_length
            model.generation_min_length = task_args.generation_min_length
            model.do_sample = task_args.do_sample
            model.temperature = task_args.temperature
            model.top_p = task_args.top_p
            model.use_chat_template = task_args.use_chat_template
            model.system_message = task_args.system_message
            if getattr(task_args, "thinking", False):
                model.max_length = task_args.input_max_length + 32768
                model.generation_max_length = task_args.generation_max_length + 32768

            sample_pairs = batch_task["sample_pairs"]
            _seed_batch_execution(task_args.seed, sample_pairs)
            data_bundle = batch_task["data_bundle"]
            batch_inputs, input_texts = _prepare_batch_inputs(model, data_bundle, sample_pairs)
            records, block_stats, elapsed_time = _generate_batch_outputs(
                model,
                sample_pairs,
                batch_inputs,
                input_texts,
                model_key=model_key,
                progress_counter=progress_counter,
                collect_block_hit_rate=getattr(base_args, "collect_block_hit_rate", False),
            )

            result_queue.put({
                "status": "batch_success",
                "task_id": batch_task["task_id"],
                "records": records,
                "block_stats": block_stats,
                "elapsed_time": elapsed_time,
            })
        except Exception as exc:
            logger.error(f"Batch task failed for {model_key} on {batch_task['dataset']}: {exc}")
            result_queue.put({
                "status": "batch_error",
                "task_id": batch_task["task_id"],
                "msg": str(exc),
                "traceback": traceback.format_exc(),
            })

    logger.info("Batch worker finished.")


def _ingest_batch_result(task_state, batch_result):
    task_args = task_state["task_args"]
    data_bundle = task_state["data_bundle"]
    metrics = task_state["metrics"]

    task_state["elapsed_time"] += batch_result["elapsed_time"]

    block_stats = batch_result.get("block_stats")
    if block_stats is not None:
        task_state["block_stats_list"].append(block_stats)

    for record in batch_result["records"]:
        sample_idx = record["sample_idx"]
        output = record["output"]
        if output is None:
            continue

        test_item = task_state["samples"][sample_idx]
        input_text = record["input_text"]

        if not task_args.use_chat_template:
            prepend_text = data_bundle["system_template"].format(**test_item)
            output["output"] = prepend_text + output["output"]

        if task_args.thinking:
            matches = re.search(r"(.*</think>)(.*)", output["output"], flags=re.DOTALL)
            if matches:
                output["output"] = matches.group(2).strip()
                output["thoughts"] = matches.group(1).strip()

        mets, others = task_state["post_process"](output, test_item)
        output.update({**others, **mets})

        for key, value in mets.items():
            metrics[key].append(value)
        metrics["input_len"].append(output["input_len"])
        metrics["output_len"].append(output["output_len"])

        result = {**test_item, **output}
        result.pop("context", None)
        result.pop("input_ids", None)
        if input_text is None:
            input_text = result["input_text"]
        task_state["results"][sample_idx] = result


def _finalize_task_output(task_state):
    task_args = task_state["task_args"]
    dataset = task_state["dataset"]
    output_path = task_state["output_path"]

    ordered_results = [task_state["results"][idx] for idx in sorted(task_state["results"])]
    averaged_metrics = {
        key: np.mean(values) * (100 if "_len" not in key else 1)
        for key, values in task_state["metrics"].items()
    }

    merged_block_stats = _merge_block_stats(task_state["block_stats_list"])
    if merged_block_stats and merged_block_stats.get("total_blocks", 0) > 0:
        averaged_metrics["block_hit_rate"] = merged_block_stats["hit_rate"] * 100

    output_data = {
        "args": task_args.__dict__,
        "data": ordered_results,
        "metrics": dict(task_state["metrics"]),
        "averaged_metrics": averaged_metrics,
        "throughput": len(ordered_results) / task_state["elapsed_time"] if task_state["elapsed_time"] > 0 else 0,
    }
    if merged_block_stats:
        output_data["block_hit_rate"] = merged_block_stats

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _write_json_atomic(output_path, output_data)

    if "alce" not in dataset:
        _write_json_atomic(output_path + ".score", averaged_metrics)
    else:
        logger.info(f"Deferring ALCE scoring until after worker teardown: {output_path}")

    return averaged_metrics


def _main_batch_queue(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model_configs = _resolve_model_configs(args)
    if not model_configs:
        return

    input_length = int(args.input_max_length) if args.input_max_length else 16384
    dataset_names = _build_dataset_names(input_length)
    logger.info(f"Input length: {input_length} -> datasets: {dataset_names}")

    tasks = _build_tasks(args, model_configs, dataset_names)
    logger.info(f"Total tasks generated: {len(tasks)}")
    logger.debug("Task list: %s", tasks)

    gpu_list = _detect_gpus(args.gpus)
    num_workers = len(gpu_list)
    logger.info(f"Launching {num_workers} batch worker(s) on GPUs: {gpu_list}")

    queue_capacity = max(num_workers * 4, 1)
    queue_low_watermark = max(num_workers, 1)
    task_queue = mp.Queue(maxsize=queue_capacity)
    result_queue = mp.Queue()
    progress_counter = mp.Value("i", 0)

    processes = []
    for gpu_id in gpu_list:
        process = mp.Process(
            target=batch_worker_process,
            args=(gpu_id, task_queue, result_queue, args, progress_counter),
        )
        process.start()
        processes.append(process)

    aggregator = defaultdict(list)
    filename_map = {}
    block_hr_data = defaultdict(list)
    deferred_alce_tasks = []
    total_tasks = len(tasks)
    completed_tasks = 0
    failed = False
    task_cursor = 0
    pending_batches = 0
    in_flight_tasks = {}
    active_model_key = None

    gpu_label = f"{num_workers} GPU{'s' if num_workers > 1 else ''}"
    pbar = tqdm(total=0, desc=f"Evaluating on {gpu_label}", unit="sample")

    import queue as _queue_mod

    def _update_progress_bar():
        current = progress_counter.value
        delta = current - pbar.n
        if delta > 0:
            pbar.update(delta)
        if current == 0 and completed_tasks == 0:
            pbar.set_postfix_str("loading model & data...")
        else:
            pbar.set_postfix_str(f"tasks: {completed_tasks}/{total_tasks}")

    def _finalize_task(task_state):
        nonlocal completed_tasks, active_model_key

        metrics = _finalize_task_output(task_state)
        if "alce" in task_state["dataset"]:
            deferred_alce_tasks.append({
                "model_key": task_state["model_key"],
                "dataset": task_state["dataset"],
                "dataset_name": task_state["dataset_name"],
                "base_filename": task_state["base_filename"],
                "output_path": task_state["output_path"],
                "overwrite": task_state["task_args"].overwrite,
                "fallback_metrics": metrics,
            })
        else:
            _record_metrics(
                aggregator,
                filename_map,
                block_hr_data,
                args.output_dir,
                task_state["model_key"],
                task_state["dataset"],
                task_state["dataset_name"],
                task_state["base_filename"],
                metrics,
            )

        completed_tasks += 1
        in_flight_tasks.pop(task_state["task_id"], None)
        if not in_flight_tasks:
            active_model_key = None
        pbar.set_postfix_str(f"tasks: {completed_tasks}/{total_tasks}")
        gc.collect()

    def _schedule_tasks(force=False):
        nonlocal task_cursor, completed_tasks, pending_batches, active_model_key

        if not in_flight_tasks and pending_batches == 0:
            active_model_key = None

        while task_cursor < total_tasks:
            if in_flight_tasks and not force and pending_batches > queue_low_watermark:
                break

            task_id = task_cursor
            task = tasks[task_cursor]
            task_cursor += 1

            task_args = _build_task_args(args, task)
            (
                model_key,
                _model_path,
                dataset,
                _test_file,
                _demo_file,
                _seed,
                _max_length,
                _gen_length,
                dataset_name,
                _use_chat_template,
                _max_test_samples,
                _shots,
                _stop_new_line,
            ) = task

            if active_model_key is None:
                active_model_key = model_key
            elif model_key != active_model_key:
                task_cursor -= 1
                break

            output_path, base_filename = _task_output_info(task_args, dataset_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            existing_metrics = _load_completed_metrics(output_path, dataset, task_args.overwrite)
            if existing_metrics is not None:
                logger.info(f"Skipping {output_path} (complete)")
                _record_metrics(
                    aggregator,
                    filename_map,
                    block_hr_data,
                    args.output_dir,
                    model_key,
                    dataset,
                    dataset_name,
                    base_filename,
                    existing_metrics,
                )
                completed_tasks += 1
                pbar.set_postfix_str(f"tasks: {completed_tasks}/{total_tasks}")
                continue

            _set_random_seeds(task_args.seed)
            data = load_data(task_args, dataset, task_args.test_files, task_args.demo_files)
            samples = list(data["data"])

            pbar.total += len(samples)
            pbar.refresh()

            task_state = {
                "task_id": task_id,
                "model_key": model_key,
                "task_args": task_args,
                "dataset": dataset,
                "dataset_name": dataset_name,
                "output_path": output_path,
                "base_filename": base_filename,
                "samples": samples,
                "data_bundle": {
                    "prompt_template": data["prompt_template"],
                    "user_template": data["user_template"],
                    "system_template": data["system_template"],
                },
                "post_process": data["post_process"],
                "results": {},
                "metrics": defaultdict(list),
                "block_stats_list": [],
                "elapsed_time": 0.0,
                "expected_batches": 0,
                "received_batches": 0,
            }
            del data

            batch_size = _task_batch_size(dataset_name)
            sample_pairs = list(enumerate(samples))
            for start in range(0, len(sample_pairs), batch_size):
                batch_pairs = sample_pairs[start:start + batch_size]
                task_queue.put({
                    "task_id": task_id,
                    "model_key": model_key,
                    "model_path": task_args.model_name_or_path,
                    "dataset": dataset,
                    "dataset_name": dataset_name,
                    "task_args": task_args,
                    "data_bundle": task_state["data_bundle"],
                    "sample_pairs": batch_pairs,
                })
                task_state["expected_batches"] += 1
                pending_batches += 1

            if task_state["expected_batches"] == 0:
                empty_metrics = _finalize_task_output(task_state)
                if "alce" in dataset:
                    deferred_alce_tasks.append({
                        "model_key": model_key,
                        "dataset": dataset,
                        "dataset_name": dataset_name,
                        "base_filename": base_filename,
                        "output_path": output_path,
                        "overwrite": task_args.overwrite,
                        "fallback_metrics": empty_metrics,
                    })
                else:
                    _record_metrics(
                        aggregator,
                        filename_map,
                        block_hr_data,
                        args.output_dir,
                        model_key,
                        dataset,
                        dataset_name,
                        base_filename,
                        empty_metrics,
                    )
                completed_tasks += 1
                pbar.set_postfix_str(f"tasks: {completed_tasks}/{total_tasks}")
                gc.collect()
                continue

            in_flight_tasks[task_id] = task_state
            if pending_batches >= queue_capacity:
                break

    try:
        _schedule_tasks(force=True)

        while task_cursor < total_tasks or in_flight_tasks:
            _update_progress_bar()

            if not in_flight_tasks:
                _schedule_tasks(force=True)
                _update_progress_bar()
                if not in_flight_tasks:
                    continue

            try:
                batch_result = result_queue.get(timeout=0.5)
            except _queue_mod.Empty:
                dead_workers = [p for p in processes if (not p.is_alive()) and p.exitcode is not None]
                if dead_workers:
                    for worker in dead_workers:
                        logger.error(
                            f"Worker {worker.name} exited during HELMET evaluation "
                            f"with exit code {worker.exitcode}."
                        )
                    failed = True
                    break
                _schedule_tasks(force=False)
                continue

            _update_progress_bar()

            task_id = batch_result.get("task_id")
            task_state = in_flight_tasks.get(task_id)
            if task_state is None:
                logger.error(f"Received batch result for unknown task {task_id}.")
                failed = True
                break

            pending_batches = max(pending_batches - 1, 0)
            task_state["received_batches"] += 1

            if batch_result["status"] == "batch_success":
                _ingest_batch_result(task_state, batch_result)
            else:
                logger.error(f"Batch worker error: {batch_result['msg']}")
                tb = batch_result.get("traceback")
                if tb:
                    logger.error(tb)
                failed = True
                break

            if task_state["received_batches"] >= task_state["expected_batches"]:
                _finalize_task(task_state)

            _schedule_tasks(force=False)
    finally:
        for _ in range(num_workers):
            try:
                task_queue.put_nowait(None)
            except _queue_mod.Full:
                break

        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Worker {process.name} still alive after join timeout, terminating.")
                process.terminate()

        _update_progress_bar()
        pbar.close()

    if failed:
        sys.exit(1)

    for alce_task in deferred_alce_tasks:
        alce_metrics = _run_alce_eval_if_needed(
            alce_task["output_path"],
            alce_task["dataset"],
            alce_task["overwrite"],
        )
        _record_metrics(
            aggregator,
            filename_map,
            block_hr_data,
            args.output_dir,
            alce_task["model_key"],
            alce_task["dataset"],
            alce_task["dataset_name"],
            alce_task["base_filename"],
            alce_metrics if alce_metrics is not None else alce_task["fallback_metrics"],
        )

    if block_hr_data:
        _print_block_hit_rate_summary(block_hr_data)
        hr_path = os.path.join(args.output_dir, "block_hit_rate_summary.json")
        hr_json = {}
        for (model_key, category), entries in block_hr_data.items():
            hr_json.setdefault(model_key, {})[category] = {
                dataset: hit_rate for dataset, hit_rate in entries
            }

        def _merge_hr_summary(existing_payload):
            merged = dict(existing_payload) if isinstance(existing_payload, dict) else {}
            for model_key, categories in hr_json.items():
                merged_model = dict(merged.get(model_key, {}))
                for category, datasets in categories.items():
                    merged_category = dict(merged_model.get(category, {}))
                    merged_category.update(datasets)
                    merged_model[category] = merged_category
                merged[model_key] = merged_model
            return merged

        _update_json_with_lock(hr_path, _merge_hr_summary)
        logger.info(f"Block hit-rate summary saved to {hr_path}")

    logger.info("All evaluations finished.")


def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    args = parse_arguments()
    _main_batch_queue(args)


if __name__ == "__main__":
    main()
