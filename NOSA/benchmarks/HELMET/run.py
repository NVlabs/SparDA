#!/usr/bin/env python3
# This file is copied/adapted from HELMET (https://github.com/princeton-nlp/HELMET).
# Copyright (c) 2024 Princeton Natural Language Processing.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
HELMET benchmark runner: evaluate and collect results.

Thinking mode is OFF by default (use --thinking to enable for reasoning models).

Data is auto-downloaded on first run if not present.

Usage:
    # Single model (NOSA 8B)
    python run.py --model-path openbmb/NOSA-8B

    # MiniCPM 8B
    python run.py --model-path openbmb/MiniCPM4.1-8B

    # With SparDA indexer
    python run.py --model-path openbmb/NOSA-8B --sparda --indexer-path /path/to/checkpoint.pt

    # Dense baseline
    python run.py --model-path openbmb/MiniCPM4.1-8B --dense

    # Explicit GPU selection
    python run.py --model-path openbmb/NOSA-8B --gpus 0,1,2,3

    # Collect/score existing results only
    python run.py --model-path openbmb/NOSA-8B --collect-only

    # Collect block hit-rate stats per task
    python run.py --model-path openbmb/NOSA-8B --block-hit-rate
"""

import argparse
import fcntl
import hashlib
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_{os.getpid()}")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_URL = "https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz"


DATA_EXPECTED_SUBDIRS = ["ruler", "json_kv", "kilt", "msmarco", "alce"]


def _missing_data_subdirs():
    return [d for d in DATA_EXPECTED_SUBDIRS if not (DATA_DIR / d).is_dir()]


def _download_data_archive(tar_path: Path):
    temp_tar_path = tar_path.with_suffix(tar_path.suffix + ".part")
    print(f"[HELMET] Downloading data from {DATA_URL} ...")
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(
            ["wget", "-c", "--tries=3", "--waitretry=5",
             DATA_URL, "-O", str(temp_tar_path)])
        if result.returncode == 0:
            temp_tar_path.replace(tar_path)
            return
        wait = min(30, 5 * 2 ** (attempt - 1))
        print(f"[HELMET] Download attempt {attempt}/{max_retries} failed "
              f"(rc={result.returncode}), retrying in {wait}s ...")
        time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {DATA_URL} after {max_retries} attempts. "
        f"Try downloading manually:\n"
        f"  wget '{DATA_URL}' -O {tar_path}\n"
        f"or set HF_TOKEN and use:\n"
        f"  wget --header='Authorization: Bearer $HF_TOKEN' '{DATA_URL}' -O {tar_path}"
    )


def ensure_data():
    missing = _missing_data_subdirs()
    if not missing:
        return

    if DATA_DIR.exists():
        print(f"[HELMET] Data directory incomplete, missing: {missing}")
    else:
        print("[HELMET] Data directory not found.")

    tar_path = SCRIPT_DIR / "data.tar.gz"
    lock_path = SCRIPT_DIR / ".data.lock"
    # Multiple benchmark jobs share one HELMET data directory; serialize setup.
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        missing = _missing_data_subdirs()
        if not missing:
            print("[HELMET] Data ready.")
            return

        for archive_attempt in range(2):
            if tar_path.exists() and tar_path.stat().st_size > 1_000_000:
                print(f"[HELMET] Found existing {tar_path.name} "
                      f"({tar_path.stat().st_size / 1e9:.1f} GB), extracting...")
            else:
                _download_data_archive(tar_path)

            print("[HELMET] Extracting data.tar.gz ...")
            try:
                with tarfile.open(tar_path) as tf:
                    tf.extractall(SCRIPT_DIR)
            except (tarfile.TarError, EOFError, OSError) as exc:
                if tar_path.exists():
                    tar_path.unlink()
                if archive_attempt == 0:
                    print(f"[HELMET] Archive extraction failed ({exc}); re-downloading once ...")
                    continue
                raise RuntimeError(f"Failed to extract HELMET archive: {tar_path}") from exc
            break

        try:
            tar_path.unlink()
        except FileNotFoundError:
            pass

    still_missing = _missing_data_subdirs()
    if still_missing:
        print(f"[HELMET] WARNING: still missing after extraction: {still_missing}")
    else:
        print("[HELMET] Data ready.")


NATIVE_MAX = {"nosa": 32768, "minicpm": 65536}

_GPT4_LONGQA_PREFIXES = ("narrativeqa_", "infbench_qa_", "infbench_choice_")
_GPT4_SUMM_PREFIXES = ("multi_lexsum_", "infbench_sum_")


def _model_lock_path(output_dir: str, model_key: str) -> Path:
    lock_root = Path(output_dir)
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha1(model_key.encode("utf-8")).hexdigest() + ".lock"
    return lock_root / f".helmet_run_{lock_name}"


def _gpt4eval_output_name(path_str):
    """Replace only the final .json with -gpt4eval_o.json."""
    if path_str.endswith(".json"):
        return path_str[:-5] + "-gpt4eval_o.json"
    return path_str + "-gpt4eval_o.json"


def _find_gpt4_eval_files(output_dir, model_key):
    """Find result JSON files that need model-based evaluation."""
    model_dir = Path(output_dir) / model_key
    if not model_dir.is_dir():
        return [], []

    longqa_files, summ_files = [], []
    for json_file in sorted(model_dir.rglob("*.json")):
        name = json_file.name
        if any(s in name for s in (".score", ".stat", "-gpt4eval", ".batch")):
            continue
        gpt4_out = Path(_gpt4eval_output_name(str(json_file)))
        if gpt4_out.exists():
            continue
        if any(name.startswith(p) for p in _GPT4_LONGQA_PREFIXES):
            longqa_files.append(json_file)
        elif any(name.startswith(p) for p in _GPT4_SUMM_PREFIXES):
            summ_files.append(json_file)
    return longqa_files, summ_files


def _judge_short_name(model_name):
    """Extract a short judge name for metric keys, e.g. 'azure/openai/gpt-5.2' -> 'gpt-5.2'."""
    return model_name.rsplit("/", 1)[-1]


def _run_gpt4_eval(output_dir, model_key):
    """Run LLM judge for summarization and long QA tasks.

    Skips gracefully (with a warning) when OPENAI_API_KEY is not set or
    required packages are missing.  All other task results remain valid.
    """
    longqa_files, summ_files = _find_gpt4_eval_files(output_dir, model_key)
    total = len(longqa_files) + len(summ_files)
    if total == 0:
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            f"\n[HELMET] WARNING: OPENAI_API_KEY not set -- skipping "
            f"model-based evaluation for {total} output file(s)."
        )
        print("[HELMET]   Affected metrics: <judge>-score (longqa), <judge>-f1 (summ)")
        print("[HELMET]   Set OPENAI_API_KEY to enable. All other metrics are valid.")
        return

    print(f"\n[HELMET] Running LLM judge evaluation ({total} file(s))...")

    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

    saved_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    try:
        from model_utils import OpenAIModel
    except ImportError as e:
        print(
            f"[HELMET] WARNING: Cannot import OpenAIModel ({e}). "
            f"Skipping judge evaluation. Install the openai package to enable."
        )
        os.chdir(saved_cwd)
        return

    _JUDGE_CANDIDATES = ["azure/openai/gpt-5.2", "gpt-5.2", "gpt-4o-2024-05-13"]
    oai = None
    judge_name = None
    last_err = None
    for candidate in _JUDGE_CANDIDATES:
        try:
            oai = OpenAIModel(candidate, temperature=0.1, generation_max_length=4096)
            test_resp = oai.generate(prompt="Say OK")
            if test_resp and test_resp.get("output"):
                judge_name = _judge_short_name(candidate)
                print(f"[HELMET] Using judge model: {candidate} (metric prefix: {judge_name})")
                break
            print(f"[HELMET] Judge model '{candidate}': test call returned empty response")
            oai = None
        except Exception as e:
            last_err = e
            print(f"[HELMET] Judge model '{candidate}' failed: {e}")
            oai = None
            # Auth errors affect all candidates — stop early
            err_msg = str(e).lower()
            if "401" in err_msg or "invalid_api_key" in err_msg or "incorrect api key" in err_msg:
                print("[HELMET] ERROR: OPENAI_API_KEY is invalid. "
                      "Get a valid key from https://platform.openai.com/api-keys")
                break
            continue
    if oai is None:
        print(
            "[HELMET] WARNING: Could not initialise any judge model. "
            "Skipping judge evaluation."
        )
        if last_err:
            print(f"[HELMET]   Last error: {last_err}")
        os.chdir(saved_cwd)
        return

    if longqa_files:
        try:
            from eval_gpt4_longqa import check_metrics as _ck_longqa

            for f in longqa_files:
                out = _gpt4eval_output_name(str(f))
                print(f"  [{judge_name} longqa] {f.relative_to(Path(output_dir))}")
                try:
                    _ck_longqa(oai, str(f), out, judge_name=judge_name)
                except Exception as exc:
                    print(f"  [{judge_name} longqa] ERROR scoring {f.name}: {exc}")
        except ImportError as e:
            print(f"[HELMET] WARNING: Cannot import eval_gpt4_longqa: {e}")

    if summ_files:
        try:
            from eval_gpt4_summ import check_metrics as _ck_summ

            for f in summ_files:
                out = _gpt4eval_output_name(str(f))
                print(f"  [{judge_name} summ] {f.relative_to(Path(output_dir))}")
                try:
                    _ck_summ(oai, str(f), out, judge_name=judge_name)
                except Exception as exc:
                    print(f"  [{judge_name} summ] ERROR scoring {f.name}: {exc}")
        except ImportError as e:
            print(f"[HELMET] WARNING: Cannot import eval_gpt4_summ: {e}")

    os.chdir(saved_cwd)
    print(f"[HELMET] Judge evaluation complete (model: {judge_name}).")


# ─── Authoritative per-dataset metrics (from collect_results.py) ───────────
# "gpt-score" / "gpt-f1" are wildcards matching any judge prefix.
_DATASET_METRICS = {
    "json_kv":            ["substring_exact_match"],
    "ruler_niah_mk_2":    ["ruler_recall"],
    "ruler_niah_mk_3":    ["ruler_recall"],
    "ruler_niah_mv":      ["ruler_recall"],
    "kilt_nq":            ["substring_exact_match"],
    "kilt_hotpotqa":      ["substring_exact_match"],
    "kilt_popqa":         ["substring_exact_match"],
    "kilt_triviaqa":      ["substring_exact_match"],
    "icl_trec_coarse":    ["exact_match"],
    "icl_trec_fine":      ["exact_match"],
    "icl_banking77":      ["exact_match"],
    "icl_clinic150":      ["exact_match"],
    "icl_nlu":            ["exact_match"],
    "alce_asqa":          ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari":       ["qampari_rec_top5", "citation_rec", "citation_prec"],
    "msmarco_rerank_psg": ["NDCG@10"],
    "narrativeqa":        ["gpt-score"],
    "infbench_qa":        ["rougeL_f1"],
    "infbench_choice":    ["exact_match"],
    "infbench_sum":       ["gpt-f1"],
    "multi_lexsum":       ["gpt-f1"],
}
_CATEGORY_ORDER = ["Recall", "RAG", "ICL", "Cite", "Rerank", "LongQA", "Summ"]
_DIR_TO_CATEGORY = {
    "recall": "Recall", "rag": "RAG", "icl": "ICL", "cite": "Cite",
    "rerank": "Rerank", "longqa": "LongQA", "summ": "Summ",
}
# Fallback for datasets not in _DATASET_METRICS
_METRIC_PRIORITY = [
    "ruler_recall", "substring_exact_match", "exact_match",
    "NDCG@10", "rougeL_f1", "rougeL_recall",
    "gpt-5.2-score", "gpt-5.2-f1",
    "gpt-4-score", "gpt-4-f1", "str_em",
    "qampari_rec_top5", "citation_rec", "citation_prec",
]
_SKIP_METRICS = {"input_len", "output_len", "throughput", "num_preds"}


def _match_dataset_prefix(ds):
    """Find the longest matching prefix in _DATASET_METRICS for *ds*."""
    best = None
    for prefix in _DATASET_METRICS:
        if ds == prefix or ds.startswith(prefix + "_"):
            if best is None or len(prefix) > len(best):
                best = prefix
    return best


def _resolve_gpt_wildcard(metrics, wildcard):
    """Resolve ``'gpt-score'`` or ``'gpt-f1'`` to the actual key present in *metrics*."""
    suffix = wildcard.split("gpt-", 1)[1]  # "score" or "f1"
    for key in metrics:
        if "gpt-" in key and key.endswith("-" + suffix):
            return key, metrics[key]
    return None, None


def _scale_gpt_metric(key, value):
    """Scale GPT judge metrics: ``*-score`` x100/3, ``*-f1`` x100."""
    if value is None or key is None:
        return value
    if "gpt-" in key:
        if key.endswith("-score"):
            return value * 100.0 / 3
        if key.endswith("-f1"):
            return value * 100.0
    return value


def _print_results_summary(output_dir, model_key):
    """Print a summary table with per-dataset metrics matching collect_results.py."""
    import json as _json
    from collections import OrderedDict

    model_dir = Path(output_dir) / model_key
    if not model_dir.is_dir():
        return

    def _dataset_from_filename(name):
        for suffix in (".json.score.stat", ".json.score", ".json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        if "_seed" in name:
            name = name[: name.rindex("_seed")]
        return name.split("_eval_")[0] if "_eval_" in name else name

    # ── Step 1: collect all available metrics per (cat_dir, dataset) ──
    available = {}  # (cat_dir, ds) -> {metric: value}

    stat_files = sorted(model_dir.rglob("*.score.stat"))
    score_files = sorted(model_dir.rglob("*.score"))
    stat_bases = {str(f).rsplit(".stat", 1)[0] for f in stat_files}
    score_files = [
        f for f in score_files
        if str(f) not in stat_bases and not str(f).endswith(".stat")
    ]

    for f in stat_files:
        cat_dir = f.parent.name
        ds = _dataset_from_filename(f.name)
        try:
            data = _json.loads(f.read_text())
        except Exception:
            continue
        metrics = {
            k: v["mean"] for k, v in data.items()
            if isinstance(v, dict) and "mean" in v
        }
        available[(cat_dir, ds)] = metrics

    for f in score_files:
        cat_dir = f.parent.name
        ds = _dataset_from_filename(f.name)
        if (cat_dir, ds) in available:
            continue
        try:
            data = _json.loads(f.read_text())
        except Exception:
            continue
        metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        available[(cat_dir, ds)] = metrics

    # Fallback: load averaged_metrics from main result JSONs when no .score files
    for f in sorted(model_dir.rglob("*.json")):
        if any(s in f.name for s in (".score", ".stat", "-gpt4eval", ".batch")):
            continue
        cat_dir = f.parent.name
        ds = _dataset_from_filename(f.name)
        if (cat_dir, ds) in available:
            continue
        try:
            data = _json.loads(f.read_text())
        except Exception:
            continue
        am = data.get("averaged_metrics", {})
        metrics = {k: v for k, v in am.items() if isinstance(v, (int, float))}
        if metrics:
            available[(cat_dir, ds)] = metrics

    # Load GPT judge metrics from -gpt4eval_o.json files
    for f in sorted(model_dir.rglob("*-gpt4eval_o.json")):
        cat_dir = f.parent.name
        ds = _dataset_from_filename(
            f.name.replace("-gpt4eval_o.json", ".json")
        )
        try:
            gpt_am = _json.loads(f.read_text()).get("averaged_metrics", {})
        except Exception:
            continue
        key = (cat_dir, ds)
        if key not in available:
            available[key] = {}
        for k, v in gpt_am.items():
            if isinstance(v, (int, float)) and "gpt-" in k:
                available[key][k] = v

    if not available:
        return

    # Block hit-rate from summary file
    hr_summary_path = Path(output_dir) / "block_hit_rate_summary.json"
    hr_lookup = {}
    if hr_summary_path.exists():
        try:
            hr_data = _json.loads(hr_summary_path.read_text())
            for mk, cats in hr_data.items():
                if mk == model_key:
                    for _cat, datasets in cats.items():
                        for ds, hr_val in datasets.items():
                            hr_lookup[ds] = hr_val
        except Exception:
            pass

    # ── Step 2: extract correct metrics per dataset ──
    # result_data: {(category, ds) -> {"rows": [(metric, value)], "hr": float|None}}
    result_data = OrderedDict()

    for (cat_dir, ds), metrics in sorted(available.items()):
        cat_prefix = cat_dir.rsplit("_", 1)[0] if "_" in cat_dir else cat_dir
        category = _DIR_TO_CATEGORY.get(cat_prefix)
        if category is None:
            continue

        hr = metrics.pop("block_hit_rate", None)
        if hr is None:
            hr = hr_lookup.get(ds)

        ds_prefix = _match_dataset_prefix(ds)
        entry_rows = []

        if ds_prefix is not None:
            for metric_spec in _DATASET_METRICS[ds_prefix]:
                if metric_spec.startswith("gpt-"):
                    actual_key, value = _resolve_gpt_wildcard(metrics, metric_spec)
                    if actual_key is not None:
                        entry_rows.append((actual_key, _scale_gpt_metric(actual_key, value)))
                else:
                    if metric_spec in metrics:
                        entry_rows.append((metric_spec, metrics[metric_spec]))

        if not entry_rows:
            # Fallback: pick first available from priority list
            for m in _METRIC_PRIORITY:
                if m in metrics:
                    entry_rows.append((m, _scale_gpt_metric(m, metrics[m])))
                    break
            if not entry_rows:
                for m in metrics:
                    if m not in _SKIP_METRICS:
                        entry_rows.append((m, metrics[m]))
                        break

        if entry_rows:
            result_data[(category, ds)] = {"rows": entry_rows, "hr": hr}

    if not result_data:
        return

    # ── Step 3: display ──
    has_hr = any(d["hr"] is not None for d in result_data.values())
    w = 90 if has_hr else 80

    print(f"\n{'=' * w}")
    print(f"  HELMET Results: {model_key}")
    print(f"{'=' * w}")

    by_cat = OrderedDict()
    for (cat, ds), data in result_data.items():
        by_cat.setdefault(cat, []).append((ds, data))

    cat_avgs = OrderedDict()
    for cat in _CATEGORY_ORDER:
        if cat not in by_cat:
            continue
        entries = by_cat[cat]
        print(f"\n  [{cat}]")
        all_values = []
        for ds, data in entries:
            label = ds
            for m_name, m_val in data["rows"]:
                line = f"    {label:38s} {m_name:24s} {m_val:6.1f}"
                if has_hr and data["hr"] is not None:
                    line += f"   hit%={data['hr']:.1f}"
                print(line)
                all_values.append(m_val)
                label = ""  # blank for subsequent metrics of same dataset
        if all_values:
            cat_avgs[cat] = sum(all_values) / len(all_values)

    if cat_avgs:
        print(f"\n  [Averages]")
        for cat, avg in cat_avgs.items():
            print(f"    {cat:38s} {'':24s} {avg:6.1f}")
        overall = sum(cat_avgs.values()) / len(cat_avgs)
        print(f"    {'OVERALL':38s} {'':24s} {overall:6.1f}")

    print(f"{'=' * w}")


def detect_method(model_key: str, model_path: str) -> str:
    combined = (model_key + model_path).lower()
    if "nosa" in combined:
        return "nosa"
    if "minicpm" in combined or "infllmv2" in combined:
        return "minicpm"
    return "other"


def main():
    parser = argparse.ArgumentParser(
        description="HELMET: evaluate and collect results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path", default=None,
        help="HuggingFace model path (e.g. openbmb/NOSA-8B).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model key (e.g. 8b_nosa_sft). Auto-derived from --model-path if omitted.",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Force dense (full) attention.",
    )
    parser.add_argument(
        "--sparda", action="store_true",
        help="Enable SparDA indexer for NOSA models. Requires --indexer-path.",
    )
    parser.add_argument(
        "--infinigen", action="store_true",
        help="Enable InfiniGen decode (InfLLMv2 sparse prefill + InfiniGen masked decode). "
             "Only for MiniCPM/NOSA models. Overrides --sparda.",
    )
    parser.add_argument(
        "--indexer-path", type=str, default=None,
        help="Path to trained indexer checkpoint for SparDA mode. "
             "Required when --sparda is set.",
    )
    parser.add_argument(
        "--long-context", action="store_true",
        help="Enable long-context extension. "
             "For NOSA: increases rope_theta to 40000 (default). "
             "For MiniCPM: applies LongRoPE. "
             "Auto-enabled when --input-max-length exceeds the model's native max.",
    )
    parser.add_argument(
        "--yarn", action="store_true",
        help="Use YaRN (factor=4.0) instead of rope_theta increase for NOSA long-context. "
             "Only effective with --long-context on NOSA models.",
    )
    parser.add_argument(
        "--gpus", default="auto",
        help="Comma-separated GPU IDs (default: auto). "
             "'auto' detects via CUDA_VISIBLE_DEVICES or nvidia-smi.",
    )
    parser.add_argument(
        "--input-max-length", type=int, default=None,
        help="Override input_max_length for eval.py.",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory for results (default: output).",
    )
    parser.add_argument(
        "--collect-only", "--eval-only", dest="collect_only", action="store_true",
        help="Skip model inference; only collect/score existing results.",
    )
    parser.add_argument(
        "--skip-gpt4-eval", action="store_true",
        help="Skip GPT-4 model-based evaluation even if OPENAI_API_KEY is set.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run: keep existing predictions and only "
             "run missing tasks. Without this flag, all old results are cleared.",
    )
    parser.add_argument(
        "--block-hit-rate", action="store_true",
        help="Collect and print block hit-rate stats per task "
             "(NOSA/MiniCPM sparse models only).",
    )
    parser.add_argument(
        "--samples", type=int, default=None, metavar="N",
        help="Max samples per task (default: use HELMET config, typically 100).",
    )
    args = parser.parse_args()

    if args.model_path is None and not args.collect_only:
        parser.error("Provide --model-path (or use --collect-only).")

    if args.dense and args.sparda:
        parser.error("--dense and --sparda are mutually exclusive.")
    if args.sparda and args.indexer_path is None:
        parser.error("--indexer-path is required when --sparda is set.")
    if args.indexer_path and not os.path.exists(args.indexer_path):
        parser.error(f"--indexer-path does not exist: {args.indexer_path}")

    # Derive model key
    if args.model_path:
        method = detect_method(args.model or "", args.model_path)
        if args.model:
            model_key = args.model
        else:
            base = Path(args.model_path).name.lower().replace(".", "-")
            if method == "nosa":
                if args.dense:
                    mode = "nosa_dense"
                else:
                    mode = "nosa_sparda" if args.sparda else "nosa"
            elif args.dense:
                mode = "dense"
            elif args.sparda:
                mode = "minicpm_sparda"
            elif method == "minicpm":
                mode = "minicpm"
            else:
                mode = "dense"
            model_key = f"{base}_{mode}" if not base.endswith(mode) else base
        model_path = args.model_path
    else:
        model_key = args.model
        model_path = None

    # Resolve input_max_length from model native max if not specified
    if args.input_max_length is None and args.model_path:
        native_max = NATIVE_MAX.get(method, 0)
        args.input_max_length = native_max or 16384

    # Auto-enable long-context when input_max_length exceeds native max
    if not args.long_context and args.model_path:
        native_max = NATIVE_MAX.get(method, 0)
        if native_max and args.input_max_length > native_max:
            args.long_context = True
            print(f"[Auto] --long-context enabled: input_max_length "
                  f"{args.input_max_length} > native max {native_max}")

    length_tag = f"{args.input_max_length // 1024}k"
    model_key = f"{model_key}_{length_tag}"

    print(f"[HELMET] model_key={model_key}, path={model_path}")
    print(f"[HELMET] input_max_length={args.input_max_length}, "
          f"gpus={args.gpus}, sparda={args.sparda}")

    ensure_data()

    if model_key:
        lock_path = _model_lock_path(args.output_dir, model_key)
        with lock_path.open("w") as lock_file:
            print(f"[HELMET] Acquiring run lock: {lock_path}")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            print(f"[HELMET] Run lock acquired for {model_key}")

            if not args.collect_only:
                model_out = Path(args.output_dir) / model_key
                if model_out.is_dir():
                    if args.resume:
                        # --resume: keep prediction files, clear only derivatives
                        stale = (
                            list(model_out.rglob("*.json.score"))
                            + list(model_out.rglob("*.json.score.stat"))
                            + list(model_out.rglob("*.json.batch"))
                            + list(model_out.rglob("*-gpt4eval_o.json"))
                        )
                    else:
                        # Default: clear everything for a fresh run
                        stale = (
                            list(model_out.rglob("*.json"))
                            + list(model_out.rglob("*.json.score"))
                            + list(model_out.rglob("*.json.score.stat"))
                            + list(model_out.rglob("*.json.batch"))
                            + list(model_out.rglob("*-gpt4eval_o.json"))
                        )
                    if not args.resume:
                        hr_summary = Path(args.output_dir) / "block_hit_rate_summary.json"
                        if hr_summary.exists():
                            stale.append(hr_summary)
                    if stale:
                        for f in stale:
                            f.unlink(missing_ok=True)
                        mode = "derivatives only" if args.resume else "all results"
                        print(f"[HELMET] Cleared {len(stale)} file(s) ({mode}) "
                              f"in {model_out}/")

                print(f"\n[HELMET] Evaluating (output_dir={args.output_dir})")
                eval_cmd = [
                    sys.executable, str(SCRIPT_DIR / "eval.py"),
                    "--output_dir", args.output_dir,
                    "--model_name_or_path", model_path,
                    "--model_key", model_key,
                    "--gpus", args.gpus,
                    "--input_max_length", str(args.input_max_length),
                ]
                if args.sparda:
                    eval_cmd.append("--sparda")
                if getattr(args, 'infinigen', False):
                    eval_cmd.append("--enable_infinigen")
                if args.indexer_path:
                    eval_cmd.extend(["--indexer_path", args.indexer_path])
                if args.long_context:
                    eval_cmd.append("--long_context")
                if args.yarn:
                    eval_cmd.append("--yarn")
                if args.block_hit_rate:
                    eval_cmd.append("--collect_block_hit_rate")
                if args.samples is not None:
                    eval_cmd.extend(["--max_test_samples", str(args.samples)])
                eval_env = os.environ.copy()
                eval_env.setdefault("HELMET_QUIET", "1")
                subprocess.run(eval_cmd, cwd=str(SCRIPT_DIR), check=True, env=eval_env)

            if not args.skip_gpt4_eval:
                _run_gpt4_eval(args.output_dir, model_key)
            else:
                print("\n[HELMET] Skipping GPT-4 evaluation (--skip-gpt4-eval).")

            _print_results_summary(args.output_dir, model_key)
    elif args.skip_gpt4_eval:
        print("\n[HELMET] Skipping GPT-4 evaluation (--skip-gpt4-eval).")

    print("\n[HELMET] Done.")


if __name__ == "__main__":
    main()
