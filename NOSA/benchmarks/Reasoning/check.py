# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reasoning benchmark evaluation.

- AIME:     exact integer match (standard, no LLM judge needed)
- MATH-500: exact match via \\boxed{} extraction + normalization (PRM800K standard)
- Optional LLM judge: GPT-5.2 answer-presence check, for parity with the
  original THUNLP/NOSA reasoning benchmark style.

Usage:
    # CLI (called by run.py automatically):
    python check.py --model 8b_nosa --dataset math-500
    python check.py --model 8b_minicpm --dataset aime24 --posfix _my_run

    # Standalone (edit MODELS_TO_CHECK / DATASETS below, then run):
    python check.py
"""
import argparse
import glob
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from lr_utils import (
    extract_last_boxed,
    parse_aime_prediction,
    parse_math_prediction,
    normalize_math_answer,
    strip_thinking_blocks,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JUDGE_MODEL = os.environ.get("REASONING_JUDGE_MODEL", "gpt-5.2")
DEFAULT_JUDGE_CANDIDATES = ["azure/openai/gpt-5.2", "gpt-5.2", "gpt-4o-2024-05-13"]

JUDGE_PROMPT = """
    You are a judge. Determine whether the generation explicitly contains the answer.
    Only explicit mention counts — implicit reasoning, hints, or derivations do not count.
    The wording does not need to match exactly; judge based on semantic equivalence.
    Ignore correctness of reasoning; only check if the answer is stated anywhere in the generation.
    If the answer appears, even if surrounded by extra text, output "yes".
    Otherwise output "no".
    Output only "yes" or "no".

    Generation: {generation}
    Answer: {answer}
    """


# ── Deterministic exact-match evaluation (AIME + MATH-500) ───────────────

def _find_result_files(model, dataset, posfix="", output_dir=None):
    """Find result JSONL files across gen_len/temp/rank combos."""
    if output_dir:
        base = os.path.join(output_dir, dataset, model)
    else:
        base = f"./results{posfix}/{dataset}/{model}"
    patterns = [
        f"{base}/{dataset}-{model}-*-rank_*.jsonl",
        f"{base}/{dataset}-{model}-*-*-rank_*.jsonl",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        for gen_len in ["65536", "8192"]:
            for temp in ["0.9", "1.0"]:
                for rank in range(8):
                    p = f"{base}/{dataset}-{model}-{gen_len}-{temp}-rank_{rank}.jsonl"
                    if os.path.exists(p):
                        files.append(p)
    return sorted(set(files))


def _expected_dataset_total(dataset: str) -> int:
    dataset_path = SCRIPT_DIR / f"{dataset}.jsonl"
    if not dataset_path.exists():
        return 0
    with dataset_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _prediction_completion(model, dataset, posfix="", output_dir=None):
    files = _find_result_files(model, dataset, posfix, output_dir=output_dir)
    expected_total = _expected_dataset_total(dataset)
    total_rows = 0
    sample_ids = set()

    for path in files:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
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
                    sample_ids.add(sample_idx)

    completed = len(sample_ids) if sample_ids else total_rows
    is_complete = bool(files) and completed == expected_total and total_rows == expected_total
    return {
        "files": files,
        "completed": completed,
        "expected_total": expected_total,
        "total_rows": total_rows,
        "has_sample_ids": bool(sample_ids),
        "is_complete": is_complete,
    }


def _load_results(model, dataset, posfix="", output_dir=None):
    files = _find_result_files(model, dataset, posfix, output_dir=output_dir)
    data = []
    for f in files:
        with open(f, 'r') as fp:
            for line in fp:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def _get_result_root(model, dataset, posfix="", output_dir=None):
    if output_dir:
        return os.path.join(output_dir, dataset, model)
    return f"./results{posfix}/{dataset}/{model}"


def _get_summary_root(model, posfix="", output_dir=None):
    if output_dir:
        return os.path.join(output_dir, "_summary", model)
    return f"./results{posfix}/_summary/{model}"


def _append_result_log(result_root: str, line: str):
    os.makedirs(result_root, exist_ok=True)
    with open(os.path.join(result_root, "result.log"), "a") as f:
        print(line, file=f)


@lru_cache(maxsize=None)
def _dataset_answers(dataset: str):
    dataset_path = SCRIPT_DIR / f"{dataset}.jsonl"
    if not dataset_path.exists():
        return {}

    answers = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            problem = str(entry.get("problem") or entry.get("question") or "")
            if problem:
                answers[problem] = _entry_answer(entry)
    return answers


def _entry_answer(entry, dataset_lookup=None) -> str:
    answer = entry.get("answer") or entry.get("final_answer")
    if answer:
        return str(answer)

    solution = entry.get("solution") or entry.get("rationale")
    if solution:
        return str(solution)

    if dataset_lookup is not None:
        problem = str(entry.get("problem") or entry.get("question") or "")
        if problem:
            return dataset_lookup.get(problem, "")

    return ""


def check_exact_match(model, dataset, posfix="", output_dir=None):
    """Evaluate using deterministic exact match (AIME: integer, MATH-500: boxed)."""
    data = _load_results(model, dataset, posfix, output_dir=output_dir)
    if not data:
        print(f"  No result files found for {model}/{dataset}")
        return

    is_aime = dataset.startswith("aime")
    dataset_lookup = _dataset_answers(dataset)
    correct = 0
    total = len(data)

    for entry in data:
        raw_generation = entry.get('generation', '')
        stripped_generation = strip_thinking_blocks(raw_generation)
        ground_truth = normalize_math_answer(_entry_answer(entry, dataset_lookup))

        # Standard approach (per Math-Verify, Qwen2.5-Math, open-r1):
        # extract \boxed{} from full text including think blocks, since
        # reasoning models often put the correct answer in \boxed{} inside
        # <think> and may hallucinate after </think>.
        # Priority: boxed in stripped text > boxed in full text > other methods on stripped text.
        boxed = extract_last_boxed(stripped_generation)
        if boxed is None:
            boxed = extract_last_boxed(raw_generation)
        if boxed is not None:
            pred = normalize_math_answer(boxed)
        elif is_aime:
            pred = parse_aime_prediction(stripped_generation)
        else:
            pred = parse_math_prediction(stripped_generation)

        if normalize_math_answer(pred) == ground_truth:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"  {dataset}/{model}: {correct}/{total} = {accuracy:.1%}")

    result_root = _get_result_root(model, dataset, posfix, output_dir=output_dir)
    method = "integer match" if is_aime else "boxed exact match"
    _append_result_log(result_root, f"{method} accuracy: {accuracy} ({correct}/{total})")

    return {"correct": correct, "total": total, "accuracy": accuracy}


def _judge_env_available(judge_model: str) -> bool:
    if judge_model in {"", "auto", "gpt-5.2"}:
        return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"))
    if judge_model.startswith("azure/"):
        return bool(os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    return bool(os.environ.get("OPENAI_API_KEY"))


def _load_openai_judge_model():
    helmet_dir = SCRIPT_DIR.parent / "HELMET"
    if str(helmet_dir) not in sys.path:
        sys.path.insert(0, str(helmet_dir))
    from model_utils import OpenAIModel

    return OpenAIModel


def _judge_short_name(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _judge_candidates(requested_model: str):
    if requested_model in {"", "auto", "gpt-5.2"}:
        return list(DEFAULT_JUDGE_CANDIDATES)
    return [requested_model]


def _init_judge_model(requested_model: str):
    OpenAIModel = _load_openai_judge_model()
    last_err = None
    for candidate in _judge_candidates(requested_model):
        try:
            judge = OpenAIModel(
                candidate,
                temperature=0.0,
                top_p=1.0,
                generation_max_length=16,
                do_sample=False,
                stop_new_line=True,
                use_chat_template=True,
                system_message=None,
            )
            test_resp = judge.generate(prompt="Say OK")
            if test_resp and test_resp.get("output"):
                judge_name = _judge_short_name(candidate)
                print(
                    f"  using judge model: {candidate} "
                    f"(metric label: {judge_name})"
                )
                return judge, judge_name, candidate, None
            last_err = RuntimeError("test call returned empty response")
            print(f"  judge model '{candidate}' returned an empty test response")
        except Exception as exc:
            last_err = exc
            print(f"  judge model '{candidate}' failed: {exc}")
    return None, _judge_short_name(requested_model), None, last_err


def _parse_judge_vote(text: Optional[str]) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    return "yes" in normalized


def check_llm_judge(model, dataset, posfix="", output_dir=None, judge_model=DEFAULT_JUDGE_MODEL):
    """Evaluate using an LLM judge for answer presence, matching the original NOSA style."""
    data = _load_results(model, dataset, posfix, output_dir=output_dir)
    result_root = _get_result_root(model, dataset, posfix, output_dir=output_dir)
    judge_label = _judge_short_name(judge_model) if judge_model else DEFAULT_JUDGE_MODEL

    if not data:
        print(f"  No result files found for {model}/{dataset}")
        _append_result_log(
            result_root,
            f"judge accuracy [{judge_label}]: skipped (no prediction files found)",
        )
        return {
            "judge_model": judge_label,
            "correct": None,
            "total": 0,
            "accuracy": None,
            "status": "skipped",
            "notes": "no prediction files found",
        }

    if not _judge_env_available(judge_model):
        note = (
            "missing OPENAI_API_KEY / AZURE_OPENAI_API_KEY"
            if judge_model in {"", "auto", "gpt-5.2"} or judge_model.startswith("azure/")
            else "missing OPENAI_API_KEY"
        )
        print(f"  Judge skipped for {dataset}/{model}: {note}")
        _append_result_log(result_root, f"judge accuracy [{judge_label}]: skipped ({note})")
        return {
            "judge_model": judge_label,
            "correct": None,
            "total": len(data),
            "accuracy": None,
            "status": "skipped",
            "notes": note,
        }

    try:
        judge, judge_label, judge_backend, init_err = _init_judge_model(judge_model)
    except ImportError as exc:
        init_err = exc
        judge = None
        judge_backend = None
    if judge is None:
        note = f"could not initialize judge model ({init_err})"
        print(f"  Judge skipped for {dataset}/{model}: {note}")
        _append_result_log(result_root, f"judge accuracy [{judge_label}]: skipped ({note})")
        return {
            "judge_model": judge_label,
            "correct": None,
            "total": len(data),
            "accuracy": None,
            "status": "skipped",
            "notes": note,
        }
    if judge_backend and judge_backend != judge_label:
        _append_result_log(result_root, f"judge backend [{judge_label}]: {judge_backend}")

    dataset_lookup = _dataset_answers(dataset)
    prompts = [
        JUDGE_PROMPT.format(
            generation=entry.get("generation", ""),
            answer=_entry_answer(entry, dataset_lookup),
        )
        for entry in data
    ]
    try:
        outputs = judge.generate_batch(prompt=prompts)
    except Exception as exc:
        note = f"judge request failed ({exc})"
        print(f"  Judge skipped for {dataset}/{model}: {note}")
        _append_result_log(result_root, f"judge accuracy [{judge_label}]: skipped ({note})")
        return {
            "judge_model": judge_label,
            "correct": None,
            "total": len(data),
            "accuracy": None,
            "status": "skipped",
            "notes": note,
        }

    correct = 0
    unparsable = 0
    for output in outputs:
        if output is None or "output" not in output:
            unparsable += 1
            continue
        if _parse_judge_vote(output["output"]):
            correct += 1
        elif not str(output["output"]).strip():
            unparsable += 1

    total = len(data)
    accuracy = correct / total if total > 0 else 0
    print(f"  judge/{dataset}/{model}: {correct}/{total} = {accuracy:.1%} [{judge_label}]")
    _append_result_log(
        result_root,
        f"judge accuracy [{judge_label}]: {accuracy} ({correct}/{total})",
    )
    if unparsable:
        _append_result_log(
            result_root,
            f"judge unparsable [{judge_label}]: {unparsable}",
        )

    return {
        "judge_model": judge_label,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "status": "valid",
        "notes": (
            ("" if unparsable == 0 else f"unparsable={unparsable}")
            if not judge_backend or judge_backend == judge_label
            else (
                f"backend={judge_backend}"
                if unparsable == 0
                else f"backend={judge_backend},unparsable={unparsable}"
            )
        ),
    }


def _macro_average(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None, 0
    return sum(valid) / len(valid), len(valid)


def summarize_model_results(
    model,
    datasets,
    exact_results,
    judge_results,
    posfix="",
    output_dir=None,
    skip_judge_eval=False,
):
    summary_root = _get_summary_root(model, posfix=posfix, output_dir=output_dir)
    exact_values = [
        exact_results.get(dataset, {}).get("accuracy")
        if exact_results.get(dataset) is not None else None
        for dataset in datasets
    ]
    exact_macro, exact_count = _macro_average(exact_values)
    judge_values = []
    judge_labels = []
    for dataset in datasets:
        result = judge_results.get(dataset)
        if result is None:
            judge_values.append(None)
            continue
        judge_values.append(result.get("accuracy"))
        judge_label = result.get("judge_model")
        if judge_label:
            judge_labels.append(judge_label)

    judge_macro, judge_count = _macro_average(judge_values)
    judge_label = judge_labels[0] if judge_labels else _judge_short_name(DEFAULT_JUDGE_MODEL)
    primary_label = None
    primary_macro = None
    primary_count = 0
    if judge_macro is not None and not skip_judge_eval:
        primary_label = f"judge [{judge_label}]"
        primary_macro = judge_macro
        primary_count = judge_count
    elif exact_macro is not None:
        primary_label = "exact"
        primary_macro = exact_macro
        primary_count = exact_count

    print(f"Summary for {model}:")
    if primary_macro is None:
        print("  primary macro avg: n/a")
        _append_result_log(summary_root, "primary macro avg: n/a")
    else:
        print(
            f"  primary macro avg ({primary_label}): {primary_macro:.1%} "
            f"({primary_count}/{len(datasets)} datasets)"
        )
        _append_result_log(
            summary_root,
            f"primary macro avg ({primary_label}): {primary_macro} "
            f"({primary_count}/{len(datasets)} datasets)",
        )

    if judge_macro is None:
        note = "skipped (--skip-judge-eval)" if skip_judge_eval else "n/a"
        print(f"  judge macro avg [{judge_label}]: {note}")
        _append_result_log(summary_root, f"judge macro avg [{judge_label}]: {note}")
    else:
        print(
            f"  judge macro avg [{judge_label}]: {judge_macro:.1%} "
            f"({judge_count}/{len(datasets)} datasets)"
        )
        _append_result_log(
            summary_root,
            f"judge macro avg [{judge_label}]: {judge_macro} "
            f"({judge_count}/{len(datasets)} datasets)",
        )

    if exact_macro is None:
        print("  exact macro avg: n/a")
        _append_result_log(summary_root, "exact macro avg: n/a")
    else:
        print(
            f"  exact macro avg: {exact_macro:.1%} "
            f"({exact_count}/{len(datasets)} datasets)"
        )
        _append_result_log(
            summary_root,
            f"exact macro avg: {exact_macro} ({exact_count}/{len(datasets)} datasets)",
        )


# ── Main ─────────────────────────────────────────────────────────────────

# For standalone use: fill in these lists and run `python check.py` directly.
MODELS_TO_CHECK = []
DATASETS = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reasoning benchmark evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", nargs="+", default=None,
        help="Model key(s) to evaluate. Overrides MODELS_TO_CHECK.",
    )
    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="Dataset(s) to evaluate. Overrides DATASETS.",
    )
    parser.add_argument(
        "--posfix", default="",
        help="Results directory suffix (e.g. '_run1').",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Explicit results root directory. Overrides --posfix if set.",
    )
    parser.add_argument(
        "--skip-judge-eval", action="store_true",
        help="Skip the GPT-5.2 judge metric and only report deterministic exact-match scores.",
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help=(
            "Judge model for answer-presence evaluation "
            f"(default: {DEFAULT_JUDGE_MODEL}; gpt-5.2 auto-resolves like HELMET)."
        ),
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow evaluation on partial prediction files. By default, evaluation fails if "
             "the prediction JSONLs do not cover the full dataset exactly once.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = args.model if args.model else MODELS_TO_CHECK
    datasets = args.dataset if args.dataset else DATASETS

    if not models or not datasets:
        print("Nothing to evaluate. Provide --model/--dataset CLI args, "
              "or edit MODELS_TO_CHECK/DATASETS in check.py.")
        return

    posfix = args.posfix
    output_dir = args.output_dir
    incomplete_failures = []

    for model in models:
        exact_results = {}
        judge_results = {}
        model_incomplete = []
        completion_by_dataset = {}
        for dataset in datasets:
            completion = _prediction_completion(model, dataset, posfix, output_dir=output_dir)
            completion_by_dataset[dataset] = completion
            if not completion["is_complete"]:
                files_desc = ", ".join(Path(path).name for path in completion["files"]) if completion["files"] else "no files"
                message = (
                    f"{model}/{dataset} predictions incomplete: "
                    f"completed={completion['completed']}/{completion['expected_total']}, "
                    f"rows={completion['total_rows']}, files={files_desc}"
                )
                result_root = _get_result_root(model, dataset, posfix, output_dir=output_dir)
                print(message)
                _append_result_log(result_root, f"incomplete predictions: {message}")
                if not args.allow_partial:
                    incomplete_failures.append(message)
                    model_incomplete.append(message)
        if model_incomplete and not args.allow_partial:
            continue
        for dataset in datasets:
            completion = completion_by_dataset[dataset]
            if not completion["is_complete"] and not args.allow_partial:
                continue
            print(f"Evaluating {model} on {dataset}:")
            exact_results[dataset] = check_exact_match(
                model=model, dataset=dataset, posfix=posfix, output_dir=output_dir
            )
            if not args.skip_judge_eval:
                judge_results[dataset] = check_llm_judge(
                    model=model,
                    dataset=dataset,
                    posfix=posfix,
                    output_dir=output_dir,
                    judge_model=args.judge_model,
                )
        summarize_model_results(
            model=model,
            datasets=datasets,
            exact_results=exact_results,
            judge_results=judge_results,
            posfix=posfix,
            output_dir=output_dir,
            skip_judge_eval=args.skip_judge_eval,
        )

    if incomplete_failures and not args.allow_partial:
        raise SystemExit(
            "Refusing to evaluate partial reasoning predictions:\n"
            + "\n".join(f"  - {message}" for message in incomplete_failures)
        )


if __name__ == "__main__":
    main()
