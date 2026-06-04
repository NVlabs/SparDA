#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# eval/run_accuracy.sh — Run SparDA benchmark configurations locally
# =============================================================================
#
# Runs separate local jobs for each combination of:
#   benchmark × model × attention_config [× seq_length]
#
# Benchmarks handle internal multi-GPU parallelism.
#
# Usage:
#   bash eval/run_accuracy.sh [OPTIONS]
#
# Examples:
#   bash eval/run_accuracy.sh --dry-run                                     # Preview all jobs
#   bash eval/run_accuracy.sh --benchmarks helmet,ruler --models nosa       # Subset
#   bash eval/run_accuracy.sh --seq-lengths 16384,32768 --block-hit-rate    # Custom lengths
#   bash eval/run_accuracy.sh --configs dense,sparse --resume               # Resume dense+sparse
#   bash eval/run_accuracy.sh --configs sparda
#
# Environment variable overrides (all have CLI equivalents):
#   NOSA_MODEL_PATH, MINICPM_MODEL_PATH, OUTPUT_DIR
# Extra cache tuning (env-only): SPARDA_RUNTIME_CACHE_ROOT, SPARDA_PERSISTENT_CACHE_ROOT,
# SPARDA_CACHE_USER
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =====================
# Defaults
# =====================
NOSA_MODEL_PATH="${NOSA_MODEL_PATH:-openbmb/NOSA-8B}"
MINICPM_MODEL_PATH="${MINICPM_MODEL_PATH:-openbmb/MiniCPM4.1-8B}"
NOSA_SPARDA_INDEXER_PATH="${NOSA_SPARDA_INDEXER_PATH:-}"
MINICPM_SPARDA_INDEXER_PATH="${MINICPM_SPARDA_INDEXER_PATH:-}"

BENCHMARKS="helmet,longbench,ruler,reasoning"
MODELS="minicpm,nosa"
CONFIGS="dense,sparse,sparda,infinigen"
SEQ_LENGTHS=""                              # Empty = use model native max for HELMET and RULER
LONGBENCH_LENGTHS=""                        # Empty = use model native max

# Common flags
BLOCK_HIT_RATE=0
RESUME=0
SKIP_GPT4_EVAL=0
EVAL_ONLY=0
PREDICT_ONLY=0
DRY_RUN=0
RUN_MODE="${RUN_MODE:-local}"
LOCAL_CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES:-}"
DEFAULT_GPU_COUNT="${DEFAULT_GPU_COUNT:-8}"

LOCAL_WORKDIR="${LOCAL_WORKDIR:-${REPO_ROOT}}"

SPARDA_CACHE_USER="${SPARDA_CACHE_USER:-${LOGNAME:-${USER:-$(id -un 2>/dev/null || echo user)}}}"
export SPARDA_CACHE_USER

# Log directory (host-visible path on shared filesystem)
LOG_DIR_EXPLICIT=0
if [[ -n "${LOG_DIR+x}" ]]; then
    LOG_DIR_EXPLICIT=1
fi
LOG_DIR="${LOG_DIR:-}"

# Root output directory for benchmark result artifacts
OUTPUT_DIR_EXPLICIT=0
if [[ -n "${OUTPUT_DIR+x}" ]]; then
    OUTPUT_DIR_EXPLICIT=1
fi
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_WORKDIR}/results}"

# =====================
# CLI parsing
# =====================
usage() {
    cat <<'HELP'
Usage: bash eval/run_accuracy.sh [OPTIONS]

Benchmark selection:
  --benchmarks LIST       Comma-separated: helmet,longbench,ruler,reasoning (default: all)
  --models LIST           Comma-separated: minicpm,nosa (default: both)
  --configs LIST          Comma-separated: dense,sparse,sparda,infinigen (default: all)

Sequence lengths:
  --seq-lengths LIST      For HELMET & RULER (default: model native max)
  --longbench-lengths LIST  For LongBench (default: model native max)

Model paths:
  --nosa-model-path PATH      (default: openbmb/NOSA-8B)
  --minicpm-model-path PATH   (default: openbmb/MiniCPM4.1-8B)
  --nosa-sparda-indexer-path PATH
                              Override the NOSA SparDA checkpoint path.
  --minicpm-sparda-indexer-path PATH
                              Override the MiniCPM SparDA checkpoint path.

Common flags:
  --block-hit-rate        Collect block hit-rate statistics (sparse only)
  --resume                Resume interrupted runs (keep existing predictions)
  --skip-gpt4-eval        Skip GPT-4 judge evaluation (HELMET longqa/summ)
  --eval-only             Skip prediction/inference and evaluate existing outputs
                          for all benchmarks
  --predict-only          Skip evaluation, run prediction only (Reasoning)
  --output-dir DIR        Root output directory (default: <repo>/results).
                          Benchmarks write under DIR/<benchmark>/<model>/<config>/...

Other:
  --run-mode local        Only local mode is supported.
  --log-dir DIR           Log directory (default: <repo>/logs/benchmarks)
  --dry-run               Print job commands without running them
  --local-cuda-visible-devices LIST
                          Override CUDA_VISIBLE_DEVICES
  --help                  Show this help
HELP
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks)          BENCHMARKS="$2"; shift 2 ;;
        --models)              MODELS="$2"; shift 2 ;;
        --configs)             CONFIGS="$2"; shift 2 ;;
        --seq-lengths)         SEQ_LENGTHS="$2"; shift 2 ;;
        --longbench-lengths)   LONGBENCH_LENGTHS="$2"; shift 2 ;;
        --nosa-model-path)     NOSA_MODEL_PATH="$2"; shift 2 ;;
        --minicpm-model-path)  MINICPM_MODEL_PATH="$2"; shift 2 ;;
        --nosa-sparda-indexer-path) NOSA_SPARDA_INDEXER_PATH="$2"; shift 2 ;;
        --minicpm-sparda-indexer-path) MINICPM_SPARDA_INDEXER_PATH="$2"; shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2"; OUTPUT_DIR_EXPLICIT=1; shift 2 ;;
        --block-hit-rate)      BLOCK_HIT_RATE=1; shift ;;
        --resume)              RESUME=1; shift ;;
        --skip-gpt4-eval)      SKIP_GPT4_EVAL=1; shift ;;
        --eval-only)           EVAL_ONLY=1; shift ;;
        --predict-only)        PREDICT_ONLY=1; shift ;;
        --run-mode)            RUN_MODE="$2"; shift 2 ;;
        --local-cuda-visible-devices) LOCAL_CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --dry-run)             DRY_RUN=1; shift ;;
        --log-dir)             LOG_DIR="$2"; shift 2 ;;
        --help|-h)             usage ;;
        *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

# Convert comma-separated lists to arrays
IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS"
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra CONFIG_LIST <<< "$CONFIGS"
if [[ -n "$SEQ_LENGTHS" ]]; then
    IFS=',' read -ra SEQ_LENGTH_LIST <<< "$SEQ_LENGTHS"
else
    SEQ_LENGTH_LIST=()
fi
if [[ -n "$LONGBENCH_LENGTHS" ]]; then
    IFS=',' read -ra LONGBENCH_LENGTH_LIST <<< "$LONGBENCH_LENGTHS"
else
    LONGBENCH_LENGTH_LIST=()
fi

SUBMITTED=0
SKIPPED=0

# Safe increment that doesn't fail under set -e when value is 0
inc_submitted() { SUBMITTED=$((SUBMITTED + 1)); }
inc_skipped() { SKIPPED=$((SKIPPED + 1)); }

# =====================
# Helper functions
# =====================

canonicalize_path() {
    local path="$1"
    [[ -z "$path" ]] && return 0
    if [[ "$path" == "~"* ]]; then
        path="${HOME}${path:1}"
    fi
    if [[ "$path" != /* ]]; then
        path="$(pwd)/$path"
    fi
    printf '%s\n' "$path"
}

LOG_DIR="$(canonicalize_path "$LOG_DIR")"
LOCAL_WORKDIR="$(canonicalize_path "$LOCAL_WORKDIR")"
NOSA_SPARDA_INDEXER_PATH="$(canonicalize_path "$NOSA_SPARDA_INDEXER_PATH")"
MINICPM_SPARDA_INDEXER_PATH="$(canonicalize_path "$MINICPM_SPARDA_INDEXER_PATH")"
if [[ "$OUTPUT_DIR_EXPLICIT" == "0" ]]; then
    OUTPUT_DIR="${LOCAL_WORKDIR}/results"
fi
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="${LOCAL_WORKDIR}/logs/benchmarks"
fi
LOG_DIR="$(canonicalize_path "$LOG_DIR")"

SPARDA_DEFAULT_PERSISTENT_CACHE_ROOT="${SPARDA_DEFAULT_PERSISTENT_CACHE_ROOT:-${HOME}/.cache/sparda/infllm_v2}"
export SPARDA_DEFAULT_PERSISTENT_CACHE_ROOT

# Mirror training launcher cache setup, but keep heavyweight Hugging Face model
# caches on persistent storage so judge/model dependencies are reused.
CACHE_SETUP_SNIPPET=$(cat <<'CACHE_EOF'
# Keep temp/build caches in /tmp, but persist HF caches across runs.
CACHE_USER="${SPARDA_CACHE_USER:-${LOGNAME:-${USER:-$(id -un 2>/dev/null || echo user)}}}"
SPARDA_RUNTIME_CACHE_ROOT="${SPARDA_RUNTIME_CACHE_ROOT:-/tmp/${CACHE_USER}/sparda/infllm_v2}"
if [[ -n "${SPARDA_INFLLMV2_CACHE_ROOT:-}" && -z "${SPARDA_PERSISTENT_CACHE_ROOT:-}" ]]; then
    SPARDA_PERSISTENT_CACHE_ROOT="${SPARDA_INFLLMV2_CACHE_ROOT}"
else
    SPARDA_PERSISTENT_CACHE_ROOT="${SPARDA_PERSISTENT_CACHE_ROOT:-${SPARDA_DEFAULT_PERSISTENT_CACHE_ROOT:-${HOME}/.cache/sparda/infllm_v2}}"
fi
mkdir -p "${SPARDA_RUNTIME_CACHE_ROOT}/tmp" \
         "${SPARDA_RUNTIME_CACHE_ROOT}/triton" \
         "${SPARDA_RUNTIME_CACHE_ROOT}/torch_extensions" \
         "${SPARDA_RUNTIME_CACHE_ROOT}/xdg" \
         "${SPARDA_PERSISTENT_CACHE_ROOT}/huggingface" \
         "${SPARDA_PERSISTENT_CACHE_ROOT}/huggingface/hub" \
         "${SPARDA_PERSISTENT_CACHE_ROOT}/huggingface/datasets" \
         "${SPARDA_PERSISTENT_CACHE_ROOT}/huggingface/transformers"
export TMPDIR="${SPARDA_RUNTIME_CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${SPARDA_RUNTIME_CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${SPARDA_RUNTIME_CACHE_ROOT}/torch_extensions"
export XDG_CACHE_HOME="${SPARDA_RUNTIME_CACHE_ROOT}/xdg"
export HF_HOME="${SPARDA_PERSISTENT_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${SPARDA_PERSISTENT_CACHE_ROOT}/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${SPARDA_PERSISTENT_CACHE_ROOT}/huggingface/datasets"
export TRANSFORMERS_CACHE="${SPARDA_PERSISTENT_CACHE_ROOT}/huggingface/transformers"
CACHE_EOF
)

if [[ -n "$NOSA_SPARDA_INDEXER_PATH" && ! -f "$NOSA_SPARDA_INDEXER_PATH" ]]; then
    echo "Invalid --nosa-sparda-indexer-path: ${NOSA_SPARDA_INDEXER_PATH} (file not found)" >&2
    exit 1
fi
if [[ -n "$MINICPM_SPARDA_INDEXER_PATH" && ! -f "$MINICPM_SPARDA_INDEXER_PATH" ]]; then
    echo "Invalid --minicpm-sparda-indexer-path: ${MINICPM_SPARDA_INDEXER_PATH} (file not found)" >&2
    exit 1
fi

if [[ "$EVAL_ONLY" == "1" && "$PREDICT_ONLY" == "1" ]]; then
    echo "--eval-only and --predict-only cannot be used together." >&2
    exit 1
fi

case "$RUN_MODE" in
    local) ;;
    *)
        echo "Invalid --run-mode: ${RUN_MODE} (expected: local)" >&2
        exit 1
        ;;
esac

get_model_path() {
    case "$1" in
        nosa)     echo "$NOSA_MODEL_PATH" ;;
        minicpm)  echo "$MINICPM_MODEL_PATH" ;;
        *)        echo "UNKNOWN_MODEL_$1" ;;
    esac
}

# Native max sequence length (without long-context extensions)
get_native_max() {
    case "$1" in
        nosa)     echo 32768 ;;
        minicpm)  echo 65536 ;;
        *)        echo 32768 ;;
    esac
}

get_sparda_indexer_path() {
    local model="$1"

    case "$model" in
        minicpm)
            if [[ -n "$MINICPM_SPARDA_INDEXER_PATH" ]]; then
                printf '%s\n' "$MINICPM_SPARDA_INDEXER_PATH"
                return 0
            fi
            ;;
        nosa)
            if [[ -n "$NOSA_SPARDA_INDEXER_PATH" ]]; then
                printf '%s\n' "$NOSA_SPARDA_INDEXER_PATH"
                return 0
            fi
            ;;
        *)
            return 1
            ;;
    esac
}

log_missing_sparda_indexer() {
    local benchmark="$1"
    local model="$2"
    local config="$3"
    local indexer_path

    indexer_path="$(get_sparda_indexer_path "$model")"
    if [[ -z "$indexer_path" ]]; then
        case "$model" in
            minicpm) echo "  [SKIP] ${benchmark} ${model} ${config}: SparDA indexer checkpoint not provided; pass --minicpm-sparda-indexer-path" ;;
            nosa) echo "  [SKIP] ${benchmark} ${model} ${config}: SparDA indexer checkpoint not provided; pass --nosa-sparda-indexer-path" ;;
            *) echo "  [SKIP] ${benchmark} ${model} ${config}: SparDA indexer checkpoint not provided" ;;
        esac
    else
        echo "  [SKIP] ${benchmark} ${model} ${config}: missing SparDA indexer checkpoint at ${indexer_path}"
    fi
}

# Build attention config flags. Returns "SKIP" if config is invalid for this model.
build_config_flags() {
    local model="$1" config="$2"
    case "$config" in
        dense)
            echo "--dense"
            ;;
        sparse)
            # Default for both models (no flags needed).
            # NOSA default = sparse with built-in SparDA indexer.
            # MiniCPM default = sparse with q_current.
            echo ""
            ;;
        sparda)
            # Resolve the user-provided indexer checkpoint.
            local indexer_path
            indexer_path="$(get_sparda_indexer_path "$model")"
            if [[ ! -f "$indexer_path" ]]; then
                echo "SKIP_NO_INDEXER"
                return
            fi
            echo "--sparda --indexer-path $indexer_path"
            ;;
        infinigen)
            # InfiniGen works with both MiniCPM and NOSA models
            echo "--infinigen"
            ;;
        *)
            echo "SKIP"
            ;;
    esac
}

# Build --long-context flag if seq_len exceeds model native max
build_long_context_flags() {
    local model="$1" seq_len="$2"
    local native_max
    native_max=$(get_native_max "$model")
    if (( seq_len > native_max )); then
        echo "--long-context"
    fi
}

# Build common flags shared across all benchmarks
build_common_flags() {
    local flags=""
    [[ "$BLOCK_HIT_RATE" == "1" ]] && flags+=" --block-hit-rate"
    [[ "$RESUME" == "1" ]] && flags+=" --resume"
    [[ "$EVAL_ONLY" == "1" ]] && flags+=" --eval-only"
    echo "$flags"
}

count_nonempty_lines() {
    local path="$1"
    [[ -f "$path" ]] || { printf '%s\n' 0; return; }
    awk 'NF {count++} END {print count + 0}' "$path"
}

reasoning_dataset_file() {
    case "$1" in
        math-500) printf '%s\n' "${REPO_ROOT}/NOSA/benchmarks/Reasoning/math-500.jsonl" ;;
        aime24)   printf '%s\n' "${REPO_ROOT}/NOSA/benchmarks/Reasoning/aime24.jsonl" ;;
        aime25)   printf '%s\n' "${REPO_ROOT}/NOSA/benchmarks/Reasoning/aime25.jsonl" ;;
        *) return 1 ;;
    esac
}

reasoning_mode_name() {
    local model="$1"
    local config="$2"
    if [[ "$model" == "nosa" ]]; then
        case "$config" in
            dense)     printf '%s\n' "nosa_dense" ;;
            sparse)    printf '%s\n' "nosa" ;;
            sparda)    printf '%s\n' "nosa_sparda" ;;
            infinigen) printf '%s\n' "nosa_infinigen" ;;
            *) return 1 ;;
        esac
    else
        case "$config" in
            dense)     printf '%s\n' "dense" ;;
            sparse)    printf '%s\n' "minicpm" ;;
            sparda)    printf '%s\n' "minicpm_sparda" ;;
            infinigen) printf '%s\n' "infinigen" ;;
            *) return 1 ;;
        esac
    fi
}

reasoning_model_mode_key() {
    local model="$1"
    local config="$2"
    local model_key="8b_${model}"
    local mode
    mode="$(reasoning_mode_name "$model" "$config")" || return 1
    if [[ "$model_key" == *"$mode" ]]; then
        printf '%s\n' "$model_key"
    else
        printf '%s\n' "${model_key}_${mode}"
    fi
}

reasoning_gen_len() {
    case "$1" in
        nosa) printf '%s\n' 8192 ;;
        *)    printf '%s\n' 65536 ;;
    esac
}

reasoning_temperature() {
    case "$1" in
        nosa) printf '%s\n' "1.0" ;;
        *)    printf '%s\n' "0.9" ;;
    esac
}

host_visible_path() {
    local path="$1"
    printf '%s\n' "$path"
}

reasoning_remaining_samples() {
    local model="$1"
    local config="$2"
    local output_root
    output_root="$(benchmark_output_dir reasoning "$config" "$model")"
    output_root="$(host_visible_path "$output_root")"
    local model_mode_key
    model_mode_key="$(reasoning_model_mode_key "$model" "$config")" || return 1
    local gen_len
    gen_len="$(reasoning_gen_len "$model")"
    local temperature
    temperature="$(reasoning_temperature "$model")"

    local total_remaining=0
    local ds ds_file total_count output_file completed_count remaining_count
    for ds in math-500 aime24 aime25; do
        ds_file="$(reasoning_dataset_file "$ds")" || return 1
        total_count="$(count_nonempty_lines "$ds_file")"
        output_file="${output_root}/${ds}/${model_mode_key}/${ds}-${model_mode_key}-${gen_len}-${temperature}-rank_0.jsonl"
        completed_count="$(count_nonempty_lines "$output_file")"
        if (( completed_count > total_count )); then
            completed_count="$total_count"
        fi
        remaining_count=$((total_count - completed_count))
        total_remaining=$((total_remaining + remaining_count))
    done
    printf '%s\n' "$total_remaining"
}

reasoning_gpu_count_for_submit() {
    local model="$1"
    local config="$2"
    local gpu_count="${DEFAULT_GPU_COUNT}"

    if [[ "$RESUME" != "1" || "$EVAL_ONLY" == "1" ]]; then
        printf '%s\n' "$gpu_count"
        return 0
    fi

    local remaining
    if ! remaining="$(reasoning_remaining_samples "$model" "$config")"; then
        printf '%s\n' "$gpu_count"
        return 0
    fi

    if (( remaining <= 0 )); then
        printf '%s\n' 1
    else
        local scaled_gpu_count=$(( remaining / 8 ))
        if (( scaled_gpu_count < 1 )); then
            scaled_gpu_count=1
        elif (( scaled_gpu_count > gpu_count )); then
            scaled_gpu_count="$gpu_count"
        fi
        printf '%s\n' "$scaled_gpu_count"
    fi
}

benchmark_output_dir() {
    local benchmark="$1"
    local config="$2"
    local model="$3"
    printf '%s/%s/%s/%s\n' "$OUTPUT_DIR" "$benchmark" "$model" "$config"
}

gpu_device_list() {
    local gpu_count="$1"
    local devices=()
    local gpu
    for ((gpu = 0; gpu < gpu_count; gpu++)); do
        devices+=("$gpu")
    done
    local joined
    IFS=',' read -r -a _ <<< "${devices[*]}"
    joined="${devices[*]}"
    printf '%s\n' "${joined// /,}"
}

run_job_local() {
    local job_name="$1"
    local python_cmd="$2"
    local gpu_count="${3:-8}"
    local log_file="${LOG_DIR}/${job_name}_local_$(date -u +%Y%m%dT%H%M%S).log"

    mkdir -p "$LOG_DIR"

    {
        echo "=== local benchmark run ==="
        echo "job_name: ${job_name}"
        echo "run_mode: ${RUN_MODE}"
        echo "local_workdir: ${LOCAL_WORKDIR}"
        echo "gpu_count_hint: ${gpu_count}"
        if [[ -n "$LOCAL_CUDA_VISIBLE_DEVICES" ]]; then
            echo "local_cuda_visible_devices: ${LOCAL_CUDA_VISIBLE_DEVICES}"
        elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            echo "local_cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
        fi
        printf 'command: %s\n' "$python_cmd"
        echo
    } > "$log_file"

    echo "  Running locally: $job_name"
    set +e
    (
        set -euo pipefail
        cd "$LOCAL_WORKDIR"
        if [[ -f /venv/sparda/bin/activate ]]; then
            # shellcheck disable=SC1091
            source /venv/sparda/bin/activate
        fi
        eval "$CACHE_SETUP_SNIPPET"
        if [[ -n "$LOCAL_CUDA_VISIBLE_DEVICES" ]]; then
            export CUDA_VISIBLE_DEVICES="$LOCAL_CUDA_VISIBLE_DEVICES"
        fi
        eval "$python_cmd"
    ) >> "$log_file" 2>&1
    local exit_code=$?
    set -e

    {
        echo
        echo "exit_code: ${exit_code}"
        echo "finished_utc: $(date -u +%FT%TZ)"
    } >> "$log_file"

    if [[ "$exit_code" -ne 0 ]]; then
        echo "  Failed locally: $job_name (see ${log_file})" >&2
        return "$exit_code"
    fi

    echo "  Completed locally: $job_name"
}

# Run or preview a single local job.
submit_job() {
    local job_name="$1"
    local python_cmd="$2"
    local _time_limit="${3:-}"
    local gpu_count="${4:-8}"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY RUN] $job_name"
        echo "            GPUs: ${gpu_count}"
        echo "            $python_cmd"
        echo ""
        inc_submitted
        return
    fi

    run_job_local "$job_name" "$python_cmd" "$gpu_count"
    inc_submitted
}

# =====================
# HELMET
# =====================
submit_helmet_jobs() {
    echo "--- HELMET ---"
    local gpu_count="${DEFAULT_GPU_COUNT}"
    for model in "${MODEL_LIST[@]}"; do
        for config in "${CONFIG_LIST[@]}"; do
            local config_flags
            config_flags=$(build_config_flags "$model" "$config")
            if [[ "$config_flags" == "SKIP" ]]; then
                inc_skipped; continue
            fi
            if [[ "$config_flags" == "SKIP_NO_INDEXER" ]]; then
                log_missing_sparda_indexer "helmet" "$model" "$config"
                inc_skipped; continue
            fi

            local lengths=("${SEQ_LENGTH_LIST[@]}")
            if [[ ${#lengths[@]} -eq 0 ]]; then
                lengths=("$(get_native_max "$model")")
            fi

            for seq_len in "${lengths[@]}"; do
                local model_path lc_flags common_flags
                model_path=$(get_model_path "$model")
                lc_flags=$(build_long_context_flags "$model" "$seq_len")
                common_flags=$(build_common_flags)

                local extra_flags=""
                [[ "$SKIP_GPT4_EVAL" == "1" ]] && extra_flags+=" --skip-gpt4-eval"
                extra_flags+=" --output-dir $(benchmark_output_dir helmet "$config" "$model")"

                local seq_k=$((seq_len / 1024))
                local job_name="helmet_${model}_${config}_${seq_k}k"

                local cmd="python -u NOSA/benchmarks/HELMET/run.py"
                cmd+=" --model-path ${model_path}"
                cmd+=" --input-max-length ${seq_len}"
                cmd+="${config_flags:+ ${config_flags}}"
                cmd+="${lc_flags:+ ${lc_flags}}"
                cmd+="${common_flags}"
                cmd+="${extra_flags}"

                submit_job "$job_name" "$cmd" "" "$gpu_count"
            done
        done
    done
    echo ""
}

# =====================
# LongBench
# =====================
submit_longbench_jobs() {
    echo "--- LongBench ---"
    local gpu_count="${DEFAULT_GPU_COUNT}"
    for model in "${MODEL_LIST[@]}"; do
        for config in "${CONFIG_LIST[@]}"; do
            local config_flags
            config_flags=$(build_config_flags "$model" "$config")
            if [[ "$config_flags" == "SKIP" ]]; then
                inc_skipped; continue
            fi
            if [[ "$config_flags" == "SKIP_NO_INDEXER" ]]; then
                log_missing_sparda_indexer "longbench" "$model" "$config"
                inc_skipped; continue
            fi

            # If no lengths specified, use model native max
            local lengths=("${LONGBENCH_LENGTH_LIST[@]}")
            if [[ ${#lengths[@]} -eq 0 ]]; then
                lengths=("$(get_native_max "$model")")
            fi

            for seq_len in "${lengths[@]}"; do
                local model_path lc_flags common_flags
                model_path=$(get_model_path "$model")
                lc_flags=$(build_long_context_flags "$model" "$seq_len")
                common_flags=$(build_common_flags)

                local extra_flags=""
                extra_flags+=" --output-dir $(benchmark_output_dir longbench "$config" "$model")"

                local seq_k=$((seq_len / 1024))
                local job_name="longbench_${model}_${config}_${seq_k}k"

                local cmd="python -u NOSA/benchmarks/LongBench/run.py"
                cmd+=" --model-path ${model_path}"
                cmd+=" --max-length ${seq_len}"
                cmd+="${config_flags:+ ${config_flags}}"
                cmd+="${lc_flags:+ ${lc_flags}}"
                cmd+="${common_flags}"
                cmd+="${extra_flags}"

                submit_job "$job_name" "$cmd" "" "$gpu_count"
            done
        done
    done
    echo ""
}

# =====================
# RULER
# =====================
submit_ruler_jobs() {
    echo "--- RULER ---"
    local gpu_count="${DEFAULT_GPU_COUNT}"
    for model in "${MODEL_LIST[@]}"; do
        for config in "${CONFIG_LIST[@]}"; do
            local config_flags
            config_flags=$(build_config_flags "$model" "$config")
            if [[ "$config_flags" == "SKIP" ]]; then
                inc_skipped; continue
            fi
            if [[ "$config_flags" == "SKIP_NO_INDEXER" ]]; then
                log_missing_sparda_indexer "ruler" "$model" "$config"
                inc_skipped; continue
            fi

            local lengths=("${SEQ_LENGTH_LIST[@]}")
            if [[ ${#lengths[@]} -eq 0 ]]; then
                lengths=("$(get_native_max "$model")")
            fi

            for seq_len in "${lengths[@]}"; do
                local model_path lc_flags common_flags
                model_path=$(get_model_path "$model")
                lc_flags=$(build_long_context_flags "$model" "$seq_len")
                common_flags=$(build_common_flags)

                local extra_flags=""
                extra_flags+=" --output-dir $(benchmark_output_dir ruler "$config" "$model")"

                local seq_k=$((seq_len / 1024))
                local job_name="ruler_${model}_${config}_${seq_k}k"

                local cmd="python -u NOSA/benchmarks/RULER/scripts/run.py"
                cmd+=" --model-path ${model_path}"
                cmd+=" --seq-len ${seq_len}"
                cmd+="${config_flags:+ ${config_flags}}"
                cmd+="${lc_flags:+ ${lc_flags}}"
                cmd+="${common_flags}"
                cmd+="${extra_flags}"

                submit_job "$job_name" "$cmd" "" "$gpu_count"
            done
        done
    done
    echo ""
}

# =====================
# Reasoning
# =====================
submit_reasoning_jobs() {
    echo "--- Reasoning ---"
    for model in "${MODEL_LIST[@]}"; do
        for config in "${CONFIG_LIST[@]}"; do
            local config_flags
            config_flags=$(build_config_flags "$model" "$config")
            if [[ "$config_flags" == "SKIP" ]]; then
                inc_skipped; continue
            fi
            if [[ "$config_flags" == "SKIP_NO_INDEXER" ]]; then
                log_missing_sparda_indexer "reasoning" "$model" "$config"
                inc_skipped; continue
            fi

            local model_path common_flags
            model_path=$(get_model_path "$model")
            common_flags=$(build_common_flags)

            local gpu_count
            gpu_count="$(reasoning_gpu_count_for_submit "$model" "$config")"
            local reasoning_gpus
            reasoning_gpus="$(gpu_device_list "$gpu_count")"

            local extra_flags=""
            [[ "$PREDICT_ONLY" == "1" ]] && extra_flags+=" --predict-only"
            extra_flags+=" --output-dir $(benchmark_output_dir reasoning "$config" "$model")"

            # Reasoning uses --model key (e.g. 8b_nosa, 8b_minicpm)
            local model_key="8b_${model}"
            local job_name="reasoning_${model}_${config}"

            local cmd="python -u NOSA/benchmarks/Reasoning/run.py"
            cmd+=" --model ${model_key}"
            cmd+=" --model-path ${model_path}"
            cmd+=" --gpus ${reasoning_gpus}"
            cmd+="${config_flags:+ ${config_flags}}"
            cmd+="${common_flags}"
            cmd+="${extra_flags}"

            submit_job "$job_name" "$cmd" "" "$gpu_count"
        done
    done
    echo ""
}

# =====================
# Main
# =====================
echo "========================================"
echo " SparDA Benchmark Sweep"
echo "========================================"
echo "Benchmarks:    ${BENCHMARKS}"
echo "Models:        ${MODELS}"
echo "Configs:       ${CONFIGS}"
if [[ -n "$SEQ_LENGTHS" ]]; then
    echo "Seq lengths:   ${SEQ_LENGTHS} (HELMET/RULER)"
else
    echo "Seq lengths:   model native max (HELMET/RULER)"
fi
if [[ -n "$LONGBENCH_LENGTHS" ]]; then
    echo "LongBench len: ${LONGBENCH_LENGTHS}"
else
    echo "LongBench len: model native max"
fi
echo "NOSA path:     ${NOSA_MODEL_PATH}"
echo "MiniCPM path:  ${MINICPM_MODEL_PATH}"
echo "SparDA MiniCPM indexer: ${MINICPM_SPARDA_INDEXER_PATH:-<not provided>}"
echo "SparDA NOSA indexer:    ${NOSA_SPARDA_INDEXER_PATH:-<not provided>}"
echo "GPU policy:     default=${DEFAULT_GPU_COUNT}"
[[ -n "$OUTPUT_DIR" ]] && echo "Output dir:    ${OUTPUT_DIR}"
echo "Block hit-rate: $( [[ "$BLOCK_HIT_RATE" == "1" ]] && echo "yes" || echo "no" )"
echo "Resume:         $( [[ "$RESUME" == "1" ]] && echo "yes" || echo "no" )"
echo "Eval only:      $( [[ "$EVAL_ONLY" == "1" ]] && echo "yes" || echo "no" )"
echo "Predict only:   $( [[ "$PREDICT_ONLY" == "1" ]] && echo "yes" || echo "no" )"
echo "Skip GPT4 eval: $( [[ "$SKIP_GPT4_EVAL" == "1" ]] && echo "yes" || echo "no" )"
echo "Local workdir:  ${LOCAL_WORKDIR}"
echo "Local CUDA:     ${LOCAL_CUDA_VISIBLE_DEVICES:-<inherit>}"
[[ "$DRY_RUN" == "1" ]] && echo "*** DRY RUN MODE ***"
echo "========================================"
echo ""

for bench in "${BENCHMARK_LIST[@]}"; do
    case "$bench" in
        helmet)    submit_helmet_jobs ;;
        longbench) submit_longbench_jobs ;;
        ruler)     submit_ruler_jobs ;;
        reasoning) submit_reasoning_jobs ;;
        *) echo "Unknown benchmark: $bench (skipping)" >&2 ;;
    esac
done

echo "========================================"
echo "Total: ${SUBMITTED} jobs, ${SKIPPED} skipped"
echo "========================================"
