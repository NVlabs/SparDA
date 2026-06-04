#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

# Local single-node training launcher for SparDA Forecast/indexer training.
# The script intentionally avoids scheduler-specific options so the release
# workflow is reproducible on any machine with the required Python/CUDA stack.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/training/train_indexer.py"
CONFIG_FILE="${PROJECT_ROOT}/training/accelerate_config.yaml"

RUN_MODE="${RUN_MODE:-local}"
GPUS="${GPUS:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOCAL_WORKDIR="${LOCAL_WORKDIR:-${PROJECT_ROOT}}"
ACTIVATE_VENV="${ACTIVATE_VENV:-1}"

SHOW_HELP=0
TRAIN_ARGS=()

usage() {
    local exit_code="${1:-0}"
    cat <<'EOF'
Usage: ./training/run_train.sh [LAUNCHER OPTIONS] --model_path MODEL [TRAIN ARGS...]

Local launcher options:
  --run-mode local            Only local single-node mode is supported.
  --gpus N                    Number of local GPU processes (default: GPUS or 8)
  --main-process-port PORT    Local master port (default: derived from current process)
  --target-batch-size N       Effective batch target for grad accumulation (default: 32)
  --batch-size N              Per-device train batch size (default: 1)
  --local-workdir DIR         Directory to run from (default: repo root)
  --no-venv-activate          Do not auto-source /venv/sparda/bin/activate if present
  --help, -h                  Show this help

Training arguments are passed through to training/train_indexer.py.
--model_path is required.

Examples:
  ./training/run_train.sh --model_path openbmb/MiniCPM4.1-8B --seq_len 32768 --steps 2000
  GPUS=4 ./training/run_train.sh --model_path openbmb/NOSA-8B
EOF
    exit "$exit_code"
}

while (($#)); do
    case "$1" in
        --run-mode)
            RUN_MODE="$2"
            shift 2
            ;;
        --run-mode=*)
            RUN_MODE="${1#*=}"
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift
            ;;
        --main-process-port)
            MAIN_PROCESS_PORT="$2"
            shift 2
            ;;
        --main-process-port=*)
            MAIN_PROCESS_PORT="${1#*=}"
            shift
            ;;
        --target-batch-size)
            TARGET_BATCH_SIZE="$2"
            shift 2
            ;;
        --target-batch-size=*)
            TARGET_BATCH_SIZE="${1#*=}"
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --batch-size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --local-workdir)
            LOCAL_WORKDIR="$2"
            shift 2
            ;;
        --local-workdir=*)
            LOCAL_WORKDIR="${1#*=}"
            shift
            ;;
        --no-venv-activate)
            ACTIVATE_VENV=0
            shift
            ;;
        --num-machines|--num-machines=*|--machine-rank|--machine-rank=*|--main-process-ip|--main-process-ip=*)
            echo "Unsupported launcher option: $1. This release launcher supports single-node, multi-GPU training only." >&2
            echo "For multi-node training, adapt training/train_indexer.py to your environment launcher or scheduler." >&2
            exit 2
            ;;
        --help|-h)
            SHOW_HELP=1
            shift
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$SHOW_HELP" == "1" ]]; then
    usage 0
fi

if [[ "$RUN_MODE" != "local" ]]; then
    echo "Unsupported --run-mode: ${RUN_MODE}. This release launcher supports local single-node mode only." >&2
    exit 2
fi

for value_name in GPUS TARGET_BATCH_SIZE BATCH_SIZE; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "Invalid ${value_name}='${value}'; expected a non-negative integer." >&2
        exit 2
    fi
done

if [[ "$GPUS" -lt 1 || "$TARGET_BATCH_SIZE" -lt 1 || "$BATCH_SIZE" -lt 1 ]]; then
    echo "GPUS, TARGET_BATCH_SIZE, and BATCH_SIZE must be positive." >&2
    exit 2
fi

if [[ -z "${MAIN_PROCESS_PORT}" ]]; then
    MAIN_PROCESS_PORT=$((20000 + ($$ % 40000)))
fi

if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

canonicalize_path() {
    local path="$1"
    [[ -z "$path" ]] && return 0
    if [[ "$path" == "~"* ]]; then
        path="${HOME}${path:1}"
    fi
    if [[ "$path" != /* ]]; then
        path="$(pwd)/$path"
    fi
    local dir base resolved_dir
    dir="$(dirname "$path")"
    base="$(basename "$path")"
    if [[ -e "$path" || -d "$dir" ]]; then
        resolved_dir="$(cd "$dir" 2>/dev/null && pwd -P)" || {
            printf '%s\n' "$path"
            return 0
        }
        printf '%s/%s\n' "$resolved_dir" "$base"
    else
        printf '%s\n' "$path"
    fi
}

PROJECT_ROOT="$(canonicalize_path "${PROJECT_ROOT}")"
TRAIN_SCRIPT="${PROJECT_ROOT}/training/train_indexer.py"
CONFIG_FILE="${PROJECT_ROOT}/training/accelerate_config.yaml"
LOCAL_WORKDIR="$(canonicalize_path "$LOCAL_WORKDIR")"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Accelerate config not found at ${CONFIG_FILE}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Training script not found at ${TRAIN_SCRIPT}" >&2
    exit 1
fi

validate_train_args() {
    local has_model_path=0
    local arg
    for arg in "${TRAIN_ARGS[@]}"; do
        case "$arg" in
            --model_path|--model-path|--model_path=*|--model-path=*)
                has_model_path=1
                ;;
        esac
    done
    if [[ "$has_model_path" == "0" ]]; then
        echo "--model_path is required." >&2
        exit 2
    fi
}

setup_local_caches() {
    local cache_user="${USER:-$(id -un 2>/dev/null || echo user)}"
    local cache_root="${SPARDA_INFLLMV2_CACHE_ROOT:-/tmp/${cache_user}/sparda/infllm_v2}"
    mkdir -p "${cache_root}/tmp" \
             "${cache_root}/triton" \
             "${cache_root}/torch_extensions" \
             "${cache_root}/xdg"
    export TMPDIR="${cache_root}/tmp"
    export TRITON_CACHE_DIR="${cache_root}/triton"
    export TORCH_EXTENSIONS_DIR="${cache_root}/torch_extensions"
    export XDG_CACHE_HOME="${cache_root}/xdg"
}

validate_train_args

if [[ "$ACTIVATE_VENV" == "1" && -f /venv/sparda/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/sparda/bin/activate
fi

setup_local_caches

grad_accum_steps=$(( TARGET_BATCH_SIZE / (BATCH_SIZE * GPUS) ))
if [[ "${grad_accum_steps}" -lt 1 ]]; then
    grad_accum_steps=1
fi

echo "=== SparDA local training ==="
echo "Workdir: ${LOCAL_WORKDIR}"
echo "GPUs: ${GPUS}"
echo "Machines: 1"
echo "Main process: localhost:${MAIN_PROCESS_PORT}"
echo "Gradient accumulation: ${grad_accum_steps}"
echo "============================="

cd "${LOCAL_WORKDIR}"

accelerate launch \
  --config_file "${CONFIG_FILE}" \
  --num_processes "${GPUS}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip localhost \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${TRAIN_SCRIPT}" \
  --gradient_accumulation_steps "${grad_accum_steps}" \
  "${TRAIN_ARGS[@]}"
