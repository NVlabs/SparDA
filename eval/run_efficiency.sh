#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Local efficiency sweep launcher.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_SCRIPT="${REPO_ROOT}/NOSA/benchmarks/Efficiency/bench.sh"
PROFILE_SCRIPT="${REPO_ROOT}/NOSA/benchmarks/Efficiency/profile_breakdown.sh"

default_cache_user() {
    printf '%s\n' "${LOGNAME:-${USER:-$(id -un 2>/dev/null || echo user)}}"
}

RUN_MODE="${RUN_MODE:-local}"

BACKEND="nosi"
TOOLS="bench"
MODELS="minicpm,nosa"
CONFIGS="dense-no-offload,sparse,sparse-no-offload,sparda,infinigen"
PREFETCH_CTAS_LIST=""
SEQ_LENS="32K,64K,96K,128K"
BATCH_SIZES="4,8,16,32,64,128"

NOSA_MODEL_PATH="${NOSA_MODEL_PATH:-openbmb/NOSA-8B}"
MINICPM_MODEL_PATH="${MINICPM_MODEL_PATH:-openbmb/MiniCPM4.1-8B}"
NOSA_INDEXER_PATH="${NOSA_INDEXER_PATH:-}"
MINICPM_INDEXER_PATH="${MINICPM_INDEXER_PATH:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
TEST_N="${TEST_N:-2}"
DATASET_NAME="${DATASET_NAME:-emozilla/pg19}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
COOLDOWN_SECS="${COOLDOWN_SECS:-1}"
RESUME=0
DRY_RUN=0

RUN_NAME="${RUN_NAME:-}"
LOG_ROOT_DEFAULT="${REPO_ROOT}/results/efficiency"
LOG_DIR="${LOG_DIR:-}"

LOCAL_CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES:-}"

CACHE_USER="${SPARDA_CACHE_USER:-$(default_cache_user)}"
if [[ -n "${SPARDA_INFLLMV2_CACHE_ROOT:-}" && -z "${SPARDA_PERSISTENT_CACHE_ROOT:-}" ]]; then
    SPARDA_PERSISTENT_CACHE_ROOT="${SPARDA_INFLLMV2_CACHE_ROOT}"
else
    SPARDA_PERSISTENT_CACHE_ROOT="${SPARDA_PERSISTENT_CACHE_ROOT:-${HOME}/.cache/sparda/infllm_v2}"
fi
SPARDA_RUNTIME_CACHE_ROOT="${SPARDA_RUNTIME_CACHE_ROOT:-/tmp/${CACHE_USER}/sparda/infllm_v2}"
export SPARDA_CACHE_USER="${CACHE_USER}"
export SPARDA_RUNTIME_CACHE_ROOT
export SPARDA_PERSISTENT_CACHE_ROOT

usage() {
    local exit_code="${1:-0}"
    cat <<'EOF'
Usage: bash eval/run_efficiency.sh [OPTIONS]

Mode:
  --run-mode local           Only local mode is supported.

Sweep selection:
  --tools LIST               Comma-separated override (default: bench)
  --models LIST              Comma-separated override (default: minicpm,nosa)
  --configs LIST             Comma-separated override
                             (default: dense-no-offload,sparse,sparse-no-offload,
                             sparda,infinigen)
                             available extras: sparda-no-offload,sparda-no-prefetch
                             sparda adaptively sets PREFETCH_CTAS on A100/H100
                             unless PREFETCH_CTAS is already set explicitly
  --prefetch-ctas LIST       Comma-separated fixed CTA counts for sparda.
                             If set, each 'sparda' entry expands into
                             sparda_<N>ctas cases
  --seq-lens LIST            Comma-separated override (default: 32K,64K,96K,128K)
  --batch-sizes LIST         Comma-separated override (default: 4,8,16,32,64,128)

Model paths:
  --nosa-model-path PATH     NOSA model path (default: openbmb/NOSA-8B)
  --minicpm-model-path PATH  MiniCPM model path (default: openbmb/MiniCPM4.1-8B)
  --nosa-indexer-path PATH   NOSA SparDA indexer checkpoint.
  --minicpm-indexer-path PATH
                             MiniCPM SparDA indexer checkpoint.
                             Release packages do not include indexer weights.

Run settings:
  --log-dir DIR              Root log/output directory for this sweep
  --run-name NAME            Run name used in the log directory.
  --max-new-tokens N         Forwarded to both tools (default: 4)
  --test-n N                 Forwarded to both tools (default: 2)
  --dataset NAME             Dataset name (default: emozilla/pg19)
  --dataset-split SPLIT      Dataset split (default: test)
  --cooldown-secs N          Sleep between runs (default: 1)
  --resume                   Reuse an existing log dir, keep summary.tsv, and
                             skip cases already recorded there
  --local-cuda-visible-devices LIST
                             Override CUDA_VISIBLE_DEVICES

Other:
  --dry-run                  Print commands without submitting/running
  --help, -h                 Show this help
EOF
    exit "$exit_code"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-mode) RUN_MODE="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --tools) TOOLS="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --configs) CONFIGS="$2"; shift 2 ;;
        --prefetch-ctas) PREFETCH_CTAS_LIST="$2"; shift 2 ;;
        --seq-lens) SEQ_LENS="$2"; shift 2 ;;
        --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
        --nosa-model-path) NOSA_MODEL_PATH="$2"; shift 2 ;;
        --minicpm-model-path) MINICPM_MODEL_PATH="$2"; shift 2 ;;
        --nosa-indexer-path) NOSA_INDEXER_PATH="$2"; shift 2 ;;
        --minicpm-indexer-path) MINICPM_INDEXER_PATH="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --test-n) TEST_N="$2"; shift 2 ;;
        --dataset) DATASET_NAME="$2"; shift 2 ;;
        --dataset-split) DATASET_SPLIT="$2"; shift 2 ;;
        --cooldown-secs) COOLDOWN_SECS="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --local-cuda-visible-devices) LOCAL_CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 2 ;;
    esac
done

IFS=',' read -r -a TOOL_LIST <<< "$TOOLS"
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"
IFS=',' read -r -a SEQ_LEN_LIST <<< "$SEQ_LENS"
IFS=',' read -r -a BATCH_SIZE_LIST <<< "$BATCH_SIZES"

PASS_COUNT=0
FAIL_COUNT=0
OOM_COUNT=0
SKIP_COUNT=0
RUN_INDEX=0
TOTAL_CASES=$(( ${#TOOL_LIST[@]} * ${#MODEL_LIST[@]} * ${#CONFIG_LIST[@]} * ${#SEQ_LEN_LIST[@]} * ${#BATCH_SIZE_LIST[@]} ))

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

NOSA_INDEXER_PATH="$(canonicalize_path "$NOSA_INDEXER_PATH")"
MINICPM_INDEXER_PATH="$(canonicalize_path "$MINICPM_INDEXER_PATH")"

if [[ -n "$NOSA_INDEXER_PATH" && ! -f "$NOSA_INDEXER_PATH" ]]; then
    echo "Invalid --nosa-indexer-path: ${NOSA_INDEXER_PATH} (file not found)" >&2
    exit 1
fi
if [[ -n "$MINICPM_INDEXER_PATH" && ! -f "$MINICPM_INDEXER_PATH" ]]; then
    echo "Invalid --minicpm-indexer-path: ${MINICPM_INDEXER_PATH} (file not found)" >&2
    exit 1
fi

get_model_path() {
    case "$1" in
        minicpm) printf '%s\n' "$MINICPM_MODEL_PATH" ;;
        nosa) printf '%s\n' "$NOSA_MODEL_PATH" ;;
        *)
            echo "Unsupported model: $1" >&2
            return 1
            ;;
    esac
}

get_indexer_path() {
    case "$1" in
        minicpm) printf '%s\n' "$MINICPM_INDEXER_PATH" ;;
        nosa) printf '%s\n' "$NOSA_INDEXER_PATH" ;;
        *)
            return 1
            ;;
    esac
}

quote_cmd() {
    local arg
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
    printf '\n'
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "$value"
}

join_by_comma() {
    local first=1 item
    for item in "$@"; do
        if [[ "$first" == "1" ]]; then
            printf '%s' "$item"
            first=0
        else
            printf ',%s' "$item"
        fi
    done
    printf '\n'
}

validate_prefetch_ctas_value() {
    local value="$1"
    [[ "$value" =~ ^[0-9]+$ ]] || return 1
    (( 10#$value > 0 )) || return 1
}

expand_sparda_prefetch_configs() {
    local config item expanded_name
    local -a cta_values=()
    local -a expanded=()
    local -A seen=()

    if [[ -z "$PREFETCH_CTAS_LIST" ]]; then
        return 0
    fi

    IFS=',' read -r -a cta_values <<< "$PREFETCH_CTAS_LIST"
    for item in "${cta_values[@]}"; do
        item="$(trim_whitespace "$item")"
        [[ -n "$item" ]] || continue
        if ! validate_prefetch_ctas_value "$item"; then
            echo "ERROR: invalid --prefetch-ctas value '$item' (expected positive integer)." >&2
            exit 2
        fi
    done

    for config in "${CONFIG_LIST[@]}"; do
        config="$(trim_whitespace "$config")"
        [[ -n "$config" ]] || continue
        if [[ "$config" == "sparda" ]]; then
            for item in "${cta_values[@]}"; do
                item="$(trim_whitespace "$item")"
                [[ -n "$item" ]] || continue
                expanded_name="sparda_${item}ctas"
                if [[ -z "${seen[$expanded_name]+x}" ]]; then
                    expanded+=("$expanded_name")
                    seen["$expanded_name"]=1
                fi
            done
            continue
        fi
        if [[ -z "${seen[$config]+x}" ]]; then
            expanded+=("$config")
            seen["$config"]=1
        fi
    done

    CONFIG_LIST=("${expanded[@]}")
    CONFIGS="$(join_by_comma "${CONFIG_LIST[@]}")"
}

expand_sparda_prefetch_configs
TOTAL_CASES=$(( ${#TOOL_LIST[@]} * ${#MODEL_LIST[@]} * ${#CONFIG_LIST[@]} * ${#SEQ_LEN_LIST[@]} * ${#BATCH_SIZE_LIST[@]} ))

detect_gpu_model_tag() {
    local name upper_name
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required to detect GPU hardware info for log names." >&2
        return 1
    fi

    name="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
    name="$(trim_whitespace "$name")"
    if [[ -z "$name" ]]; then
        echo "ERROR: failed to detect GPU name with nvidia-smi." >&2
        return 1
    fi

    upper_name="$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')"

    if [[ "$upper_name" =~ (H200|H100|A100|A10G|A10|V100|L40S|L40|L4|T4|B200|B100) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ "$upper_name" =~ RTX[[:space:]_-]*A([0-9]{4}) ]]; then
        printf 'RTXA%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ "$upper_name" =~ RTX[[:space:]_-]*([0-9]{4}) ]]; then
        printf 'RTX%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    echo "ERROR: unsupported GPU name format for hardware tag: ${name}" >&2
    return 1
}

detect_pcie_gen_width() {
    local out gen width bdf bdf_short sysfs speed_str gt_s

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required to detect PCIe link info for log names." >&2
        return 1
    fi

    out="$(nvidia-smi --id=0 --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
    if [[ -n "$out" ]]; then
        IFS=',' read -r gen width <<< "$out"
        gen="$(trim_whitespace "$gen")"
        width="$(trim_whitespace "$width")"
        if [[ "$gen" =~ ^[0-9]+$ ]] && [[ "$width" =~ ^[0-9]+$ ]]; then
            printf 'PCIEGen%sx%s\n' "$gen" "$width"
            return 0
        fi
    fi

    bdf="$(nvidia-smi --id=0 --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -n 1)"
    bdf="$(trim_whitespace "$bdf")"
    if [[ -z "$bdf" ]]; then
        echo "ERROR: failed to detect PCIe bus id with nvidia-smi." >&2
        return 1
    fi

    bdf_short="${bdf,,}"
    bdf_short="${bdf_short#gpu-}"
    for sysfs in /sys/bus/pci/devices/*"${bdf_short: -7}" /sys/bus/pci/devices/*"${bdf_short}"; do
        [[ -e "$sysfs/max_link_speed" && -e "$sysfs/max_link_width" ]] || continue
        speed_str="$(<"$sysfs/max_link_speed")"
        width="$(<"$sysfs/max_link_width")"
        speed_str="$(trim_whitespace "$speed_str")"
        width="$(trim_whitespace "$width")"
        gt_s="${speed_str%% *}"
        case "$gt_s" in
            2.5) gen=1 ;;
            5.0) gen=2 ;;
            8.0) gen=3 ;;
            16.0) gen=4 ;;
            32.0) gen=5 ;;
            64.0) gen=6 ;;
            *) gen="" ;;
        esac
        if [[ -n "$gen" ]] && [[ "$width" =~ ^[0-9]+$ ]]; then
            printf 'PCIEGen%sx%s\n' "$gen" "$width"
            return 0
        fi
    done

    echo "ERROR: failed to detect max PCIe generation/width for GPU 0." >&2
    return 1
}

detect_gpu_hw_tag() {
    local gpu_tag pcie_tag
    gpu_tag="$(detect_gpu_model_tag)" || return 1
    pcie_tag="$(detect_pcie_gen_width)" || return 1
    printf '%s_%s\n' "$gpu_tag" "$pcie_tag"
}

adaptive_sparda_prefetch_ctas() {
    local batch_size="$1"

    case "$GPU_HW_TAG" in
        H100*)
            if (( batch_size <= 16 )); then
                printf '%s\n' 16
            else
                printf '%s\n' 32
            fi
            ;;
        A100*)
            if (( batch_size <= 32 )); then
                printf '%s\n' 16
            else
                printf '%s\n' 32
            fi
            ;;
        *)
            return 1
            ;;
    esac
}

case_key() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6"
}

SUMMARY_HEADER=$'tool\tmodel\tconfig\trequested_backend\tseq_len\tbatch_size\tgpu_hw_tag\tstatus\texit_code\tlog_file'
declare -A EXISTING_CASE_STATUS=()
declare -A EXISTING_CASE_LOG=()
declare -A SELECTED_TOOL_SET=()
declare -A SELECTED_MODEL_SET=()
declare -A SELECTED_CONFIG_SET=()
declare -A SELECTED_SEQ_LEN_SET=()
declare -A SELECTED_BATCH_SIZE_SET=()
SUMMARY_TSV=""
SUMMARY_TXT=""
GPU_HW_TAG=""

check_existing_hardware_tag() {
    local config_file existing_tag
    config_file="${LOG_DIR}/run_config.txt"
    if [[ ! -f "$config_file" ]]; then
        return 0
    fi

    existing_tag="$(sed -n 's/^gpu_hw_tag: //p' "$config_file" | head -n 1)"
    existing_tag="$(trim_whitespace "$existing_tag")"
    if [[ -z "$existing_tag" ]]; then
        return 0
    fi

    if [[ "$existing_tag" != "$GPU_HW_TAG" ]]; then
        echo "ERROR: log directory ${LOG_DIR} was created for GPU hardware tag ${existing_tag}," >&2
        echo "but current hardware tag is ${GPU_HW_TAG}. Reusing the same run directory across" >&2
        echo "different hardware types is not allowed." >&2
        return 1
    fi
}

build_selection_sets() {
    local item
    SELECTED_TOOL_SET=()
    SELECTED_MODEL_SET=()
    SELECTED_CONFIG_SET=()
    SELECTED_SEQ_LEN_SET=()
    SELECTED_BATCH_SIZE_SET=()

    for item in "${TOOL_LIST[@]}"; do
        [[ -n "$item" ]] && SELECTED_TOOL_SET["$item"]=1
    done
    for item in "${MODEL_LIST[@]}"; do
        [[ -n "$item" ]] && SELECTED_MODEL_SET["$item"]=1
    done
    for item in "${CONFIG_LIST[@]}"; do
        [[ -n "$item" ]] && SELECTED_CONFIG_SET["$item"]=1
    done
    for item in "${SEQ_LEN_LIST[@]}"; do
        [[ -n "$item" ]] && SELECTED_SEQ_LEN_SET["$item"]=1
    done
    for item in "${BATCH_SIZE_LIST[@]}"; do
        [[ -n "$item" ]] && SELECTED_BATCH_SIZE_SET["$item"]=1
    done
}

case_matches_current_sweep() {
    local tool="$1"
    local model="$2"
    local config="$3"
    local backend="$4"
    local seq_len="$5"
    local batch_size="$6"

    [[ -n "${SELECTED_TOOL_SET[$tool]+x}" ]] || return 1
    [[ -n "${SELECTED_MODEL_SET[$model]+x}" ]] || return 1
    [[ -n "${SELECTED_CONFIG_SET[$config]+x}" ]] || return 1
    [[ "$backend" == "$BACKEND" ]] || return 1
    [[ -n "${SELECTED_SEQ_LEN_SET[$seq_len]+x}" ]] || return 1
    [[ -n "${SELECTED_BATCH_SIZE_SET[$batch_size]+x}" ]] || return 1
}

remove_case_artifacts() {
    local tool="$1"
    local model="$2"
    local config="$3"
    local seq_len="$4"
    local batch_size="$5"
    local case_dir log_stem output_prefix
    local -a matches=()
    local nullglob_was_set=0

    case_dir="${LOG_DIR}/${tool}/${model}/${config}"
    [[ -d "$case_dir" ]] || return 0

    log_stem="${tool}_${model}_${config}_${BACKEND}_L${seq_len}_B${batch_size}"
    output_prefix="${case_dir}/${log_stem}"

    if shopt -q nullglob; then
        nullglob_was_set=1
    fi
    shopt -s nullglob
    matches=( "${output_prefix}"* )
    if [[ "$nullglob_was_set" != "1" ]]; then
        shopt -u nullglob
    fi

    ((${#matches[@]})) || return 0
    rm -f "${matches[@]}"
}

clear_log_dir_for_fresh_run() {
    [[ "$RESUME" == "1" || "$DRY_RUN" == "1" ]] && return 0
    [[ -d "$LOG_DIR" ]] || return 0

    case "$LOG_DIR" in
        ""|"/"|".")
            echo "ERROR: refusing to clear unsafe log directory: ${LOG_DIR}" >&2
            return 1
            ;;
    esac

    local tool model config seq_len batch_size
    for tool in "${TOOL_LIST[@]}"; do
        for model in "${MODEL_LIST[@]}"; do
            for config in "${CONFIG_LIST[@]}"; do
                for seq_len in "${SEQ_LEN_LIST[@]}"; do
                    for batch_size in "${BATCH_SIZE_LIST[@]}"; do
                        remove_case_artifacts "$tool" "$model" "$config" "$seq_len" "$batch_size"
                    done
                done
            done
        done
    done

    local summary_tsv="${LOG_DIR}/summary.tsv"
    [[ -f "$summary_tsv" ]] || return 0

    local existing_header tmp_summary
    existing_header="$(head -n 1 "$summary_tsv" 2>/dev/null || true)"
    existing_header="${existing_header%$'\r'}"
    if [[ "$existing_header" != "$SUMMARY_HEADER" ]]; then
        echo "ERROR: cannot reuse malformed summary file: ${summary_tsv}" >&2
        return 1
    fi

    tmp_summary="${summary_tsv}.tmp.$$"
    printf '%s\n' "$SUMMARY_HEADER" > "$tmp_summary"

    local line_no=0
    local tool_name model_name config_name backend seq_len_name batch_size_name gpu_hw_tag status exit_code log_file
    while IFS=$'\t' read -r tool_name model_name config_name backend seq_len_name batch_size_name gpu_hw_tag status exit_code log_file; do
        line_no=$((line_no + 1))
        if [[ "$line_no" -eq 1 ]]; then
            continue
        fi
        [[ -n "$tool_name" ]] || continue

        if case_matches_current_sweep "$tool_name" "$model_name" "$config_name" "$backend" "$seq_len_name" "$batch_size_name"; then
            continue
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$tool_name" "$model_name" "$config_name" "$backend" "$seq_len_name" "$batch_size_name" \
            "$gpu_hw_tag" "$status" "$exit_code" "$log_file" >> "$tmp_summary"
    done < "$summary_tsv"

    mv "$tmp_summary" "$summary_tsv"
}

check_resume_config_compatibility() {
    [[ "$RESUME" == "1" ]] || return 0

    local config_file="${LOG_DIR}/run_config.txt"
    [[ -f "$config_file" ]] || return 0

    # Resume is allowed to add or remove sweep-list entries. Existing rows stay
    # in summary.tsv; the current invocation simply iterates the current lists.
    local mismatch=0
    local field expected existing
    while IFS='|' read -r field expected; do
        existing="$(sed -n "s/^${field}: //p" "$config_file" | head -n 1)"
        existing="$(trim_whitespace "$existing")"
        [[ -n "$existing" ]] || continue

        if [[ "$existing" != "$expected" ]]; then
            echo "ERROR: cannot resume ${LOG_DIR}: ${field} is '${existing}', expected '${expected}'." >&2
            mismatch=1
        fi
    done <<EOF
requested_backend|${BACKEND}
nosa_model_path|${NOSA_MODEL_PATH}
minicpm_model_path|${MINICPM_MODEL_PATH}
max_new_tokens|${MAX_NEW_TOKENS}
test_n|${TEST_N}
dataset|${DATASET_NAME}
dataset_split|${DATASET_SPLIT}
EOF

    [[ "$mismatch" -eq 0 ]]
}

init_summary_tsv() {
    if [[ -f "$SUMMARY_TSV" ]]; then
        local existing_header
        existing_header="$(head -n 1 "$SUMMARY_TSV" 2>/dev/null || true)"
        existing_header="${existing_header%$'\r'}"
        if [[ "$existing_header" != "$SUMMARY_HEADER" ]]; then
            echo "ERROR: cannot resume with malformed summary file: ${SUMMARY_TSV}" >&2
            return 1
        fi
        return 0
    fi

    printf '%s\n' "$SUMMARY_HEADER" > "$SUMMARY_TSV"
}

recount_existing_statuses() {
    PASS_COUNT=0
    FAIL_COUNT=0
    OOM_COUNT=0
    SKIP_COUNT=0

    local key status tool model config backend seq_len batch_size
    for key in "${!EXISTING_CASE_STATUS[@]}"; do
        IFS=$'\t' read -r tool model config backend seq_len batch_size <<< "$key"
        if ! case_matches_current_sweep "$tool" "$model" "$config" "$backend" "$seq_len" "$batch_size"; then
            continue
        fi
        status="${EXISTING_CASE_STATUS[$key]}"
        increment_status_count "$status"
    done
}

increment_status_count() {
    local status="$1"
    case "$status" in
        pass) PASS_COUNT=$((PASS_COUNT + 1)) ;;
        fail) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
        oom) OOM_COUNT=$((OOM_COUNT + 1)) ;;
        skip) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
    esac
}

decrement_status_count() {
    local status="$1"
    case "$status" in
        pass) PASS_COUNT=$((PASS_COUNT - 1)) ;;
        fail) FAIL_COUNT=$((FAIL_COUNT - 1)) ;;
        oom) OOM_COUNT=$((OOM_COUNT - 1)) ;;
        skip) SKIP_COUNT=$((SKIP_COUNT - 1)) ;;
    esac
}

resume_should_skip_status() {
    local status="$1"
    case "$status" in
        pass|oom) return 0 ;;
        *) return 1 ;;
    esac
}

load_existing_summary() {
    [[ -f "$SUMMARY_TSV" ]] || return 0

    local line_no=0
    local tool model config backend seq_len batch_size gpu_hw_tag status exit_code log_file key
    while IFS=$'\t' read -r tool model config backend seq_len batch_size gpu_hw_tag status exit_code log_file; do
        line_no=$((line_no + 1))
        if [[ "$line_no" -eq 1 ]]; then
            continue
        fi
        [[ -n "$tool" ]] || continue

        key="$(case_key "$tool" "$model" "$config" "$backend" "$seq_len" "$batch_size")"
        EXISTING_CASE_STATUS["$key"]="$status"
        EXISTING_CASE_LOG["$key"]="$log_file"
    done < "$SUMMARY_TSV"

    recount_existing_statuses
}

write_summary_row() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" >> "$SUMMARY_TSV"
}

record_case_status() {
    local key="$1"
    local tool="$2"
    local model="$3"
    local config="$4"
    local backend="$5"
    local seq_len="$6"
    local batch_size="$7"
    local status="$8"
    local exit_code="$9"
    local log_file="${10}"
    local previous_status=""

    previous_status="${EXISTING_CASE_STATUS[$key]:-}"
    if [[ -n "$previous_status" ]]; then
        decrement_status_count "$previous_status"
    fi

    write_summary_row "$tool" "$model" "$config" "$backend" "$seq_len" "$batch_size" "$GPU_HW_TAG" "$status" "$exit_code" "$log_file"
    EXISTING_CASE_STATUS["$key"]="$status"
    EXISTING_CASE_LOG["$key"]="$log_file"
    increment_status_count "$status"
}

detect_status() {
    local exit_code="$1"
    local log_file="$2"
    local oom_pattern='cuda out of memory|outofmemoryerror|torch\.cuda\.outofmemoryerror|killed|oom-kill|oom kill|out of memory'
    if [[ "$exit_code" -eq 0 ]]; then
        printf '%s\n' "pass"
        return
    fi
    if [[ "$exit_code" -eq 137 ]]; then
        printf '%s\n' "oom"
        return
    fi
    if command -v rg >/dev/null 2>&1; then
        if rg -i -q "$oom_pattern" "$log_file"; then
            printf '%s\n' "oom"
        else
            printf '%s\n' "fail"
        fi
    elif grep -Eiq "$oom_pattern" "$log_file"; then
        printf '%s\n' "oom"
    else
        printf '%s\n' "fail"
    fi
}

find_smaller_batch_oom() {
    local tool="$1"
    local model="$2"
    local config="$3"
    local seq_len="$4"
    local batch_size="$5"
    local prior_batch key

    [[ "$batch_size" =~ ^[0-9]+$ ]] || return 1

    for prior_batch in "${BATCH_SIZE_LIST[@]}"; do
        [[ "$prior_batch" =~ ^[0-9]+$ ]] || continue
        if (( 10#$prior_batch >= 10#$batch_size )); then
            continue
        fi
        key="$(case_key "$tool" "$model" "$config" "$BACKEND" "$seq_len" "$prior_batch")"
        if [[ "${EXISTING_CASE_STATUS[$key]:-}" == "oom" ]]; then
            printf '%s\n' "$prior_batch"
            return 0
        fi
    done

    return 1
}

run_case() {
    local tool="$1"
    local model="$2"
    local config="$3"
    local seq_len="$4"
    local batch_size="$5"

    local model_path
    model_path="$(get_model_path "$model")"

    local tool_script=""
    case "$tool" in
        bench) tool_script="$BENCH_SCRIPT" ;;
        profile_breakdown) tool_script="$PROFILE_SCRIPT" ;;
        *)
            echo "Unsupported tool: $tool" >&2
            return 1
            ;;
    esac

    local log_stem="${tool}_${model}_${config}_${BACKEND}_L${seq_len}_B${batch_size}"
    local case_dir="${LOG_DIR}/${tool}/${model}/${config}"
    local log_file="${case_dir}/${log_stem}.log"
    local output_prefix="${case_dir}/${log_stem}"
    local skip_reason=""
    local indexer_path=""
    local key

    key="$(case_key "$tool" "$model" "$config" "$BACKEND" "$seq_len" "$batch_size")"

    case "$config" in
        sparda|sparda-no-offload|sparda-no-prefetch|sparda_[0-9]*ctas)
            indexer_path="$(get_indexer_path "$model")"
            if [[ -z "$indexer_path" ]]; then
                case "$model" in
                    minicpm) skip_reason="SparDA indexer checkpoint not provided; pass --minicpm-indexer-path" ;;
                    nosa) skip_reason="SparDA indexer checkpoint not provided; pass --nosa-indexer-path" ;;
                    *) skip_reason="SparDA indexer checkpoint not provided for model: ${model}" ;;
                esac
            elif [[ ! -e "$indexer_path" ]]; then
                skip_reason="SparDA indexer checkpoint not found: ${indexer_path}"
            fi
            ;;
    esac

    RUN_INDEX=$((RUN_INDEX + 1))

    if [[ "$RESUME" == "1" && -n "${EXISTING_CASE_STATUS[$key]+x}" ]]; then
        if resume_should_skip_status "${EXISTING_CASE_STATUS[$key]}"; then
            echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> resume-skip (${EXISTING_CASE_STATUS[$key]})"
            return 0
        fi
        echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> resume-rerun (${EXISTING_CASE_STATUS[$key]})"
    fi

    local smaller_oom_batch=""
    smaller_oom_batch="$(find_smaller_batch_oom "$tool" "$model" "$config" "$seq_len" "$batch_size" || true)"
    if [[ -n "$smaller_oom_batch" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> dry-run auto-oom"
            echo "    reason: smaller batch size B${smaller_oom_batch} already OOM"
            return 0
        fi
        mkdir -p "$case_dir"
        {
            echo "=== run_efficiency auto-oom ==="
            echo "time_utc: $(date -u +%FT%TZ)"
            echo "tool: $tool"
            echo "model: $model"
            echo "config: $config"
            echo "requested_backend: $BACKEND"
            echo "seq_len: $seq_len"
            echo "batch_size: $batch_size"
            echo "reason: smaller batch size B${smaller_oom_batch} already recorded as OOM"
        } > "$log_file"
        record_case_status "$key" "$tool" "$model" "$config" "$BACKEND" "$seq_len" "$batch_size" "oom" "" "$log_file"
        echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> auto-oom (smaller batch B${smaller_oom_batch})"
        return 0
    fi

    if [[ -n "$skip_reason" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> dry-run skip"
            echo "    reason: ${skip_reason}"
            return 0
        fi
        mkdir -p "$case_dir"
        {
            echo "=== run_efficiency skip ==="
            echo "time_utc: $(date -u +%FT%TZ)"
            echo "tool: $tool"
            echo "model: $model"
            echo "config: $config"
            echo "requested_backend: $BACKEND"
            echo "seq_len: $seq_len"
                echo "batch_size: $batch_size"
                echo "reason: $skip_reason"
            } > "$log_file"
        record_case_status "$key" "$tool" "$model" "$config" "$BACKEND" "$seq_len" "$batch_size" "skip" "" "$log_file"
        echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> skip"
        return 0
    fi

    local -a cmd
    local effective_prefetch_ctas=""
    local prefetch_ctas_source=""
    cmd=(bash "$tool_script"
         --model-path "$model_path"
         --backend "$BACKEND"
         -B "$batch_size"
         -L "$seq_len"
         --max-new-tokens "$MAX_NEW_TOKENS"
         --test-n "$TEST_N"
         --dataset "$DATASET_NAME"
         --dataset-split "$DATASET_SPLIT")

    case "$config" in
        dense-no-offload)
            cmd+=(--dense --no-offload)
            ;;
        sparse)
            ;;
        dense)
            cmd+=(--dense --no-offload)
            ;;
        sparse-no-offload)
            cmd+=(--no-offload)
            ;;
        sparda)
            if [[ -n "${PREFETCH_CTAS:-}" ]]; then
                effective_prefetch_ctas="${PREFETCH_CTAS}"
                prefetch_ctas_source="env"
            else
                effective_prefetch_ctas="$(adaptive_sparda_prefetch_ctas "$batch_size" 2>/dev/null || true)"
                if [[ -n "$effective_prefetch_ctas" ]]; then
                    prefetch_ctas_source="adaptive"
                fi
            fi
            if [[ -n "$effective_prefetch_ctas" ]]; then
                cmd=(env PREFETCH_CTAS="$effective_prefetch_ctas" "${cmd[@]}")
            fi
            cmd+=(--sparda --indexer-path "$indexer_path")
            ;;
        sparda_[0-9]*ctas)
            effective_prefetch_ctas="${config#sparda_}"
            effective_prefetch_ctas="${effective_prefetch_ctas%ctas}"
            prefetch_ctas_source="config"
            cmd=(env PREFETCH_CTAS="$effective_prefetch_ctas" "${cmd[@]}")
            cmd+=(--sparda --indexer-path "$indexer_path")
            ;;
        sparda-no-offload)
            cmd+=(--sparda --no-offload --indexer-path "$indexer_path")
            ;;
        sparda-no-prefetch)
            cmd+=(--sparda-no-prefetch --indexer-path "$indexer_path")
            ;;
        infinigen)
            cmd+=(--infinigen)
            ;;
        *)
            if [[ "$DRY_RUN" == "1" ]]; then
                echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> dry-run skip"
                echo "    reason: unsupported config"
                return 0
            fi
            mkdir -p "$case_dir"
            {
                echo "=== run_efficiency skip ==="
                echo "time_utc: $(date -u +%FT%TZ)"
                echo "tool: $tool"
                echo "model: $model"
                echo "config: $config"
                echo "reason: unsupported config"
            } > "$log_file"
            record_case_status "$key" "$tool" "$model" "$config" "$BACKEND" "$seq_len" "$batch_size" "skip" "" "$log_file"
            echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> skip"
            return 0
            ;;
    esac

    if [[ "$tool" == "profile_breakdown" ]]; then
        cmd+=(--output-prefix "$output_prefix")
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> dry-run"
        printf '    '
        quote_cmd "${cmd[@]}"
        return 0
    fi

    mkdir -p "$case_dir"

    {
        echo "=== run_efficiency ==="
        echo "time_utc: $(date -u +%FT%TZ)"
        echo "tool: $tool"
        echo "model: $model"
        echo "config: $config"
        echo "requested_backend: $BACKEND"
        echo "seq_len: $seq_len"
        echo "batch_size: $batch_size"
        echo "model_path: $model_path"
        if [[ -n "$indexer_path" ]]; then
            echo "indexer_path: $indexer_path"
        fi
        if [[ -n "$effective_prefetch_ctas" ]]; then
            echo "prefetch_ctas: $effective_prefetch_ctas"
            echo "prefetch_ctas_source: $prefetch_ctas_source"
        fi
        printf 'command: '
        quote_cmd "${cmd[@]}"
        echo
    } > "$log_file"

    echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> running"

    "${cmd[@]}" >> "$log_file" 2>&1
    local exit_code=$?
    local status
    status="$(detect_status "$exit_code" "$log_file")"

    {
        echo
        echo "exit_code: $exit_code"
        echo "status: $status"
        echo "finished_utc: $(date -u +%FT%TZ)"
    } >> "$log_file"

    record_case_status "$key" "$tool" "$model" "$config" "$BACKEND" "$seq_len" "$batch_size" "$status" "$exit_code" "$log_file"

    echo "[$RUN_INDEX/$TOTAL_CASES] ${log_stem} -> ${status} (exit ${exit_code})"

    sleep "$COOLDOWN_SECS"
    return 0
}

run_local_sweep() {
    local maybe_activate="/venv/sparda/bin/activate"
    if [[ -n "$LOCAL_CUDA_VISIBLE_DEVICES" ]]; then
        export CUDA_VISIBLE_DEVICES="$LOCAL_CUDA_VISIBLE_DEVICES"
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        GPU_HW_TAG="$(detect_gpu_hw_tag 2>/dev/null || true)"
        GPU_HW_TAG="${GPU_HW_TAG:-LOCAL_DRY_RUN}"
    else
        GPU_HW_TAG="$(detect_gpu_hw_tag)" || exit 1
    fi

    if [[ -z "$RUN_NAME" ]]; then
        RUN_NAME="$GPU_HW_TAG"
    fi

    if [[ -z "$LOG_DIR" ]]; then
        LOG_DIR="${LOG_ROOT_DEFAULT}/${RUN_NAME}"
    fi
    LOG_DIR="$(canonicalize_path "$LOG_DIR")"
    if [[ "$DRY_RUN" != "1" ]]; then
        build_selection_sets
        check_existing_hardware_tag || exit 1
        clear_log_dir_for_fresh_run || exit 1
        mkdir -p "$LOG_DIR"

        SUMMARY_TSV="${LOG_DIR}/summary.tsv"
        SUMMARY_TXT="${LOG_DIR}/summary.txt"

        check_resume_config_compatibility || exit 1
        init_summary_tsv || exit 1
        load_existing_summary

        mkdir -p "${HOME}" \
                 "${SPARDA_RUNTIME_CACHE_ROOT}/tmp" \
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

        {
            echo "run_name: ${RUN_NAME}"
            echo "run_mode: ${RUN_MODE}"
            echo "log_dir: ${LOG_DIR}"
            echo "requested_backend: ${BACKEND}"
            echo "tools: ${TOOLS}"
            echo "models: ${MODELS}"
            echo "configs: ${CONFIGS}"
            echo "prefetch_ctas: ${PREFETCH_CTAS_LIST}"
            echo "seq_lens: ${SEQ_LENS}"
            echo "batch_sizes: ${BATCH_SIZES}"
            echo "resume: ${RESUME}"
            echo "nosa_model_path: ${NOSA_MODEL_PATH}"
            echo "minicpm_model_path: ${MINICPM_MODEL_PATH}"
            echo "nosa_indexer_path: ${NOSA_INDEXER_PATH:-<not provided>}"
            echo "minicpm_indexer_path: ${MINICPM_INDEXER_PATH:-<not provided>}"
            echo "max_new_tokens: ${MAX_NEW_TOKENS}"
            echo "test_n: ${TEST_N}"
            echo "dataset: ${DATASET_NAME}"
            echo "dataset_split: ${DATASET_SPLIT}"
            echo "cooldown_secs: ${COOLDOWN_SECS}"
            echo "gpu_hw_tag: ${GPU_HW_TAG}"
            echo "planned_cases: ${TOTAL_CASES}"
        } > "${LOG_DIR}/run_config.txt"
    fi

    echo "=== run_efficiency ==="
    echo "Run mode: ${RUN_MODE}"
    echo "Log directory: ${LOG_DIR}"
    echo "Requested backend: ${BACKEND}"
    echo "GPU hardware tag: ${GPU_HW_TAG}"
    echo "Planned cases: ${TOTAL_CASES}"
    echo "Resume: ${RESUME}"
    if [[ -n "$LOCAL_CUDA_VISIBLE_DEVICES" ]]; then
        echo "CUDA_VISIBLE_DEVICES: ${LOCAL_CUDA_VISIBLE_DEVICES}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    fi
    if [[ "$RESUME" == "1" ]]; then
        echo "Existing completed cases: ${#EXISTING_CASE_STATUS[@]}"
    fi

    if [[ "$DRY_RUN" != "1" && -f "$maybe_activate" ]]; then
        # shellcheck disable=SC1090
        source "$maybe_activate"
    fi

    for tool in "${TOOL_LIST[@]}"; do
        for model in "${MODEL_LIST[@]}"; do
            for config in "${CONFIG_LIST[@]}"; do
                for seq_len in "${SEQ_LEN_LIST[@]}"; do
                    for batch_size in "${BATCH_SIZE_LIST[@]}"; do
                        run_case "$tool" "$model" "$config" "$seq_len" "$batch_size"
                    done
                done
            done
        done
    done

    echo
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "Dry run complete."
    else
        {
            echo "run_name: ${RUN_NAME}"
            echo "log_dir: ${LOG_DIR}"
            echo "planned_cases: ${TOTAL_CASES}"
            echo "pass: ${PASS_COUNT}"
            echo "oom: ${OOM_COUNT}"
            echo "fail: ${FAIL_COUNT}"
            echo "skip: ${SKIP_COUNT}"
            echo "summary_tsv: ${SUMMARY_TSV}"
        } > "$SUMMARY_TXT"

        echo "Completed sweep."
        echo "Pass: ${PASS_COUNT}"
        echo "OOM:  ${OOM_COUNT}"
        echo "Fail: ${FAIL_COUNT}"
        echo "Skip: ${SKIP_COUNT}"
        echo "Summary: ${SUMMARY_TSV}"
    fi
}

case "$RUN_MODE" in
    local)
        run_local_sweep
        ;;
    *)
        echo "Unsupported --run-mode: ${RUN_MODE} (expected: local)" >&2
        exit 1
        ;;
esac
