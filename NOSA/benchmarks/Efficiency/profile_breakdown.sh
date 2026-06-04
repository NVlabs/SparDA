#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wall-time breakdown profiler launcher.
#
# Usage:
#   bash profile_breakdown.sh --model-path openbmb/NOSA-8B -B 16 -L 128K
#   bash profile_breakdown.sh --model-path openbmb/NOSA-8B --no-offload -B 4 -L 16K
#   bash profile_breakdown.sh --model-path openbmb/MiniCPM4.1-8B
#   bash profile_breakdown.sh --model-path openbmb/MiniCPM4.1-8B --sparda --indexer-path /path/to/weights
#   bash profile_breakdown.sh --model-path openbmb/MiniCPM4.1-8B --backend hf --sparda --indexer-path /path/to/weights -B 16 -L 16K
#   bash profile_breakdown.sh --nsys --model-path openbmb/NOSA-8B -B 16 -L 128K
#
# All arguments are forwarded to profile_breakdown.py.

set -euo pipefail

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_SCHEDULE=static
export OMP_WAIT_POLICY=PASSIVE
export KMP_AFFINITY=granularity=fine,compact,1,0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Root inside shared containers often defaults to /root, which can be tiny or
# unwritable for JIT/model caches. Redirect HOME into the repo-adjacent cache.
if [ "$(id -u)" = "0" ] && { [ -z "${HOME:-}" ] || [ "${HOME}" = "/root" ] || [ "${HOME#/root/}" != "${HOME}" ]; }; then
    _cache_base="${SPARDA_ROOT_CACHE_BASE:-$(dirname "${REPO_ROOT}")/cache}"
    export HOME="${_cache_base}/root_home"
    mkdir -p "${HOME}" || true
fi

# Keep runtime/compiler/model caches away from quota-limited HOME by default.
if [ -z "${TRITON_CACHE_DIR:-}" ] || [ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ] || [ -z "${XDG_CACHE_HOME:-}" ] || [ -z "${MPLCONFIGDIR:-}" ] || [ -z "${HF_MODULES_CACHE:-}" ] || [ -z "${HF_HOME:-}" ] || [ -z "${HF_HUB_CACHE:-}" ] || [ -z "${TRANSFORMERS_CACHE:-}" ] || [ -z "${HF_DATASETS_CACHE:-}" ]; then
    _cache_root="${TMPDIR:-${REPO_ROOT}/.cache}/nosa-cache-$(id -u)"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_cache_root}/triton}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${_cache_root}/torchinductor}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_cache_root}/xdg_cache}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${_cache_root}/hf_modules}"
    export HF_HOME="${HF_HOME:-${_cache_root}/hf_home}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

    mkdir -p \
        "${TRITON_CACHE_DIR}" \
        "${TORCHINDUCTOR_CACHE_DIR}" \
        "${XDG_CACHE_HOME}" \
        "${MPLCONFIGDIR}" \
        "${HF_MODULES_CACHE}" \
        "${HF_HOME}" \
        "${HF_HUB_CACHE}" \
        "${TRANSFORMERS_CACHE}" \
        "${HF_DATASETS_CACHE}" || true
fi

# Pin JIT builds to the current GPU arch. This avoids invalid base-image defaults
# (e.g. 12.0+PTX on A100) and keeps extension builds targeted to the active node.
_arch=$(python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || true)
if [ -n "$_arch" ] && [ "${SPARDA_KEEP_TORCH_CUDA_ARCH_LIST:-0}" != "1" ]; then
    case ";${TORCH_CUDA_ARCH_LIST:-};" in
        *";${_arch};"*)
            if printf '%s' "${TORCH_CUDA_ARCH_LIST}" | grep -Eq '(^|[ ;])12\.0(\+PTX)?([ ;]|$)'; then
                export TORCH_CUDA_ARCH_LIST="$_arch"
            fi
            ;;
        *)
            export TORCH_CUDA_ARCH_LIST="$_arch"
            ;;
    esac
fi

if [ -n "${SPARDA_TASKSET_CPUS:-}" ]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
    taskset -c "${SPARDA_TASKSET_CPUS}" \
    python -B "${SCRIPT_DIR}/profile_breakdown.py" "$@"
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
    python -B "${SCRIPT_DIR}/profile_breakdown.py" "$@"
fi
