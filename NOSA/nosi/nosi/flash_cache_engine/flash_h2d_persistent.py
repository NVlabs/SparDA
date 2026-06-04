# This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
# Copyright (c) 2026 THUNLP.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Persistent Triton H2D kernel for decoupled sparse attention prefetch.

Used on the prefetch stream only. Copies K, V (and optionally CIS bias) blocks
from CPU pinned memory to GPU, guided by a load_ids mask.

Key design decisions:
  - total_tasks is a runtime value (not constexpr) to avoid recompilation
  - while-loop instead of for-range to avoid static unrolling
  - num_ctas is configurable via grid[0] for SM-friendly overlap
  - HAS_BIAS constexpr: True for NOSA (copies CIS bias), False for InfLLMv2
"""

import os
import torch
import triton
import triton.language as tl


@triton.jit
def flash_h2d_kv_persistent_kernel(
    k_gpu_ptr, k_cpu_ptr,
    v_gpu_ptr, v_cpu_ptr,
    bias_gpu_ptr, bias_src_ptr,
    load_ids_ptr,
    stride_b_cpu, stride_s_cpu, stride_h_cpu, stride_d_cpu,
    stride_b_gpu, stride_s_gpu, stride_h_gpu, stride_d_gpu,
    stride_bias_b, stride_bias_s, stride_bias_h,
    stride_biassrc_b, stride_biassrc_s, stride_biassrc_h,
    S_GPU, S_CPU,
    B, H, M,
    total_tasks,                    # runtime value: B * H * M
    BLOCK_SIZE: tl.constexpr,       # 64
    D: tl.constexpr,               # 128
    HAS_BIAS: tl.constexpr,        # True for NOSA, False for InfLLMv2
):
    pid = tl.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0).to(tl.int64)
    task_id = pid

    while task_id < total_tasks:
        m = task_id % M
        bh = task_id // M
        h = bh % H
        b = bh // H

        cpu_block_id = tl.load(load_ids_ptr + h * B * M + b * M + m).to(tl.int64)

        if cpu_block_id >= 0:
            offset_s = tl.arange(0, BLOCK_SIZE)[:, None].to(tl.int64)
            offset_d = tl.arange(0, D)[None, :].to(tl.int64)

            src_s = cpu_block_id * BLOCK_SIZE + offset_s
            dst_s = m * BLOCK_SIZE + offset_s
            mask_src = src_s < S_CPU
            mask_dst = dst_s < S_GPU

            k_src = k_cpu_ptr + b * stride_b_cpu + src_s * stride_s_cpu + h * stride_h_cpu + offset_d * stride_d_cpu
            k_dst = k_gpu_ptr + b * stride_b_gpu + dst_s * stride_s_gpu + h * stride_h_gpu + offset_d * stride_d_gpu
            tl.store(k_dst, tl.load(k_src, mask=mask_src, other=0), mask=mask_dst)

            v_src = v_cpu_ptr + b * stride_b_cpu + src_s * stride_s_cpu + h * stride_h_cpu + offset_d * stride_d_cpu
            v_dst = v_gpu_ptr + b * stride_b_gpu + dst_s * stride_s_gpu + h * stride_h_gpu + offset_d * stride_d_gpu
            tl.store(v_dst, tl.load(v_src, mask=mask_src, other=0), mask=mask_dst)

            if HAS_BIAS:
                bias_s = tl.arange(0, BLOCK_SIZE).to(tl.int64)
                bias_src_s = cpu_block_id * BLOCK_SIZE + bias_s
                bias_dst_s = m * BLOCK_SIZE + bias_s
                b_src = bias_src_ptr + b * stride_biassrc_b + bias_src_s * stride_biassrc_s + h * stride_biassrc_h
                b_dst = bias_gpu_ptr + b * stride_bias_b + bias_dst_s * stride_bias_s + h * stride_bias_h
                tl.store(b_dst, tl.load(b_src, mask=bias_src_s < S_CPU, other=0),
                         mask=bias_dst_s < S_GPU)

        task_id += num_ctas


_DEFAULT_CTAS = int(os.environ.get('PREFETCH_CTAS', '16'))
_DEFAULT_WARPS = int(os.environ.get('PREFETCH_WARPS', '8'))


def flash_h2d_kv_persistent(
    k_gpu,          # (B, S_GPU, H, D)
    k_cpu,          # (B, S_CPU, H, D)
    v_gpu,          # (B, S_GPU, H, D)
    v_cpu,          # (B, S_CPU, H, D)
    load_ids,       # (H, B, M) int64, -1 means skip
    bias_gpu=None,  # (B, S_GPU, H) -- NOSA only
    bias_src=None,  # (B, S_CPU, H) -- NOSA only (total_cis)
    block_size: int = 64,
    num_ctas: int = None,
    num_warps: int = None,
):
    """
    Persistent H2D kernel for decoupled sparse attention prefetch.

    Copies K and V blocks (and optionally CIS bias) from CPU pinned memory
    to GPU, guided by load_ids mask. Designed to run on the prefetch stream
    with a small number of CTAs to avoid SM contention with compute kernels.
    """
    if num_ctas is None:
        num_ctas = _DEFAULT_CTAS
    if num_warps is None:
        num_warps = _DEFAULT_WARPS
    num_ctas = max(1, int(num_ctas))
    num_warps = max(1, int(num_warps))

    B, S_GPU, H, D = k_gpu.shape
    _, S_CPU, _, _ = k_cpu.shape
    H2, B2, M = load_ids.shape
    assert H == H2 and B == B2

    has_bias = bias_gpu is not None and bias_src is not None
    total_tasks = B * H * M

    sb_cpu, ss_cpu, sh_cpu, sd_cpu = k_cpu.stride()
    sb_gpu, ss_gpu, sh_gpu, sd_gpu = k_gpu.stride()

    if has_bias:
        sbb, sbs, sbh = bias_gpu.stride()
        sbsb, sbss, sbsh = bias_src.stride()
    else:
        sbb = sbs = sbh = 0
        sbsb = sbss = sbsh = 0
        bias_gpu = k_gpu
        bias_src = k_cpu

    grid = (num_ctas,)

    flash_h2d_kv_persistent_kernel[grid](
        k_gpu, k_cpu,
        v_gpu, v_cpu,
        bias_gpu, bias_src,
        load_ids,
        sb_cpu, ss_cpu, sh_cpu, sd_cpu,
        sb_gpu, ss_gpu, sh_gpu, sd_gpu,
        sbb, sbs, sbh,
        sbsb, sbss, sbsh,
        S_GPU, S_CPU,
        B, H, M,
        total_tasks,
        block_size, D,
        has_bias,
        num_warps=num_warps,
    )
