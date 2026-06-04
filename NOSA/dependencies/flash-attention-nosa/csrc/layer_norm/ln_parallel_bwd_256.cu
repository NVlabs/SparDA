// This file is copied/adapted from FlashAttention (https://github.com/Dao-AILab/flash-attention).
// Copyright (c) 2022-2024 Tri Dao and contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include "ln_parallel_residual_bwd_kernels.cuh"

// Create backward launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_PARALLEL_BWD_LAUNCHER(  256, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_PARALLEL_BWD_LAUNCHER(  256, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);
