# This file is copied/adapted from infllmv2_cuda_impl (https://github.com/OpenBMB/infllmv2_cuda_impl).
# Copyright (c) OpenBMB.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__version__ = "0.128"

from .blockmask_to_uint64 import blockmask_to_uint64
from .topk_to_uint64 import topk_to_uint64
from .uint64_to_bool import uint64_to_bool
from .max_pooling_1d import max_pooling_1d, max_pooling_1d_varlen
from .infllmv2_sparse_attention import (
    infllmv2_attn_varlen_func,
    infllmv2_attn_stage1,
    infllmv2_attn_stage1_fast,
    infllmv2_attn_stage1_kl_teacher,
    infllmv2_attn_with_kvcache,
)
