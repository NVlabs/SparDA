# This file is copied/adapted from infllmv2_cuda_impl (https://github.com/OpenBMB/infllmv2_cuda_impl).
# Copyright (c) OpenBMB.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from infllm_v2 import infllmv2_attn_stage1_kl_teacher


def _reference_stage1_kl_teacher(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal_stride: int,
    query_offset: int,
    causal: bool,
) -> torch.Tensor:
    q_len, nheads_q, head_dim = q.shape
    k_len, nheads_k, _ = k.shape
    group_size = nheads_q // nheads_k

    q_grouped = q.float().view(q_len, nheads_k, group_size, head_dim).permute(1, 2, 0, 3).contiguous()
    k_t = k.float().permute(1, 2, 0).unsqueeze(1)
    scores = torch.matmul(q_grouped, k_t) / math.sqrt(head_dim)

    if causal:
        q_positions = torch.arange(q_len, device=q.device, dtype=torch.int64) + query_offset
        k_positions = torch.arange(k_len, device=q.device, dtype=torch.int64) * causal_stride
        causal_mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        scores = scores.masked_fill(causal_mask.view(1, 1, q_len, k_len), float("-inf"))

    probs = torch.softmax(scores, dim=-1).sum(dim=1)
    probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
    return probs.to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("causal_stride", [1, 2])
def test_stage1_kl_teacher_matches_reference(causal_stride: int):
    torch.manual_seed(0)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_len = 12
    k_len = 19
    query_offset = 7
    num_q_heads = 32
    num_kv_heads = 2
    head_dim = 128

    q = torch.randn(q_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(k_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, k_len], dtype=torch.int32, device=device)

    actual = infllmv2_attn_stage1_kl_teacher(
        q,
        k,
        k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
        causal_stride=causal_stride,
        query_offset=query_offset,
        causal=True,
    )
    actual = actual[:, :q_len, :k_len]
    expected = _reference_stage1_kl_teacher(
        q,
        k,
        causal_stride=causal_stride,
        query_offset=query_offset,
        causal=True,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
