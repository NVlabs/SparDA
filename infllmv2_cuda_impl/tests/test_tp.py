# This file is copied/adapted from infllmv2_cuda_impl (https://github.com/OpenBMB/infllmv2_cuda_impl).
# Copyright (c) OpenBMB.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch


TENSOR_NAMES = [
    "q",
    "k",
    "v",
    "bwd_blockmask_uint64",
    "fwd_blockmask_bool",
    "out",
    "softmax_lse",
    "dout",
    "dq",
    "dk",
    "dv",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "head_mask_type",
    "streaming_info",
    "ctx.max_seqlen_k_",
    "ctx.n_block_dim",
    "ctx.m_block_dim",
    "ctx.window_size_left",
    "ctx.window_size_right",
    "ctx.p_dropout",
    "ctx.softmax_scale",
    "ctx.is_causal",
    "ctx.exact_streaming",
    "ctx.deterministic",
    "topk_attn_output",
    "compressed_attn_output",
    "gated",
    "query_layer",
    "key_layer",
    "value_layer",
]


def _artifact_dir(env_var):
    value = os.environ.get(env_var)
    if not value:
        pytest.skip(
            "Set INFLLMV2_TP1_DIR, INFLLMV2_TP2_RANK0_DIR, and "
            "INFLLMV2_TP2_RANK1_DIR to run tensor-parallel artifact checks."
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        pytest.skip(f"{env_var} does not point to an existing directory: {path}")
    return path


def _load_artifact(directory, name):
    artifact_path = directory / f"{name}.pt"
    if not artifact_path.is_file():
        pytest.skip(f"Missing tensor-parallel artifact: {artifact_path}")
    return torch.load(artifact_path, map_location="cpu")


def test_tensor_parallel_artifacts_match_single_rank_reference():
    tp1_path = _artifact_dir("INFLLMV2_TP1_DIR")
    tp2_rank0_path = _artifact_dir("INFLLMV2_TP2_RANK0_DIR")
    tp2_rank1_path = _artifact_dir("INFLLMV2_TP2_RANK1_DIR")

    for tensor_name in TENSOR_NAMES:
        tp1_tensor = _load_artifact(tp1_path, tensor_name)
        tp2_rank0_tensor = _load_artifact(tp2_rank0_path, tensor_name)
        tp2_rank1_tensor = _load_artifact(tp2_rank1_path, tensor_name)

        if not isinstance(tp1_tensor, torch.Tensor):
            assert tp1_tensor == tp2_rank0_tensor
            assert tp1_tensor == tp2_rank1_tensor
            continue

        if tensor_name in ["topk_idx", "bwd_blockmask_uint64", "fwd_blockmask_bool"]:
            tp2_tensor = torch.cat((tp2_rank0_tensor, tp2_rank1_tensor), dim=0)
        elif tp1_tensor.shape == tp2_rank0_tensor.shape:
            tp2_tensor = tp2_rank0_tensor
        elif tensor_name in ["head_mask_type", "streaming_info"]:
            tp2_tensor = torch.cat((tp2_rank0_tensor, tp2_rank1_tensor), dim=0)
        else:
            tp2_tensor = torch.cat((tp2_rank0_tensor, tp2_rank1_tensor), dim=-2)

        if tp1_tensor.dtype == torch.bool:
            assert torch.equal(tp1_tensor, tp2_tensor), tensor_name
        else:
            tp1_tensor = torch.nan_to_num(tp1_tensor, nan=0.0)
            tp2_tensor = torch.nan_to_num(tp2_tensor, nan=0.0)
            assert torch.allclose(tp1_tensor, tp2_tensor, rtol=1e-3, atol=1e-3), tensor_name
