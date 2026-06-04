# This file is copied/adapted from infllmv2_cuda_impl (https://github.com/OpenBMB/infllmv2_cuda_impl).
# Copyright (c) OpenBMB.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from infllm_v2 import infllmv2_attn_stage1


def _artifact_dir():
    value = os.environ.get("INFLLMV2_STAGE1_CP_LOAD_DIR")
    if not value:
        pytest.skip("Set INFLLMV2_STAGE1_CP_LOAD_DIR to run this optional artifact-based test.")
    path = Path(value).expanduser()
    if not path.is_dir():
        pytest.skip(f"INFLLMV2_STAGE1_CP_LOAD_DIR does not point to an existing directory: {path}")
    return path


def test_stage1_cp_load(causal=True):
    # assert causal, 'causal=False的情况这组数据不能对拍，因为k不一样。实际上causal=False时已经对拍过了'
    artifact_dir = _artifact_dir()
    
    # load configs
    device = torch.device("cuda:0")
    window_size = 0
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    init_blocks = 1
    local_blocks = 0
    topk = 32
    total_seqlen_q = 16384
    cp_size = 2
    block_size = total_seqlen_q // (cp_size * 2)

    # cp = 2, load tensors and compute
    cp2_q = [[None] * 4 for _ in range(1)]
    cp2_k = [None] * 1
    cp2_scores = [[None] * 4 for _ in range(1)]
    for rank in range(0, 2):
        cp_group = rank // 2
        cp_rank = rank % 2
        head, tail = cp_rank + 1, cp_size * 2 - cp_rank
        collect_scores = []
        collect_q = []
        # 每个cp rank的两块需要分开计算，然后拼接起来
        for j in [head, tail]:
            cp2_q_j = torch.load(artifact_dir / f"q_2_{rank}_{j * block_size}.pt", map_location=device)
            cp2_k_j = torch.load(artifact_dir / f"k_2_{rank}_{j * block_size}.pt", map_location=device)
            cu_seqlens_q = torch.load(artifact_dir / f"cu_seqlens_q_2_{rank}_{j * block_size}.pt", map_location=device)
            cu_seqlens_k = torch.load(artifact_dir / f"cu_seqlens_k_2_{rank}_{j * block_size}.pt", map_location=device)
            max_seqlen_q = torch.load(artifact_dir / f"max_seqlen_q_2_{rank}_{j * block_size}.pt", map_location=device)
            max_seqlen_k = torch.load(artifact_dir / f"max_seqlen_k_2_{rank}_{j * block_size}.pt", map_location=device)
            print(cp2_q_j.shape, cp2_k_j.shape, cu_seqlens_q.shape, cu_seqlens_k.shape, max_seqlen_q, max_seqlen_k)
        
            # (Pdb) print(cp2_scores[0][0].shape)
            # torch.Size([2, 64, 31])
            # (Pdb) print(cp2_scores[0][1].shape)
            # torch.Size([2, 64, 63])
            # (Pdb) print(cp2_scores[0][2].shape)
            # torch.Size([2, 64, 95])
            # (Pdb) print(cp2_scores[0][3].shape)
            # torch.Size([2, 64, 127])

            cp_2_score = infllmv2_attn_stage1(
                cp2_q_j,
                cp2_k_j,
                cp2_k_j,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_v=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal,
            )
            # exit()
            collect_scores.append(cp_2_score)
            collect_q.append(cp2_q_j)
        # breakpoint()
            
        # 按顺序放回方便对拍
        cp2_scores[cp_group][head - 1] = collect_scores[0]
        cp2_scores[cp_group][tail - 1] = collect_scores[1]
        cp2_q[cp_group][head - 1] = collect_q[0]
        cp2_q[cp_group][tail - 1] = collect_q[1]
        if cp_rank == 0:
            cp2_k[cp_group] = cp2_k_j # rank0 会拿到最后一块，对应完整的k

    cp2_q[0] = torch.cat(cp2_q[0], dim=0)

    # cp = 1 tp = 2, load tensors and compute
    cp1_q = []
    cp1_k = []
    cp1_scores = []
    for rank in range(2): # 模拟在2个rank上的计算
        cp1_q.append(torch.load(artifact_dir / f"q_1_{rank}_None.pt", map_location=device))
        cp1_k.append(torch.load(artifact_dir / f"k_1_{rank}_None.pt", map_location=device))
        cu_seqlens_q = torch.load(artifact_dir / f"cu_seqlens_q_1_{rank}_None.pt", map_location=device)
        cu_seqlens_k = torch.load(artifact_dir / f"cu_seqlens_k_1_{rank}_None.pt", map_location=device)
        max_seqlen_q = torch.load(artifact_dir / f"max_seqlen_q_1_{rank}_None.pt", map_location=device)
        max_seqlen_k = torch.load(artifact_dir / f"max_seqlen_k_1_{rank}_None.pt", map_location=device)

        cp_1_score = infllmv2_attn_stage1(
            cp1_q[rank],
            cp1_k[rank],
            cp1_k[rank],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )
        cp1_scores.append(cp_1_score)
    # breakpoint()
    
    # # 拼接成完整的结果, tp=2 在 dim=1上拼接
    std_q = []
    std_k = []
    std_scores = []
    for rank in range(1): # [2, 256, 127]
        tp_q_combine = torch.cat([cp1_q[rank * 2], cp1_q[rank * 2 + 1]], dim=1)
        tp_k_combine = torch.cat([cp1_k[rank * 2], cp1_k[rank * 2 + 1]], dim=1)
        tp_scores_combine = torch.cat([cp1_scores[rank * 2], cp1_scores[rank * 2 + 1]], dim=0)
        std_q.append(tp_q_combine)
        std_k.append(tp_k_combine)
        std_scores.append(tp_scores_combine)

    # ---- Compare ----

    # 验证前序逻辑正确性，assert表明cp=1和cp=2分发和收集kv的逻辑已对拍
    assert torch.allclose(std_q[0][:block_size], cp2_q[0][:block_size]), 'q应该一致'
    assert torch.allclose(std_q[0], cp2_q[0]), 'q应该一致'
    assert torch.allclose(std_k[0], cp2_k[0]), 'k应该一致'

    breakpoint()
    max_diff_1 = torch.max(torch.abs(std_scores[0][:, :block_size, :cp2_scores[0][0].shape[-1]] - cp2_scores[0][0]))
    max_diff_2 = torch.max(torch.abs(std_scores[0][:, block_size:2*block_size, :cp2_scores[0][1].shape[-1]] - cp2_scores[0][1]))
    max_diff_3 = torch.max(torch.abs(std_scores[0][:, 2*block_size:3*block_size, :cp2_scores[0][2].shape[-1]] - cp2_scores[0][2]))
    max_diff_4 = torch.max(torch.abs(std_scores[0][:, 3*block_size:4*block_size, :cp2_scores[0][3].shape[-1]] - cp2_scores[0][3]))
    print(f'max_diff_1 = {max_diff_1}, max_diff_2 = {max_diff_2}, max_diff_3 = {max_diff_3}, max_diff_4 = {max_diff_4}')
    breakpoint()
    max_diff_1 = torch.max(torch.abs(std_scores[0][0, :block_size, :cp2_scores[0][0].shape[-1]] - cp2_scores[0][0][0]))
    max_diff_2 = torch.max(torch.abs(std_scores[0][0, block_size:2*block_size, :cp2_scores[0][1].shape[-1]] - cp2_scores[0][1][0]))
    max_diff_3 = torch.max(torch.abs(std_scores[0][0, 2*block_size:3*block_size, :cp2_scores[0][2].shape[-1]] - cp2_scores[0][2][0]))
    max_diff_4 = torch.max(torch.abs(std_scores[0][0, 3*block_size:4*block_size, :cp2_scores[0][3].shape[-1]] - cp2_scores[0][3][0]))
    print(f'max_diff_1_head_0 = {max_diff_1}, max_diff_2_head_0 = {max_diff_2}, max_diff_3_head_0 = {max_diff_3}, max_diff_4_head_0 = {max_diff_4}')
    breakpoint()
    max_diff_1 = torch.max(torch.abs(std_scores[0][1, :block_size, :cp2_scores[0][0].shape[-1]] - cp2_scores[0][0][1]))
    max_diff_2 = torch.max(torch.abs(std_scores[0][1, block_size:2*block_size, :cp2_scores[0][1].shape[-1]] - cp2_scores[0][1][1]))
    max_diff_3 = torch.max(torch.abs(std_scores[0][1, 2*block_size:3*block_size, :cp2_scores[0][2].shape[-1]] - cp2_scores[0][2][1]))
    max_diff_4 = torch.max(torch.abs(std_scores[0][1, 3*block_size:4*block_size, :cp2_scores[0][3].shape[-1]] - cp2_scores[0][3][1]))
    print(f'max_diff_1_head_1 = {max_diff_1}, max_diff_2_head_1 = {max_diff_2}, max_diff_3_head_1 = {max_diff_3}, max_diff_4_head_1 = {max_diff_4}')
    breakpoint()
    sum_diff_1 = torch.sum(torch.abs(std_scores[0][:, :block_size, :cp2_scores[0][0].shape[-1]] - cp2_scores[0][0]))
    sum_diff_2 = torch.sum(torch.abs(std_scores[0][:, block_size:2*block_size, :cp2_scores[0][1].shape[-1]] - cp2_scores[0][1]))
    sum_diff_3 = torch.sum(torch.abs(std_scores[0][:, 2*block_size:3*block_size, :cp2_scores[0][2].shape[-1]] - cp2_scores[0][2]))
    sum_diff_4 = torch.sum(torch.abs(std_scores[0][:, 3*block_size:4*block_size, :cp2_scores[0][3].shape[-1]] - cp2_scores[0][3]))
    print(f'sum_diff_1 = {sum_diff_1}, sum_diff_2 = {sum_diff_2}, sum_diff_3 = {sum_diff_3}, sum_diff_4 = {sum_diff_4}')
    breakpoint()
    assert torch.allclose(std_scores[0][:, :block_size, :cp2_scores[0][0].shape[-1]], cp2_scores[0][0]), '第一组的score应该一致'
    breakpoint()
    assert torch.allclose(std_scores[0][:, block_size:2*block_size, :cp2_scores[0][1].shape[-1]], cp2_scores[0][1]), '第二组的score应该一致'
    breakpoint()
    assert torch.allclose(std_scores[0][:, 2*block_size:3*block_size, :cp2_scores[0][2].shape[-1]], cp2_scores[0][2]), '第三组的score应该一致'
    assert torch.allclose(std_scores[0][:, 3*block_size:4*block_size, :cp2_scores[0][3].shape[-1]], cp2_scores[0][3]), '第四组的score应该一致'
    print(std_scores[0][:, 3*block_size:4*block_size, :cp2_scores[0][3].shape[-1]][0, :, :])
    print(cp2_scores[0][3][0, :, :])


if __name__ == "__main__":
    test_stage1_cp_load(True)
