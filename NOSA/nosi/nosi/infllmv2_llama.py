# This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
# Copyright (c) 2026 THUNLP.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import math
import os
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from flashinfer.norm import rmsnorm
from flashinfer.activation import silu_and_mul
from tqdm import tqdm
import torch.nn.functional as F
from functools import lru_cache
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from infllm_v2 import (
    infllmv2_attn_stage1,
    infllmv2_attn_stage1_fast,
    infllmv2_attn_varlen_func,
    max_pooling_1d_varlen
)
import time
import torch.cuda.nvtx as nvtx

from .cache_engine import (
    InfLLMv2Cache,
    diff as _diff_offload_module,
    decode_update_fastpath as _decode_update_fastpath_module,
)
from .cache_engine_gpu import InfLLMv2Cache as InfLLMv2CacheNoOffload
from .max_pooling_fused import nosa_pooling
from .nosa_linear import nosa_linear
from flash_attn import flash_attn_with_kvcache

from .infinigen_controllers import (
    compute_skewing_matrix,
    compute_partial_weight_indices,
    infinigen_token_scores_nosi,
    infinigen_build_partial_cache,
    infinigen_update_partial_cache,
    get_precompute_a_partial,
    InfiniGenGatherWorker,
)


def _call_infllmv2_stage1_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
):
    """Call stage1 against either the repo build or the legacy NOSA package."""
    try:
        return infllmv2_attn_stage1(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )
    except TypeError as exc:
        if "cu_seqlens_v" not in str(exc):
            raise
        return infllmv2_attn_stage1(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )


def _load_torch_checkpoint(path: str):
    """Load a trusted local checkpoint across PyTorch 2.5/2.6 defaults."""
    try:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _layer_norm(hidden_states, eps, w):
    return rmsnorm(hidden_states.view(-1, hidden_states.size(-1)), w, eps).view_as(hidden_states)


def layer_norm(
    hidden_states: torch.Tensor,
    eps: float,
    w: torch.Tensor,
):
    return rmsnorm(hidden_states.view(-1, hidden_states.size(-1)), w, eps).view_as(hidden_states)


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require Sparse attention, with stride support.

    Args:
        cu_seqlen (torch.Tensor): Cumulative sequence lengths for each sample.
        chunk_size (int): Chunk size used for Sparse attention.
        kernel_stride (int): Stride size when sliding over the sequence.

    Returns:
        filtered_indices (torch.Tensor): Indices used to directly index into the key/value tensors.
        cu_seqlens_compressed (torch.Tensor): Cumulative sequence lengths after compression.
    """
    # 1. Compute the length of each sequence
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 2. Compute the start positions of chunks for each sequence (with stride)
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device)
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]  # [batch_size, max_num_chunks_per_seq]

    # 3. Filter out chunks that exceed sequence length or are smaller than the full chunk size
    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None]))

    # 4. Filter valid chunk start positions using the valid_chunk_mask
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks]
    del chunk_start_in_seq
    # 5. Generate filtered_indices
    chunk_indices = torch.arange(
        0, chunk_size, device=cu_seqlen.device
    )[None, :]  # [1, chunk_size]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, chunk_size]
    filtered_indices = filtered_indices.view(-1)  # Flatten to 1D indices

    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)  # Number of valid chunks per batch
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, chunk_indices
    return filtered_indices, cu_seqlens_compressed

def compress_k(k: torch.Tensor, cu_seqlens, head_num_k, head_dim, kernel_size=32, kernel_stride=16):
    filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
        cu_seqlens, kernel_size, kernel_stride
    )

    # Extract filtered key vectors
    filtered_k = k.index_select(0, filtered_k_indices.view(-1))

    # split
    filtered_k = filtered_k.view(filtered_k.shape[0] // kernel_size, kernel_size, head_num_k, head_dim)  # [l, block_size,h,d]

    compressed_k = filtered_k.mean(dim=1)
    return compressed_k, cu_seqlens_compressed


_DECOUPLED_STAGE1_OUT_SUPPORT = None


def decoupled_stage1(q_future, compressed_k_4d, out=None):
    """Compute Q-K scores for decoupled block selection using simple batched GEMM.
    
    Args:
        q_future: (B, H, D) -- 2 KV heads, after RoPE
        compressed_k_4d: (B, C, H, D) -- compress_k_cache_varlen (un-flattened)
    
    Returns:
        score: (H, B, C) -- per-head, per-batch, per-compressed-block scores
    """
    B, C, H, D = compressed_k_4d.shape
    q = q_future.permute(1, 0, 2).reshape(H * B, 1, D)            # (H*B, 1, D)
    k = compressed_k_4d.permute(2, 0, 1, 3).reshape(H * B, C, D)  # (H*B, C, D)
    k_t = k.transpose(1, 2)

    if out is None:
        score = torch.bmm(q, k_t)                                  # (H*B, 1, C)
        return score.reshape(H, B, C)                              # (H, B, C)

    # Direct-write path: avoids an extra per-step copy before pooling.
    global _DECOUPLED_STAGE1_OUT_SUPPORT
    out_3d = out
    out_2d = out_3d.reshape(H * B, 1, C)
    if _DECOUPLED_STAGE1_OUT_SUPPORT is None:
        try:
            torch.bmm(q, k_t, out=out_2d)
            _DECOUPLED_STAGE1_OUT_SUPPORT = True
            return out_3d
        except RuntimeError:
            _DECOUPLED_STAGE1_OUT_SUPPORT = False

    if _DECOUPLED_STAGE1_OUT_SUPPORT:
        torch.bmm(q, k_t, out=out_2d)
        return out_3d

    # Fallback for environments where bmm(out=...) cannot target this stride.
    score = torch.bmm(q, k_t).reshape(H, B, C)
    out_3d.copy_(score)
    return out_3d


def decoupled_stage1_prefill(q_pred, compressed_k, cu_seqlens_q, cu_seqlens_k, kernel_stride):
    """Compute Q-K scores for decoupled block selection during prefill.

    Matches HF ``compressed_attention`` q_pred path (per-batch BMM with causal mask).

    Args:
        q_pred: (total_q, num_kv_heads, head_dim) — q_future or q_curr after RoPE
        compressed_k: (total_k, num_kv_heads, head_dim) — varlen compressed keys
        cu_seqlens_q: (batch_size + 1,)
        cu_seqlens_k: (batch_size + 1,)
        kernel_stride: int — stride used during key compression

    Returns:
        score: (num_kv_heads, total_q, max_k) — padded with -inf
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    num_kv_heads = q_pred.shape[1]
    total_q = q_pred.shape[0]
    k_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_k = k_seqlens.max().item()

    score = torch.full((num_kv_heads, total_q, max_k), float('-inf'),
                       device=q_pred.device, dtype=q_pred.dtype)
    for b in range(batch_size):
        q_s, q_e = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        k_s, k_e = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
        q_len_b, k_len_b = q_e - q_s, k_e - k_s
        if q_len_b == 0 or k_len_b == 0:
            continue
        q_b = q_pred[q_s:q_e].permute(1, 0, 2)      # [H, q_len, D]
        k_b_t = compressed_k[k_s:k_e].permute(1, 2, 0)  # [H, D, k_len]
        raw = torch.bmm(q_b, k_b_t)                  # [H, q_len, k_len]
        # Causal mask: compressed block at k_pos covers tokens [k_pos*stride, k_pos*stride+stride).
        # A query at q_pos cannot attend to future compressed blocks.
        q_pos = torch.arange(q_len_b, device=raw.device)
        k_pos = torch.arange(k_len_b, device=raw.device) * kernel_stride
        raw = raw.masked_fill((k_pos[None, :] > q_pos[:, None]).unsqueeze(0), float('-inf'))
        score[:, q_s:q_e, :k_len_b] = raw
    return score


class LlamaLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wqkv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_up_proj :torch.Tensor = None 
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.has_buffers = False

        # Decoupled sparse attention
        self.has_q_future = False
        self.has_q_curr = False
        self._rope_scratch_k = None

        # InfiniGen state (populated by warmup)
        self.ig_enabled = False
        self.ig_skewing_matrix = None
        self.ig_partial_indices = None
        self.ig_partial_key_cache = None
        self.ig_num_channels = 32
        self.ig_a_partial = None
        self._ig_cache_max = None
        self._rope_extra_buf = None

    def init_parameters(self, hf_layer, num_heads, num_key_value_heads, head_dim,
                        skip_indexer=False,
                        residual_scale=1.0,
                        kernel_size=32, kernel_stride=16, block_size=64,
                        topk=64, init_blocks=1, local_blocks=16):
        self.residual_scale = residual_scale
        # Sparse attention parameters (from model config or defaults)
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size = block_size
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.q_size = hf_layer.self_attn.q_proj.weight.shape[0]
        self.kv_size = hf_layer.self_attn.k_proj.weight.shape[0]

        # Build fused QKV(+q_future+q_curr) weight
        wqkv_parts = [
            hf_layer.self_attn.q_proj.weight.detach(),
            hf_layer.self_attn.k_proj.weight.detach(),
            hf_layer.self_attn.v_proj.weight.detach(),
        ]

        self.has_q_future = (not skip_indexer
                              and hasattr(hf_layer.self_attn, 'q_future_proj')
                              and hf_layer.self_attn.q_future_proj is not None)
        if self.has_q_future:
            wqkv_parts.append(hf_layer.self_attn.q_future_proj.weight.detach())
            self.q_future_size = hf_layer.self_attn.q_future_proj.weight.shape[0]
        else:
            self.q_future_size = 0

        self.has_q_curr = (not skip_indexer
                             and hasattr(hf_layer.self_attn, 'q_curr_proj')
                             and hf_layer.self_attn.q_curr_proj is not None)
        if self.has_q_curr:
            wqkv_parts.append(hf_layer.self_attn.q_curr_proj.weight.detach())
            self.q_curr_size = hf_layer.self_attn.q_curr_proj.weight.shape[0]
        else:
            self.q_curr_size = 0

        self.wqkv = torch.cat(wqkv_parts, dim=0)
        self.wo = hf_layer.self_attn.o_proj.weight.detach()

        self.gate_up_proj = torch.cat((hf_layer.mlp.gate_proj.weight.detach(), hf_layer.mlp.up_proj.weight.detach()), dim=0)
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
    
    def init_gpu(self, device:str = 'cuda:0'):
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wqkv = self.wqkv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_up_proj = self.gate_up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

        # Note: _rope_scratch_k is allocated on-the-fly in decode_forward_decoupled
        # to match the actual batch size (bsz * q_len, num_kv_heads * head_dim).

    def _get_rope_scratch(self, rows: int, cols: int, device, dtype):
        buf = self._rope_scratch_k
        if (buf is None or buf.device != device or buf.dtype != dtype
                or buf.ndim != 2 or buf.shape[0] < rows or buf.shape[1] < cols):
            buf = torch.empty(rows, cols, device=device, dtype=dtype)
            self._rope_scratch_k = buf
        return buf[:rows, :cols]

    def _get_rope_extra_buf(self, rows: int, cols: int, device, dtype):
        buf = self._rope_extra_buf
        if (buf is None or buf.device != device or buf.dtype != dtype
                or buf.ndim != 2 or buf.shape[0] < rows or buf.shape[1] < cols):
            buf = torch.empty(rows, cols, device=device, dtype=dtype)
            self._rope_extra_buf = buf
        return buf[:rows, :cols]

    @torch.inference_mode()
    def dense_prefill_forward(self, hidden_states, position_ids, cos_sin_cache, cache_engine, total_bsz, current_batch_pos):
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()

        nvtx.range_push("linear")
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)

        qkv = F.linear(hidden_states, self.wqkv[:, :])
        qkv = qkv[:, :, :self.q_size + 2 * self.kv_size]
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        nvtx.range_pop()

        nvtx.range_push("rope")
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(
            position_ids.flatten(), query_states, key_states, self.head_dim, cos_sin_cache, True
        )

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        nvtx.range_pop()

        kv_bias = torch.empty((bsz, q_len, self.num_key_value_heads), dtype=key_states.dtype, device=key_states.device)
        nvtx.range_push("offloading")
        cache_engine.prefill_update_kv(
            key_states, value_states, kv_bias, self.layer_idx, current_batch_pos, total_bsz
        )
        nvtx.range_pop()

        from flash_attn import flash_attn_func
        nvtx.range_push("attention")
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        attn_output = attn_output.view(bsz, q_len, -1)
        nvtx.range_pop()

        nvtx.range_push("o linear")
        hidden_states = residual + F.linear(attn_output, self.wo) * self.residual_scale
        nvtx.range_pop()

        residual = hidden_states
        nvtx.range_push("ffn")
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        gate_up = F.linear(hidden_states, self.gate_up_proj)
        dd = gate_up.shape[-1] // 2
        out = torch.empty((*gate_up.shape[:-1], dd), dtype=gate_up.dtype, device=gate_up.device)
        silu_and_mul(gate_up, out)
        hidden_states = residual + F.linear(out, self.down_proj) * self.residual_scale
        nvtx.range_pop()
        return hidden_states

    @torch.inference_mode()
    def dense_decode_forward(self, hidden_states, position_ids, cos_sin_cache, cache_engine):
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()

        nvtx.range_push("linear")
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)
        qkv = F.linear(hidden_states, self.wqkv[:, :])
        qkv = qkv[:, :, :self.q_size + 2 * self.kv_size]
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        nvtx.range_pop()

        nvtx.range_push("rope")
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(
            position_ids.flatten(), query_states, key_states, self.head_dim, cos_sin_cache, True
        )

        query_states = query_states.view(bsz, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, self.num_key_value_heads, self.head_dim)
        nvtx.range_pop()

        nvtx.range_push("attention")
        full_k, full_v, cache_lens = cache_engine.dense_decode_update_kv(
            key_states, value_states, self.layer_idx
        )
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            full_k,
            full_v,
            cache_seqlens=cache_lens,
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        nvtx.range_pop()

        nvtx.range_push("o linear")
        hidden_states = residual + F.linear(attn_output, self.wo) * self.residual_scale
        nvtx.range_pop()

        residual = hidden_states
        nvtx.range_push("ffn")
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        gate_up = F.linear(hidden_states, self.gate_up_proj)
        dd = gate_up.shape[-1] // 2
        out = torch.empty((*gate_up.shape[:-1], dd), dtype=gate_up.dtype, device=gate_up.device)
        silu_and_mul(gate_up, out)
        hidden_states = residual + F.linear(out, self.down_proj) * self.residual_scale
        nvtx.range_pop()
        return hidden_states

    @torch.inference_mode()
    def prefill_forward(self, hidden_states, position_ids, cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, total_bsz, current_batch_pos, decoupled_mode=False, q_future_from_prev=None):
        # [ATTN] prepare
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()

        # [ATTN] prenorm
        nvtx.range_push("linear")
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)

        # [ATTN] calculate q k v (bsz, seq_len, *)
        qkvqf = F.linear(hidden_states, self.wqkv)
        qkv_end = self.q_size + 2 * self.kv_size
        qkv = qkvqf[:, :, :qkv_end]
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # [ATTN] extract q_future / q_curr for decoupled prefill
        q_future_raw = None
        q_curr_raw = None
        if decoupled_mode:
            extra = qkvqf[:, :, qkv_end:]
            if extra.shape[-1] > 0:
                offset = 0
                if self.has_q_future:
                    q_future_raw = extra[:, :, offset:offset + self.q_future_size]
                    offset += self.q_future_size
                if self.has_q_curr:
                    q_curr_raw = extra[:, :, offset:offset + self.q_curr_size]
        nvtx.range_pop()

        # [ATTN] apply rope
        nvtx.range_push("rope")
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(position_ids.flatten(), query_states, key_states, self.head_dim, cos_sin_cache, True)

        # [ATTN] apply rope to q_future / q_curr
        q_future = None
        q_curr = None
        if decoupled_mode:
            pos_flat = position_ids.flatten()
            if q_future_raw is not None:
                q_future_flat = q_future_raw.reshape(bsz * q_len, -1)
                rope_extra = self._get_rope_extra_buf(
                    q_future_flat.shape[0], q_future_flat.shape[1], q_future_flat.device, q_future_flat.dtype)
                apply_rope_with_cos_sin_cache_inplace(pos_flat, q_future_flat, rope_extra, self.head_dim, cos_sin_cache, True)
                q_future = q_future_flat.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
            if q_curr_raw is not None:
                q_curr_flat = q_curr_raw.reshape(bsz * q_len, -1)
                rope_extra = self._get_rope_extra_buf(
                    q_curr_flat.shape[0], q_curr_flat.shape[1], q_curr_flat.device, q_curr_flat.dtype)
                apply_rope_with_cos_sin_cache_inplace(pos_flat, q_curr_flat, rope_extra, self.head_dim, cos_sin_cache, True)
                q_curr = q_curr_flat.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        nvtx.range_pop()

        # [ATTN] unpad data
        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)

        # [ATTN] compress k
        nvtx.range_push("compress")
        compressed_k, cu_seqlens_comp = compress_k(key_states, cu_seqlens, self.num_key_value_heads, self.head_dim, kernel_size=self.kernel_size, kernel_stride=self.kernel_stride)
        compressed_k = compressed_k.contiguous()
        max_seqlen_comp = (cu_seqlens_comp[1:] - cu_seqlens_comp[:-1]).max().item()
        nvtx.range_pop()

        # [ATTN] update kv
        nvtx.range_push("offloading")
        cache_engine.prefill_update_kv(
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim),
            value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim),
            torch.empty((bsz, q_len, self.num_key_value_heads), dtype=key_states.dtype, device=key_states.device),
            self.layer_idx, current_batch_pos, total_bsz
        )
        nvtx.range_pop()

        # [ATTN] update compressed k
        nvtx.range_push("compress")
        cache_engine.update_compress_k_prefill(compressed_k, self.layer_idx, cu_seqlens_comp, max_seqlen=max_seqlen_comp, current_batch_pos=current_batch_pos, total_bsz=total_bsz)
        pad_key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        no_compress_k_start = max_seqlen_comp * self.kernel_stride
        no_compress_k = pad_key_states[:, no_compress_k_start:]
        cache_engine.update_no_compress_k_prefill(no_compress_k, self.layer_idx, self.kernel_size, self.kernel_stride, current_batch_pos=current_batch_pos, total_bsz=total_bsz)
        nvtx.range_pop()

        # [ATTN] stage 1: decoupled or dense
        nvtx.range_push("stage 1")
        q_pred_for_scoring = None
        if decoupled_mode:
            if self.layer_idx == 0 and q_curr is not None:
                q_pred_for_scoring = q_curr
            elif q_future_from_prev is not None:
                q_pred_for_scoring = q_future_from_prev

        if q_pred_for_scoring is not None:
            score = decoupled_stage1_prefill(
                q_pred_for_scoring, compressed_k, cu_seqlens, cu_seqlens_comp, self.kernel_stride)
        else:
            score = _call_infllmv2_stage1_compat(
                query_states,
                compressed_k,
                compressed_k,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens_comp,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen_comp,
                causal=True
            )
        nvtx.range_pop()

        # [ATTN] max pooling
        nvtx.range_push("pooling")
        cache_lens = torch.zeros(bsz, dtype=torch.int32, device=query_states.device)
        q_idx = torch.cat([
            (torch.arange(cu_seqlens[i + 1] - cu_seqlens[i], device=query_states.device) + 
                max_seqlen - (cu_seqlens[i + 1] - cu_seqlens[i])) // self.block_size
            for i in range(bsz)
        ], dim=0)
        score = score[:, :q_idx.shape[0], :]
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens,
            cu_seqlens_comp,
            cache_lens,
            max_seqlen,
            max_seqlen_comp,
            local_blocks=self.local_blocks,
            init_blocks=self.init_blocks,
            block_size=self.block_size,
            stride=self.kernel_stride)
        del score

        # generate topk index
        topk = min(self.topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)
        del block_score
        nvtx.range_pop()
        # [ATTN] sparse attention
        nvtx.range_push("stage 2")
        topk_attn_output = infllmv2_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=0.0,
            deterministic=False,
            softmax_scale=None,
            causal=True,
            return_attn_probs=False,
            topk_idx=topk_idx
        )
        attn_output = topk_attn_output
        attn_output = attn_output.view(bsz, q_len, -1)
        nvtx.range_pop()

        del query_states, key_states

        # [ATTN] wo
        nvtx.range_push("o linear")
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()

        # [FFN] 
        residual = hidden_states
        nvtx.range_push("ffn")
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()
        if decoupled_mode:
            return hidden_states, q_future
        return hidden_states


    @torch.inference_mode()
    def decode_forward_warmup(self, hidden_states, position_ids, cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx, pooling_buf, topk_buf_val, topk_buf_indices, topk_val_buf_q, topk_idx_buf_q, topk_val_buf, topk_idx_buf, mask_buf, decoupled_mode=False, q_future_from_prev=None):
        # [ATTN] prepare
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        max_pooling_buf = pooling_buf[:self.num_key_value_heads]
        max_pooling_buf_cis = pooling_buf[self.num_key_value_heads:]

        # [ATTN] prenorm
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)

        # [ATTN] calculate q k v (bsz, seq_len, *)
        nvtx.range_push("qkv_proj")
        qkvqf = F.linear(hidden_states, self.wqkv)
        qkv_end = self.q_size + 2 * self.kv_size
        qkv = qkvqf[:, :, :qkv_end]
        q_future_raw = None
        q_curr_raw = None
        if decoupled_mode:
            extra = qkvqf[:, :, qkv_end:]
            if extra.shape[-1] > 0:
                offset = 0
                if self.has_q_future:
                    q_future_raw = extra[:, :, offset:offset + self.q_future_size]
                    offset += self.q_future_size
                if self.has_q_curr:
                    q_curr_raw = extra[:, :, offset:offset + self.q_curr_size]
        nvtx.range_pop()


        nvtx.range_push("split and cis")
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        nvtx.range_pop()


        # [ATTN] apply rope
        nvtx.range_push("rope")
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(position_ids.flatten(), query_states, key_states, self.head_dim, cos_sin_cache, True)
        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)

        q_future = None
        q_curr = None
        if decoupled_mode:
            pos_flat = position_ids.flatten()
            if q_future_raw is not None:
                q_future_flat = q_future_raw.view(bsz * q_len, -1)
                rope_scratch = self._get_rope_scratch(
                    q_future_flat.shape[0], q_future_flat.shape[1], q_future_flat.device, q_future_flat.dtype)
                apply_rope_with_cos_sin_cache_inplace(pos_flat, q_future_flat, rope_scratch, self.head_dim, cos_sin_cache, True)
                q_future = q_future_flat.reshape(bsz, self.num_key_value_heads, self.head_dim)
            if q_curr_raw is not None:
                q_curr_flat = q_curr_raw.view(bsz * q_len, -1)
                rope_scratch = self._get_rope_scratch(
                    q_curr_flat.shape[0], q_curr_flat.shape[1], q_curr_flat.device, q_curr_flat.dtype)
                apply_rope_with_cos_sin_cache_inplace(pos_flat, q_curr_flat, rope_scratch, self.head_dim, cos_sin_cache, True)
                q_curr = q_curr_flat.reshape(bsz, self.num_key_value_heads, self.head_dim)
        nvtx.range_pop()
        
        # [ATTN] compress k
        nvtx.range_push("compress")
        no_compress_k = cache_engine.update_no_compress_k_decode(key_states.unsqueeze(1), self.layer_idx, self.kernel_size, self.kernel_stride)
        if no_compress_k is not None:
            new_compressed_k = no_compress_k.mean(dim=1, keepdim=True)
        else:
            new_compressed_k = None
        compressed_k, cu_seqlens_comp, max_seqlen_comp = cache_engine.update_compress_k_decode(new_compressed_k, self.layer_idx)
        compressed_k = compressed_k.contiguous().flatten(0, 1)
        nvtx.range_pop()


        # [ATTN] stage 1
        nvtx.range_push("stage 1")
        nvtx.range_push("real stage 1")

        if decoupled_mode:
            compressed_k_4d = cache_engine.layers[self.layer_idx].compress_k_cache_varlen
            _use_decoupled = False
            if self.layer_idx == 0 and q_curr is not None:
                score = decoupled_stage1(q_curr, compressed_k_4d)
                _use_decoupled = True
            elif q_future_from_prev is not None:
                score = decoupled_stage1(q_future_from_prev, compressed_k_4d)
                _use_decoupled = True

            if _use_decoupled:
                H_s, B_s, C = score.shape
                padded_C = 1 << max(0, (C - 1).bit_length()) if C > 0 else 1
                self.score_buf = torch.full(
                    (H_s, B_s, padded_C), -float('inf'),
                    dtype=score.dtype, device=score.device)
                self.compressed_cis_buf = torch.empty_like(self.score_buf)
                self.score_copy_dst = self.score_buf[:, :, :C]
                self.score_copy_dst.copy_(score)
            else:
                score = infllmv2_attn_stage1_fast(
                    query_states, compressed_k, compressed_k,
                    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens_comp,
                    max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen_comp,
                    causal=False,
                )
                self.score_buf = score
                self.compressed_cis_buf = torch.empty_like(score)
                self.score_copy_dst = self.score_buf[:, :, :max_seqlen_comp]
        else:
            score = infllmv2_attn_stage1_fast(
                query_states, compressed_k, compressed_k,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens_comp,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen_comp,
                causal=False,
            )
            self.score_buf = score
            self.compressed_cis_buf = torch.empty_like(score)
            self.score_copy_dst = self.score_buf

        nvtx.range_pop()
        nvtx.range_push("pooling")
        nvtx.range_push("kernel: max pooling")
        nvtx.range_pop()
        nvtx.range_pop()
        nvtx.range_pop()

        nosa_pooling(
            self.score_buf,
            self.compressed_cis_buf,
            max_seqlen_comp,
            max_pooling_buf,
            max_pooling_buf_cis,
            stride=self.kernel_stride,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
            init_blocks=self.init_blocks
        )

        self.after_pooling_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.after_pooling_graph):
            nosa_pooling(
                self.score_buf,
                self.compressed_cis_buf,
                max_seqlen_comp,
                max_pooling_buf,
                max_pooling_buf_cis,
                stride=self.kernel_stride,
                block_size=self.block_size,
                local_blocks=self.local_blocks,
                init_blocks=self.init_blocks
            )

            # torch.topk(
            #     max_pooling_buf, qk_select, dim=-1, sorted=False, out=(topk_val_buf_q, topk_idx_buf_q)
            # )
            # scatter_mask = mask_buf
            # scatter_mask.zero_()
            # scatter_mask.scatter_(2, topk_idx_buf_q, True)
            # max_pooling_buf_cis.masked_fill_(scatter_mask, float('inf'))

            torch.topk(
                max_pooling_buf, self.topk, dim=-1, sorted=False, out=(topk_val_buf, topk_idx_buf)
            )
        self.after_pooling_graph.replay()

        topk_idx = topk_idx_buf

        # [ATTN] offloading-update
        nvtx.range_push("offloading")
        if (
            decoupled_mode
            and self.layer_idx > 0
            and not isinstance(cache_engine, InfLLMv2CacheNoOffload)
            and not getattr(self, "decoupled_no_prefetch", False)
        ):
            # Match steady-state decode_forward_decoupled: skip_offload in
            # decode_update_kv (tail write only), then diff_offload +
            # flash_h2d_kv_persistent for H2D.  This ensures the Triton
            # H2D kernel is JIT-compiled during warmup, not during the
            # first timed decode step.
            from .flash_cache_engine.flash_h2d_persistent import flash_h2d_kv_persistent
            key_states, value_states, kv_bias, _cache_lens = cache_engine.decode_update_kv(
                key_states, value_states, None, self.layer_idx, topk_idx, skip_offload=True)
            ce = cache_engine.layers[self.layer_idx].cache_engine
            if hasattr(_diff_offload_module, "diff_offload_fused"):
                ce.record_true_miss_stats(ce._block_map, topk_idx)
                _diff_offload_module.diff_offload_fused(
                    ce._block_map, topk_idx,
                    ce._new_block_map_buf, ce._prefetch_load_mask)
                ce._block_map.copy_(ce._new_block_map_buf)
            else:
                ce._new_block_map_buf.copy_(ce._block_map)
                ce._prefetch_load_mask.fill_(-1)
                ce.record_true_miss_stats(ce._block_map, topk_idx)
                _diff_offload_module.diff_offload(
                    ce._block_map, topk_idx,
                    ce._new_block_map_buf, ce._prefetch_load_mask)
                ce._new_block_map_buf[:, :, -1].copy_(ce._block_map[:, :, -1])
                ce._prefetch_load_mask[:, :, -1].fill_(-1)
                ce._block_map.copy_(ce._new_block_map_buf)
            flash_h2d_kv_persistent(
                ce._k_gpu, ce._k_cpu,
                ce._v_gpu, ce._v_cpu,
                load_ids=ce._prefetch_load_mask)
        else:
            key_states, value_states, kv_bias, _cache_lens = cache_engine.decode_update_kv(key_states, value_states, None, self.layer_idx, topk_idx)
        nvtx.range_pop()
        # [ATTN] stage 2
        nvtx.range_push("stage 2")
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            key_states,
            value_states,
            cache_seqlens=_cache_lens,
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        nvtx.range_pop()

        # [ATTN] wo
        nvtx.range_push("o linear")
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()

        # [FFN]
        nvtx.range_push("ffn forward norm")
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        nvtx.range_pop()
        nvtx.range_push("ffn forward actual")
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()
        if decoupled_mode:
            return hidden_states, q_future
        return hidden_states


    @torch.inference_mode()
    def decode_forward(self, hidden_states, position_ids, cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx, pooling_buf, topk_buf_val, topk_buf_indices, topk_val_buf_q, topk_idx_buf_q, topk_val_buf, topk_idx_buf, mask_buf):
        # [ATTN] prepare
        nvtx.range_push("linear")
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        max_pooling_buf = pooling_buf[:self.num_key_value_heads]
        max_pooling_buf_cis = pooling_buf[self.num_key_value_heads:]

        # [ATTN] prenorm
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)

        # [ATTN] calculate q k v (bsz, seq_len, *)
        qkvqf = F.linear(hidden_states, self.wqkv)
        qkv_end = self.q_size + 2 * self.kv_size
        if self.has_q_future or self.has_q_curr:
            qkv = qkvqf[:, :, :qkv_end]
            extra = qkvqf[:, :, qkv_end:]
        else:
            qkv = qkvqf
            extra = None
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        nvtx.range_pop()

        # [ATTN] apply rope
        nvtx.range_push("rope")
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(position_ids.flatten(), query_states, key_states, self.head_dim, cos_sin_cache, True)
        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        nvtx.range_pop()

        # [ATTN] compress k
        nvtx.range_push("compress")
        no_compress_k = cache_engine.update_no_compress_k_decode(key_states.unsqueeze(1), self.layer_idx, self.kernel_size, self.kernel_stride)
        if no_compress_k is not None:
            new_compressed_k = no_compress_k.mean(dim=1, keepdim=True)
        else:
            new_compressed_k = None
        compressed_k, cu_seqlens_comp, max_seqlen_comp = cache_engine.update_compress_k_decode(new_compressed_k, self.layer_idx)
        compressed_k = compressed_k.contiguous().flatten(0, 1)
        nvtx.range_pop()


        # [ATTN] stage 1
        nvtx.range_push("stage 1")

        score = infllmv2_attn_stage1_fast(
            query_states,
            compressed_k,
            compressed_k,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_comp,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen_comp,
            causal=False,
        )
        nvtx.range_pop()

        nvtx.range_push("pooling")

        self.score_buf.copy_(score)
        self.after_pooling_graph.replay()
        topk_idx = topk_idx_buf

        nvtx.range_pop()
        # [ATTN] offloading-update
        nvtx.range_push("offloading")
        key_states, value_states, kv_bias, _cache_lens = cache_engine.decode_update_kv(key_states, value_states, None, self.layer_idx, topk_idx)
        nvtx.range_pop()

        # [ATTN] stage 2
        nvtx.range_push("stage 2")
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            key_states,
            value_states,
            cache_seqlens=_cache_lens,
        )
        nvtx.range_pop()

        attn_output = attn_output.view(bsz, q_len, -1)
        # [ATTN] wo
        nvtx.range_push("o linear")
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()

        # [FFN] 
        nvtx.range_push("ffn")
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()
        return hidden_states

    @torch.inference_mode()
    def decode_forward_decoupled(self, hidden_states, position_ids, cos_sin_cache, cu_seqlens, max_seqlen,
                                  cache_engine, cache_lens, q_idx,
                                  pooling_buf, topk_buf_val, topk_buf_indices,
                                  topk_val_buf_q, topk_idx_buf_q, topk_val_buf, topk_idx_buf, mask_buf,
                                  skip_wait=False,
                                  prefetch_done_event=None, prefetch_ready_event=None,
                                  prep_graph=None,
                                  score_buf_prep=None, block_map_buf=None,
                                  new_map_buf=None, load_mask_buf=None,
                                  pooled_q_buf=None, pooled_cis_dummy_buf=None,
                                  next_layer_cache_layer=None,
                                 score_buf_prep_input=None,
                                 prefetch_stream=None):
        """Decoupled decode path (InfLLMv2 + offload).
        
        Per-layer pipeline (no CIS):
        1. QKV+qf fused GEMM
        2. RoPE + RoPE_qf
        3. compress
        4. [layer 0 only] own stage1 with q_curr -> topk(self.topk)
        5. decode_update_kv (layer 0: normal offload/H2D path; layers 1+: skip_offload=True)
        6. bmm for next layer
        7. prep_graph.replay() -> pooling + topk(self.topk) (diff_offload runs inline after replay) for N+1
        8. copy-out + tail protect + signal
        9. WAIT (skip for layer 0)
        10. Stage2 (1x FA)
        11. O+FFN
        """
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()

        # === Step 1: QKV+qf fused GEMM ===
        nvtx.range_push("linear")
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)
        qkvqf = F.linear(hidden_states, self.wqkv)
        qkv_end = self.q_size + 2 * self.kv_size
        if self.has_q_future or self.has_q_curr:
            qkv_part = qkvqf[:, :, :qkv_end]
            extra = qkvqf[:, :, qkv_end:]
        else:
            qkv_part = qkvqf
            extra = None

        query_states, key_states, value_states = qkv_part.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Split q_future / q_curr from extra
        q_future_raw = None
        q_curr_raw = None
        if extra is not None:
            offset = 0
            if self.has_q_future:
                q_future_raw = extra[:, :, offset:offset + self.q_future_size]
                offset += self.q_future_size
            if self.has_q_curr:
                q_curr_raw = extra[:, :, offset:offset + self.q_curr_size]
        nvtx.range_pop()

        # === Step 2: RoPE ===
        nvtx.range_push("rope")
        pos_flat = position_ids.flatten()
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(pos_flat, query_states, key_states, self.head_dim, cos_sin_cache, True)

        q_future = None
        q_curr = None
        if q_future_raw is not None:
            q_future_flat = q_future_raw.view(bsz * q_len, -1)
            rope_scratch = self._get_rope_scratch(
                q_future_flat.shape[0], q_future_flat.shape[1], q_future_flat.device, q_future_flat.dtype)
            apply_rope_with_cos_sin_cache_inplace(pos_flat, q_future_flat, rope_scratch, self.head_dim, cos_sin_cache, True)
            q_future = q_future_flat.reshape(bsz, self.num_key_value_heads, self.head_dim)

        if q_curr_raw is not None:
            q_curr_flat = q_curr_raw.view(bsz * q_len, -1)
            rope_scratch = self._get_rope_scratch(
                q_curr_flat.shape[0], q_curr_flat.shape[1], q_curr_flat.device, q_curr_flat.dtype)
            apply_rope_with_cos_sin_cache_inplace(pos_flat, q_curr_flat, rope_scratch, self.head_dim, cos_sin_cache, True)
            q_curr = q_curr_flat.reshape(bsz, self.num_key_value_heads, self.head_dim)

        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        nvtx.range_pop()

        # === Step 3: compress ===
        nvtx.range_push("compress")
        no_compress_k = cache_engine.update_no_compress_k_decode(key_states.unsqueeze(1), self.layer_idx, self.kernel_size, self.kernel_stride)
        if no_compress_k is not None:
            new_compressed_k = no_compress_k.mean(dim=1, keepdim=True)
        else:
            new_compressed_k = None
        compressed_k, cu_seqlens_comp, max_seqlen_comp = cache_engine.update_compress_k_decode(new_compressed_k, self.layer_idx)
        nvtx.range_pop()

        # === L0 early PREP: move PREP(L1) before L0's own stage1 for more H2D overlap ===
        l0_early_prep_done = False
        if self.layer_idx == 0 and next_layer_cache_layer is not None and q_future_raw is not None:
            next_compressed_k_4d = cache_engine.layers[self.layer_idx + 1].compress_k_cache_varlen
            nvtx.range_push("stage 1")
            bmm_scores = decoupled_stage1(q_future, next_compressed_k_4d)
            nvtx.range_pop()

            nvtx.range_push("pooling")
            score_buf_prep_input.copy_(bmm_scores)
            ce_next = next_layer_cache_layer.cache_engine
            if prep_graph is not None:
                prep_graph.replay()
            else:
                score_buf_prep.copy_(score_buf_prep_input)
                nosa_pooling(
                    score_buf_prep, score_buf_prep,
                    max_seqlen_comp,
                    pooled_q_buf, pooled_cis_dummy_buf,
                    stride=self.kernel_stride, block_size=self.block_size, local_blocks=self.local_blocks, init_blocks=self.init_blocks)
                torch.topk(pooled_q_buf, self.topk, dim=-1, sorted=False,
                            out=(topk_val_buf, topk_idx_buf))
            nvtx.range_pop()

            nvtx.range_push("kv_prep")
            if hasattr(_diff_offload_module, "diff_offload_fused"):
                ce_next.record_true_miss_stats(ce_next._block_map, topk_idx_buf)
                _diff_offload_module.diff_offload_fused(
                    ce_next._block_map,
                    topk_idx_buf,
                    ce_next._new_block_map_buf,
                    ce_next._prefetch_load_mask,
                )
                ce_next._block_map.copy_(ce_next._new_block_map_buf)
            else:
                block_map_buf.copy_(ce_next._block_map)
                new_map_buf.copy_(block_map_buf)
                load_mask_buf.fill_(-1)
                ce_next.record_true_miss_stats(block_map_buf, topk_idx_buf)
                _diff_offload_module.diff_offload(block_map_buf, topk_idx_buf, new_map_buf, load_mask_buf)
                ce_next._block_map.copy_(new_map_buf)
                ce_next._prefetch_load_mask.copy_(load_mask_buf)
                ce_next._block_map[:, :, -1].copy_(block_map_buf[:, :, -1])
                ce_next._prefetch_load_mask[:, :, -1].fill_(-1)
            nvtx.range_pop()

            if prefetch_ready_event is not None:
                prefetch_ready_event.record()
            l0_early_prep_done = True

        # === Step 4: Layer 0 own stage1 with q_curr ===
        topk_idx = None
        if self.layer_idx == 0 and q_curr_raw is not None:
            nvtx.range_push("stage 1")
            compressed_k_4d = cache_engine.layers[self.layer_idx].compress_k_cache_varlen
            own_score = decoupled_stage1(q_curr, compressed_k_4d)
            nvtx.range_pop()

            nvtx.range_push("pooling")
            self.score_copy_dst.copy_(own_score)
            # Keep L0 pooling path consistent with non-decoupled:
            # replay captured pooling+topk graph to reduce host launch overhead.
            self.after_pooling_graph.replay()
            topk_idx = topk_idx_buf
            nvtx.range_pop()

        # === Step 5: decode_update_kv ===
        is_layer0 = self.layer_idx == 0
        kv_label = "offloading_l0" if is_layer0 else "offloading_l1p"
        nvtx.range_push(kv_label)
        ce_layer = cache_engine.layers[self.layer_idx]
        ce = ce_layer.cache_engine
        use_step5_fastpath = (
            (not is_layer0)
            and (_decode_update_fastpath_module is not None)
            and (not ce.has_kv_bias)
        )
        if not use_step5_fastpath:
            key_states, value_states, kv_bias, _cache_lens = cache_engine.decode_update_kv(
                key_states, value_states, None, self.layer_idx, topk_idx,
                skip_offload=(not is_layer0))
        else:
            # Fastpath for layers 1+ (skip_offload=True): fuse cache_lens update
            # and single-token K/V tail write into one CUDA launch.
            tail_write_pos = ce._tail_block_idx_on_gpu * ce.block_size + ce._tail_block_len_on_gpu
            _decode_update_fastpath_module.decode_update_fastpath(
                ce._k_gpu,
                ce._v_gpu,
                key_states,
                value_states,
                ce._cache_lens,
                int(tail_write_pos),
            )

            # Keep Python-side scalar/cache metadata updates equivalent to
            # decode_update_no_kv_bias(skip_offload=True).
            ce.seq_length += 1
            ce_layer.seq_length += 1
            ce._tail_block_len_on_gpu += 1
            tail_full = ce._tail_block_len_on_gpu == ce.block_size
            if tail_full:
                tail_block_base_pos = ce._tail_block_idx_on_gpu * ce.block_size
                cpu_block_base_pos = ce.seq_length - ce.block_size
                ce._k_cpu[
                    :,
                    cpu_block_base_pos:cpu_block_base_pos + ce.block_size,
                    :,
                    :,
                ].copy_(
                    ce._k_gpu[
                        :,
                        tail_block_base_pos:tail_block_base_pos + ce.block_size,
                        :,
                        :,
                    ],
                    non_blocking=True,
                )
                ce._v_cpu[
                    :,
                    cpu_block_base_pos:cpu_block_base_pos + ce.block_size,
                    :,
                    :,
                ].copy_(
                    ce._v_gpu[
                        :,
                        tail_block_base_pos:tail_block_base_pos + ce.block_size,
                        :,
                        :,
                    ],
                    non_blocking=True,
                )
                ce._block_map[..., ce._tail_block_idx_on_gpu] += 1
                ce._tail_block_len_on_gpu = 0
                ce._cache_lens[...] = (ce.topk - 1) * ce.block_size

            key_states = ce._k_gpu
            value_states = ce._v_gpu
            kv_bias = ce._kv_bias_gpu
            _cache_lens = ce._cache_lens
        nvtx.range_pop()

        # === Steps 6-8: PREP for next layer (skipped for L0 -- already done above) ===
        if not l0_early_prep_done and next_layer_cache_layer is not None and q_future_raw is not None:
            next_compressed_k_4d = cache_engine.layers[self.layer_idx + 1].compress_k_cache_varlen
            nvtx.range_push("stage 1")
            bmm_scores = decoupled_stage1(q_future, next_compressed_k_4d)
            nvtx.range_pop()

            nvtx.range_push("pooling")
            # Keep external copy contiguous->contiguous; the strided copy into
            # score_buf_prep (padded-pitch view) is captured inside prep_graph.
            score_buf_prep_input.copy_(bmm_scores)
            ce_next = next_layer_cache_layer.cache_engine
            # prep_graph captures pooling + topk only; diff_offload runs
            # inline because it's a custom C++ ext not captured by CUDA graphs.
            if prep_graph is not None:
                prep_graph.replay()
            else:
                # Inline fallback: pooling + topk
                score_buf_prep.copy_(score_buf_prep_input)
                nosa_pooling(
                    score_buf_prep, score_buf_prep,
                    max_seqlen_comp,
                    pooled_q_buf, pooled_cis_dummy_buf,
                    stride=self.kernel_stride, block_size=self.block_size, local_blocks=self.local_blocks, init_blocks=self.init_blocks)
                torch.topk(pooled_q_buf, self.topk, dim=-1, sorted=False,
                            out=(topk_val_buf, topk_idx_buf))
            nvtx.range_pop()

            # diff_offload + block_map mutation for next layer
            nvtx.range_push("kv_prep")
            if hasattr(_diff_offload_module, "diff_offload_fused"):
                # For decoupled InfLLMv2, q_future topk is the final topk for next layer.
                # Record true misses here so true_miss/wasted_reload includes layers 1+.
                ce_next.record_true_miss_stats(ce_next._block_map, topk_idx_buf)
                _diff_offload_module.diff_offload_fused(
                    ce_next._block_map,
                    topk_idx_buf,
                    ce_next._new_block_map_buf,
                    ce_next._prefetch_load_mask,
                )
                ce_next._block_map.copy_(ce_next._new_block_map_buf)
            else:
                block_map_buf.copy_(ce_next._block_map)
                new_map_buf.copy_(block_map_buf)
                load_mask_buf.fill_(-1)
                # For decoupled InfLLMv2, q_future topk is the final topk for next layer.
                # Record true misses here so true_miss/wasted_reload includes layers 1+.
                ce_next.record_true_miss_stats(block_map_buf, topk_idx_buf)
                _diff_offload_module.diff_offload(block_map_buf, topk_idx_buf, new_map_buf, load_mask_buf)
                ce_next._block_map.copy_(new_map_buf)
                ce_next._prefetch_load_mask.copy_(load_mask_buf)
                ce_next._block_map[:, :, -1].copy_(block_map_buf[:, :, -1])
                ce_next._prefetch_load_mask[:, :, -1].fill_(-1)
            nvtx.range_pop()

            if prefetch_ready_event is not None:
                prefetch_ready_event.record()

        # === Step 9: WAIT ===
        if not skip_wait and prefetch_done_event is not None:
            nvtx.range_push("h2d_wait")
            torch.cuda.current_stream().wait_event(prefetch_done_event)
            nvtx.range_pop()

        # === Step 10: Stage2 (1x FA) ===
        nvtx.range_push("stage 2")
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            key_states, value_states,
            cache_seqlens=_cache_lens,
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        nvtx.range_pop()

        nvtx.range_push("o linear")
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()

        # === Step 11: FFN ===
        nvtx.range_push("ffn")
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd,))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()

        return hidden_states

    @torch.inference_mode()
    def decode_forward_decoupled_no_offload(self, hidden_states, position_ids, cos_sin_cache,
                                             cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx,
                                             pooling_buf, topk_buf_val, topk_buf_indices,
                                             topk_val_buf_q, topk_idx_buf_q, topk_val_buf, topk_idx_buf, mask_buf,
                                             q_future_from_prev=None):
        """Decoupled decode path without async prefetch staging (InfLLMv2, no CIS).
        
        Replaces expensive 32-head stage1 with 2-head decoupled_stage1 using q_future_from_prev.
        Despite the historical name, this helper is also used for offload mode
        when decoupled prefetch is disabled and the normal decode_update_kv
        correction-fetch path should run on the current layer.
        Returns (hidden_states, q_future).
        """
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()

        # Step 1: QKV + qf fused GEMM
        nvtx.range_push("linear")
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)
        qkvqf = F.linear(hidden_states, self.wqkv)
        qkv_end = self.q_size + 2 * self.kv_size
        qkv = qkvqf[:, :, :qkv_end]
        extra = qkvqf[:, :, qkv_end:]

        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        q_future_raw = None
        q_curr_raw = None
        if extra is not None and extra.shape[-1] > 0:
            offset = 0
            if self.has_q_future:
                q_future_raw = extra[:, :, offset:offset + self.q_future_size]
                offset += self.q_future_size
            if self.has_q_curr:
                q_curr_raw = extra[:, :, offset:offset + self.q_curr_size]
        nvtx.range_pop()

        # Step 2: RoPE
        nvtx.range_push("rope")
        pos_flat = position_ids.flatten()
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(pos_flat, query_states, key_states, self.head_dim, cos_sin_cache, True)

        q_future = None
        q_curr = None
        if q_future_raw is not None:
            q_future_flat = q_future_raw.view(bsz * q_len, -1)
            rope_scratch = self._get_rope_scratch(
                q_future_flat.shape[0], q_future_flat.shape[1], q_future_flat.device, q_future_flat.dtype)
            apply_rope_with_cos_sin_cache_inplace(pos_flat, q_future_flat, rope_scratch, self.head_dim, cos_sin_cache, True)
            q_future = q_future_flat.reshape(bsz, self.num_key_value_heads, self.head_dim)

        if q_curr_raw is not None:
            q_curr_flat = q_curr_raw.view(bsz * q_len, -1)
            rope_scratch = self._get_rope_scratch(
                q_curr_flat.shape[0], q_curr_flat.shape[1], q_curr_flat.device, q_curr_flat.dtype)
            apply_rope_with_cos_sin_cache_inplace(pos_flat, q_curr_flat, rope_scratch, self.head_dim, cos_sin_cache, True)
            q_curr = q_curr_flat.reshape(bsz, self.num_key_value_heads, self.head_dim)

        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        nvtx.range_pop()

        # Step 3: compress
        nvtx.range_push("compress")
        no_compress_k = cache_engine.update_no_compress_k_decode(key_states.unsqueeze(1), self.layer_idx, self.kernel_size, self.kernel_stride)
        if no_compress_k is not None:
            new_compressed_k = no_compress_k.mean(dim=1, keepdim=True)
        else:
            new_compressed_k = None
        compressed_k, cu_seqlens_comp, max_seqlen_comp = cache_engine.update_compress_k_decode(new_compressed_k, self.layer_idx)
        nvtx.range_pop()

        # Step 4: decoupled stage1
        nvtx.range_push("stage 1")
        compressed_k_4d = cache_engine.layers[self.layer_idx].compress_k_cache_varlen
        score_ready = False
        if self.layer_idx == 0 and q_curr is not None:
            decoupled_stage1(q_curr, compressed_k_4d, out=self.score_copy_dst)
            score_ready = True
            stage1_query = q_curr
            stage1_score = self.score_copy_dst
        elif q_future_from_prev is not None:
            decoupled_stage1(q_future_from_prev, compressed_k_4d, out=self.score_copy_dst)
            score_ready = True
            stage1_query = q_future_from_prev
            stage1_score = self.score_copy_dst
        else:
            compressed_k_flat = compressed_k.contiguous().flatten(0, 1)
            score = infllmv2_attn_stage1_fast(
                query_states, compressed_k_flat, compressed_k_flat,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens_comp,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen_comp, causal=False)
            score = score[:, :, :self.score_copy_dst.shape[-1]]
            stage1_query = query_states
            stage1_score = score
        nvtx.range_pop()

        # Step 5: pooling + topk (reuse existing after_pooling_graph)
        nvtx.range_push("pooling")
        if not score_ready:
            self.score_copy_dst.copy_(score)
        self.after_pooling_graph.replay()
        topk_idx = topk_idx_buf
        nvtx.range_pop()

        # Step 6: decode_update_kv (normal)
        nvtx.range_push("offloading")
        key_states, value_states, kv_bias, _cache_lens = cache_engine.decode_update_kv(
            key_states, value_states, None, self.layer_idx, topk_idx)
        nvtx.range_pop()

        # Step 7: Stage2
        nvtx.range_push("stage 2")
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            key_states, value_states,
            cache_seqlens=_cache_lens,
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        nvtx.range_pop()

        nvtx.range_push("o linear")
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()

        nvtx.range_push("ffn")
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd,))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()
        return hidden_states, q_future


    @torch.inference_mode()
    def _infinigen_prep_next_layer(self, attn_input, bsz, cos_sin_cache, position_ids,
                                     next_layer, cache_engine,
                                     score_buf_prep_input, prep_graph,
                                     topk_idx_buf, prefetch_ready_event):
        """Compute InfiniGen scoring for next layer → pooling → topk → diff_offload → signal.

        Shared by layer-0 early PREP and layers-1+ PREP.
        """
        nvtx.range_push("stage 1")
        _head_dim = next_layer.head_dim
        def _rope_q_only(q_flat, csn_cache, pos_flat):
            dummy_k = torch.zeros_like(q_flat[:, :next_layer.num_key_value_heads])
            apply_rope_with_cos_sin_cache_inplace(
                pos_flat, q_flat, dummy_k, _head_dim, csn_cache, True)

        _max_cache = next_layer.ig_partial_key_cache.shape[1]
        _C_next = cache_engine.layers[self.layer_idx + 1].cached_compressed_max_seqlen
        _actual_seq = min(_C_next * next_layer.kernel_stride, _max_cache)

        token_scores = infinigen_token_scores_nosi(
            attn_input,
            next_layer.wqkv[:next_layer.q_size],
            next_layer.ig_skewing_matrix,
            next_layer.ig_partial_indices,
            next_layer.ig_partial_key_cache[:, :_actual_seq],
            cos_sin_cache, position_ids,
            next_layer.num_heads, next_layer.num_key_value_heads, next_layer.head_dim,
            next_layer.ig_num_channels,
            apply_rope_inplace_fn=_rope_q_only,
            a_partial=getattr(next_layer, 'ig_a_partial', None),
        )

        score_pool = token_scores.view(next_layer.num_key_value_heads * bsz, 1, -1)
        block_scores = F.max_pool1d(score_pool.float(),
            kernel_size=next_layer.kernel_stride,
            stride=next_layer.kernel_stride, padding=0)
        score = block_scores.view(next_layer.num_key_value_heads, bsz, -1)
        if score.shape[-1] < _C_next:
            score = F.pad(score, (0, _C_next - score.shape[-1]), value=float('-inf'))
        elif score.shape[-1] > _C_next:
            score = score[:, :, :_C_next]
        nvtx.range_pop()

        nvtx.range_push("pooling")
        score_buf_prep_input.copy_(score)
        if prep_graph is not None:
            prep_graph.replay()
        else:
            next_layer.score_buf.copy_(score)
            next_layer.after_pooling_graph.replay()
        nvtx.range_pop()

        nvtx.range_push("kv_prep")
        ce_next = cache_engine.layers[self.layer_idx + 1].cache_engine
        # Use _prefetch_load_mask for offload (separate from main stream's _load_mask),
        # fall back to _load_mask for no-offload (no gather worker to consume it).
        load_mask_out = getattr(ce_next, '_prefetch_load_mask', None)
        if load_mask_out is None:
            load_mask_out = ce_next._load_mask
        if hasattr(_diff_offload_module, "diff_offload_fused"):
            ce_next.record_true_miss_stats(ce_next._block_map, topk_idx_buf)
            _diff_offload_module.diff_offload_fused(
                ce_next._block_map,
                topk_idx_buf,
                ce_next._new_block_map_buf,
                load_mask_out,
            )
            ce_next._block_map.copy_(ce_next._new_block_map_buf)
        else:
            ce_next.record_true_miss_stats(ce_next._block_map, topk_idx_buf)
            # Save tail slot before diff_offload overwrites it
            old_tail = ce_next._block_map[:, :, -1].clone()
            ce_next._new_block_map_buf.copy_(ce_next._block_map)
            load_mask_out.fill_(-1)
            _diff_offload_module.diff_offload(
                ce_next._block_map, topk_idx_buf,
                ce_next._new_block_map_buf, load_mask_out)
            ce_next._block_map.copy_(ce_next._new_block_map_buf)
            ce_next._block_map[:, :, -1].copy_(old_tail)
            load_mask_out[:, :, -1].fill_(-1)

        if prefetch_ready_event is not None:
            prefetch_ready_event.record()
        nvtx.range_pop()

    def decode_forward_infinigen(self, hidden_states, position_ids, cos_sin_cache,
                                  cu_seqlens, max_seqlen,
                                  cache_engine, cache_lens, q_idx,
                                  pooling_buf, topk_buf_val, topk_buf_indices,
                                  topk_val_buf_q, topk_idx_buf_q, topk_val_buf, topk_idx_buf, mask_buf,
                                  skip_wait=False,
                                  prefetch_done_event=None,
                                  prefetch_ready_event=None,
                                  next_layer=None,
                                  prep_graph=None,
                                  score_buf_prep_input=None,
                                  next_layer_cache_layer=None):
        """InfiniGen decode path with per-layer offload and prefetch.

        Pipeline (mirrors decoupled):
        1. QKV Linear + LayerNorm
        2. RoPE
        3. Compress K
        4a. [L0 only] Early PREP(L1) via infinigen scoring → diff_offload → signal
        4b. [L0 only] Own stage1 with q_current → topk
        5. decode_update_kv (L0: normal offload/H2D path; L1+: skip_offload)
        6-8. [L1+] PREP(N+1) via infinigen scoring → diff_offload → signal
        9. WAIT for H2D (skip L0)
        10. Stage 2
        11. O Linear + FFN + partial key cache update
        """
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        is_layer0 = self.layer_idx == 0

        # === Step 1: QKV Linear ===
        nvtx.range_push("linear")
        hidden_states = _layer_norm(hidden_states, self.input_layernorm_variance_epsilon,
                                     self.input_layernorm_weight)
        attn_input = hidden_states  # post-layernorm, for PREP scoring

        qkv = F.linear(hidden_states, self.wqkv)
        qkv_end = self.q_size + 2 * self.kv_size
        query_states = qkv[:, :, :self.q_size]
        key_states = qkv[:, :, self.q_size:self.q_size + self.kv_size]
        value_states = qkv[:, :, self.q_size + self.kv_size:qkv_end]
        nvtx.range_pop()

        # === Step 2: RoPE ===
        nvtx.range_push("rope")
        query_states = query_states.reshape(bsz * q_len, -1)
        key_states = key_states.reshape(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(
            position_ids.flatten(), query_states, key_states,
            self.head_dim, cos_sin_cache, True)
        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        nvtx.range_pop()

        # === Step 3: Compress K ===
        nvtx.range_push("compress")
        no_compress_k = cache_engine.update_no_compress_k_decode(
            key_states.unsqueeze(1), self.layer_idx, self.kernel_size, self.kernel_stride)
        new_compressed_k = no_compress_k.mean(dim=1, keepdim=True) if no_compress_k is not None else None
        compressed_k, cu_seqlens_comp, max_seqlen_comp = cache_engine.update_compress_k_decode(
            new_compressed_k, self.layer_idx)
        compressed_k_flat = compressed_k.contiguous().flatten(0, 1)
        nvtx.range_pop()

        original_key_states = key_states.clone()

        # === Step 4a: [L0 only] Early PREP for L1 ===
        l0_early_prep_done = False
        if (is_layer0 and next_layer is not None
                and next_layer_cache_layer is not None
                and next_layer.ig_enabled
                and next_layer.ig_partial_key_cache is not None):
            self._infinigen_prep_next_layer(
                attn_input, bsz, cos_sin_cache, position_ids,
                next_layer, cache_engine,
                score_buf_prep_input, prep_graph,
                topk_idx_buf, prefetch_ready_event)
            l0_early_prep_done = True

        # === Step 4b: [L0 only] Own stage1 with q_current ===
        topk_idx = None
        if is_layer0:
            nvtx.range_push("stage 1")
            score = infllmv2_attn_stage1_fast(
                query_states, compressed_k_flat, compressed_k_flat,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens_comp,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen_comp,
                causal=False)
            self.score_buf.copy_(score)
            nvtx.range_pop()

            nvtx.range_push("pooling")
            self.after_pooling_graph.replay()
            topk_idx = topk_idx_buf
            nvtx.range_pop()

        # === Step 5: decode_update_kv ===
        kv_label = "offloading_l0" if is_layer0 else "offloading_l1p"
        nvtx.range_push(kv_label)
        ce_layer = cache_engine.layers[self.layer_idx]
        ce = ce_layer.cache_engine
        use_step5_fastpath = (
            (not is_layer0)
            and (_decode_update_fastpath_module is not None)
            and (not ce.has_kv_bias)
        )
        if not use_step5_fastpath:
            key_states, value_states, _kv_bias, _cache_lens = cache_engine.decode_update_kv(
                key_states, value_states, None, self.layer_idx, topk_idx,
                skip_offload=(not is_layer0))
        else:
            # Fastpath for layers 1+ (skip_offload=True): fuse cache_lens update
            # and single-token K/V tail write into one CUDA launch.
            tail_write_pos = ce._tail_block_idx_on_gpu * ce.block_size + ce._tail_block_len_on_gpu
            _decode_update_fastpath_module.decode_update_fastpath(
                ce._k_gpu, ce._v_gpu, key_states, value_states,
                ce._cache_lens, int(tail_write_pos))

            ce.seq_length += 1
            ce_layer.seq_length += 1
            ce._tail_block_len_on_gpu += 1
            tail_full = ce._tail_block_len_on_gpu == ce.block_size
            if tail_full:
                tail_block_base_pos = ce._tail_block_idx_on_gpu * ce.block_size
                cpu_block_base_pos = ce.seq_length - ce.block_size
                ce._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos + ce.block_size, :, :].copy_(
                    ce._k_gpu[:, tail_block_base_pos:tail_block_base_pos + ce.block_size, :, :],
                    non_blocking=True)
                ce._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos + ce.block_size, :, :].copy_(
                    ce._v_gpu[:, tail_block_base_pos:tail_block_base_pos + ce.block_size, :, :],
                    non_blocking=True)
                ce._block_map[..., ce._tail_block_idx_on_gpu] += 1
                ce._tail_block_len_on_gpu = 0
                ce._cache_lens[...] = (ce.topk - 1) * ce.block_size

            key_states = ce._k_gpu
            value_states = ce._v_gpu
            _cache_lens = ce._cache_lens
        nvtx.range_pop()

        # === Steps 6-8: PREP for next layer (L1+, not done at L0) ===
        if (not l0_early_prep_done
                and next_layer is not None
                and next_layer_cache_layer is not None
                and next_layer.ig_enabled
                and next_layer.ig_partial_key_cache is not None):
            self._infinigen_prep_next_layer(
                attn_input, bsz, cos_sin_cache, position_ids,
                next_layer, cache_engine,
                score_buf_prep_input, prep_graph,
                topk_idx_buf, prefetch_ready_event)

        # === Step 9: WAIT for H2D ===
        if not skip_wait and prefetch_done_event is not None:
            nvtx.range_push("h2d_wait")
            torch.cuda.current_stream().wait_event(prefetch_done_event)
            nvtx.range_pop()

        # === Update partial key cache (before stage 2, after H2D wait) ===
        if self.ig_partial_key_cache is not None:
            new_k = original_key_states.view(bsz, self.num_key_value_heads, self.head_dim)
            if cache_lens is not None:
                infinigen_update_partial_cache(
                    self.ig_partial_key_cache, new_k, cache_lens,
                    self.ig_skewing_matrix, self.ig_partial_indices)
            else:
                infinigen_update_partial_cache(
                    self.ig_partial_key_cache, new_k, 0,
                    self.ig_skewing_matrix, self.ig_partial_indices)

        # === Step 10: Stage 2 ===
        nvtx.range_push("stage 2")
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            key_states, value_states,
            cache_seqlens=_cache_lens)
        nvtx.range_pop()

        attn_output = attn_output.view(bsz, q_len, -1)

        # === Step 11: O Linear + FFN ===
        nvtx.range_push("o linear")
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()

        nvtx.range_push("ffn")
        residual = hidden_states
        hidden_states = _layer_norm(hidden_states,
                                     self.post_attention_layernorm_variance_epsilon,
                                     self.post_attention_layernorm_weight)
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd,))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states * self.residual_scale
        nvtx.range_pop()

        return hidden_states


# Backward-compat alias (now in infinigen_controllers)
_InfiniGenGatherWorker = InfiniGenGatherWorker


class Llama:
    def __init__(self,
        model_name: str = "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        offload: bool = True,
        offload_layer0: bool = False,
        max_length :int = 128*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full',
        decoupled: bool = False,
        decoupled_no_prefetch: bool = False,
        indexer_path: str = None,
        long_context: bool = False,
        infinigen: bool = False,
        infinigen_num_channels: int = 32) -> None:
        

        self.device = device
        self.dtype = dtype
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False, trust_remote_code=True)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = getattr(self.config, 'head_dim', self.config.hidden_size // self.config.num_attention_heads)
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.vocab_size = self.config.vocab_size
        self.past_len = 0
        self.has_buffers = False
        self.offload = offload
        self.offload_layer0 = bool(offload_layer0)
        # InfiniGen overrides decoupled
        self.infinigen = infinigen
        self.infinigen_num_channels = infinigen_num_channels
        self._infinigen_initialized = False
        if infinigen:
            self.decoupled = False
            self.decoupled_no_prefetch = False
        else:
            self.decoupled = decoupled or getattr(self.config, 'decoupled', False)
            self.decoupled_no_prefetch = bool(decoupled_no_prefetch)
        self._decoupled_initialized = False
        self._ig_offload_initialized = False
        self._dense_decode_active = False
        self.indexer_path = indexer_path if not infinigen else None
        self.long_context = long_context

        # muP scaling (MiniCPM-style); defaults to 1.0 so Llama models are unchanged
        self.scale_emb = float(getattr(self.config, 'scale_emb', 1.0))
        scale_depth = float(getattr(self.config, 'scale_depth', 0.0))
        if scale_depth > 0:
            self.residual_scale = scale_depth / math.sqrt(self.config.num_hidden_layers)
        else:
            self.residual_scale = 1.0

        # LM head muP scaling (MiniCPM-style): logits = lm_head(h / scale)
        dim_model_base = float(getattr(self.config, 'dim_model_base', 0))
        if dim_model_base > 0:
            self.lm_head_scale = self.hidden_size / dim_model_base
        else:
            self.lm_head_scale = 1.0

        # Normalize sparse config so model-side and cache-engine settings stay in sync.
        sparse_cfg = dict(getattr(self.config, 'sparse_config', None) or {})
        block_size_cfg = int(sparse_cfg.get('block_size', 64))
        sparse_cfg['kernel_size'] = int(sparse_cfg.get('kernel_size', 32))
        sparse_cfg['kernel_stride'] = int(sparse_cfg.get('kernel_stride', 16))
        sparse_cfg['block_size'] = block_size_cfg
        sparse_cfg['init_blocks'] = 1
        sparse_cfg['window_size'] = int(32 * block_size_cfg)  # local window blocks = 32
        sparse_cfg['topk'] = 96
        sparse_cfg['use_nope'] = bool(sparse_cfg.get('use_nope', False))
        sparse_cfg['dense_len'] = int(sparse_cfg.get('dense_len', -1))
        sparse_cfg['force_dense_inference'] = bool(sparse_cfg.get('force_dense_inference', False))
        sparse_cfg['use_q_future_for_topk'] = bool(sparse_cfg.get('use_q_future_for_topk', True))
        sparse_cfg['offload_layer0'] = self.offload_layer0
        self.config.sparse_config = sparse_cfg

        self.kernel_size = int(sparse_cfg['kernel_size'])
        self.kernel_stride = int(sparse_cfg['kernel_stride'])
        self.block_size = int(sparse_cfg['block_size'])
        self.topk = int(sparse_cfg['topk'])
        self.init_blocks = int(sparse_cfg['init_blocks'])
        window_size = int(sparse_cfg['window_size'])
        self.local_blocks = window_size // self.block_size
        self.dense_len = int(sparse_cfg['dense_len'])
        self.force_dense_inference = bool(sparse_cfg['force_dense_inference'])
        self.use_nope = bool(sparse_cfg['use_nope'])
        self.use_q_future_for_topk = bool(sparse_cfg['use_q_future_for_topk'])

        self.init_parameters()
        self._dense_decode_active = False

    def init_parameters(self):
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype, trust_remote_code=True)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        # ── Build cos/sin cache ──────────────────────────────────────────
        if hasattr(hf_model.model, 'rotary_emb'):
            rotary_emb = hf_model.model.rotary_emb
        else:
            rotary_emb = hf_model.model.layers[0].self_attn.rotary_emb

        required_len = self.max_length + 1024

        if self.long_context and hasattr(rotary_emb, '_set_cos_sin_cache'):
            self._apply_minicpm_128k_longrope(rotary_emb)
            rotary_emb._set_cos_sin_cache(
                seq_len=required_len,
                device=rotary_emb.inv_freq.device,
                dtype=torch.float32,
            )
            cos_cache = rotary_emb.cos_cached[:required_len].to(device=self.device, dtype=self.dtype)
            sin_cache = rotary_emb.sin_cached[:required_len].to(device=self.device, dtype=self.dtype)
            print(f"[RoPE/LongContext] Applied official MiniCPM4.1 128K LongRoPE factors "
                  f"(len={cos_cache.shape[0]}, dim={cos_cache.shape[1]})")
        elif hasattr(rotary_emb, 'cos_cached') and rotary_emb.cos_cached is not None:
            if rotary_emb.cos_cached.shape[0] < required_len:
                rotary_emb._set_cos_sin_cache(
                    seq_len=required_len,
                    device=rotary_emb.inv_freq.device,
                    dtype=torch.float32,
                )
            cos_cache = rotary_emb.cos_cached[:required_len].to(device=self.device, dtype=self.dtype)
            sin_cache = rotary_emb.sin_cached[:required_len].to(device=self.device, dtype=self.dtype)
            rope_type = type(rotary_emb).__name__
            print(f"[RoPE] Extracted pre-computed cos/sin from {rope_type} "
                  f"(len={cos_cache.shape[0]}, dim={cos_cache.shape[1]})")
        else:
            inv_freq = rotary_emb.inv_freq.to(self.device)
            cos_cache, sin_cache = self._set_cos_sin_cache(inv_freq)
            print(f"[RoPE] Computed standard cos/sin from inv_freq (fallback)")

        half_dim = self.head_dim // 2
        self.cos_sin_cache = torch.cat((cos_cache[:, :half_dim], sin_cache[:, :half_dim]), dim=-1).to(torch.float32)
        
        del cos_cache, sin_cache

        # Load indexer weights (q_future_proj / q_curr_proj) from separate checkpoint if provided
        indexer_sd = None
        if self.indexer_path is not None:
            import os, safetensors.torch
            if os.path.isdir(self.indexer_path):
                st_files = [f for f in os.listdir(self.indexer_path) if f.endswith('.safetensors')]
                if st_files:
                    indexer_sd = {}
                    for f in st_files:
                        indexer_sd.update(safetensors.torch.load_file(os.path.join(self.indexer_path, f)))
                else:
                    pt_file = os.path.join(self.indexer_path, 'pytorch_model.bin')
                    if os.path.exists(pt_file):
                        indexer_sd = _load_torch_checkpoint(pt_file)
            else:
                if self.indexer_path.endswith('.safetensors'):
                    indexer_sd = safetensors.torch.load_file(self.indexer_path)
                else:
                    indexer_sd = _load_torch_checkpoint(self.indexer_path)
        if indexer_sd is not None:
            indexer_sd = self._normalize_indexer_state_dict(indexer_sd)
            if not indexer_sd:
                print(f"WARNING: indexer checkpoint at {self.indexer_path} did not contain a usable state_dict; "
                      f"decoupled projections will be missing.")
                indexer_sd = None

        self.layers :list[LlamaLayer] = []

        for idx, hf_layer in enumerate(tqdm(hf_model.model.layers, desc="coverting model")):
            # Inject indexer weights into the HF layer if loaded from separate checkpoint
            if indexer_sd is not None:
                self._inject_indexer_weights(hf_layer, idx, indexer_sd)

            layer = LlamaLayer(idx)
            layer.init_parameters(hf_layer=hf_layer, head_dim=self.head_dim, num_heads=self.num_heads, num_key_value_heads=self.num_key_value_heads,
                                  residual_scale=self.residual_scale,
                                  kernel_size=self.kernel_size, kernel_stride=self.kernel_stride,
                                  block_size=self.block_size, topk=self.topk,
                                  init_blocks=self.init_blocks, local_blocks=self.local_blocks,
                                  skip_indexer=self.infinigen)
            layer.init_gpu(self.device)
            layer.decoupled_no_prefetch = self.decoupled_no_prefetch
            self.layers.append(layer)
            hf_model.model.layers[idx] = None

        self.num_layers = len(self.layers)

        # Fail-fast: decoupled offload requires q_future_proj (layers 0..L-2)
        # and q_curr_proj (layer 0) to be present, otherwise PREP/prefetch
        # is silently skipped and profiling becomes misleading.
        if self.decoupled and self.offload:
            missing_qf = [i for i in range(max(self.num_layers - 1, 0))
                          if not self.layers[i].has_q_future]
            if missing_qf:
                raise RuntimeError(
                    "Decoupled is enabled but q_future_proj is missing for "
                    f"{len(missing_qf)}/{max(self.num_layers - 1, 1)} layers "
                    f"(e.g. {missing_qf[:8]}...). "
                    "This usually means --indexer-path did not match the checkpoint key format. "
                    "Decoupled PREP/prefetch will be skipped."
                )
            if self.num_layers > 0 and not self.layers[0].has_q_curr:
                raise RuntimeError(
                    "Decoupled is enabled but q_curr_proj is missing for layer 0. "
                    "Layer-0 QK top-k will be stale and results will be incorrect."
                )
            print(f"[Decoupled] Enabled: q_future_proj on {self.num_layers - 1}/{self.num_layers - 1} layers, "
                  f"q_curr_proj on layer0={self.layers[0].has_q_curr}")

    @staticmethod
    def _normalize_indexer_state_dict(obj):
        """Normalize various checkpoint formats into a flat {str: Tensor} state_dict.

        Handles common wrappers like {'state_dict': ...} and strips a leading
        'module.' prefix from DDP checkpoints.
        """
        if obj is None:
            return None
        # Unwrap common checkpoint containers
        if isinstance(obj, dict):
            for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
                if k in obj and isinstance(obj[k], dict):
                    cand = obj[k]
                    if any(torch.is_tensor(v) for v in cand.values()):
                        obj = cand
                        break
        if not isinstance(obj, dict):
            return None
        # Keep only tensor weights (ignore optimizer/meta fields)
        sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        # Strip 'module.' prefix if present for all keys
        if sd and all(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        return sd

    @staticmethod
    def _inject_indexer_weights(hf_layer, layer_idx, indexer_sd):
        """Inject q_future_proj / q_curr_proj weights from an external state_dict into a HF layer."""
        import torch.nn as nn
        for proj_name in ("q_future_proj", "q_curr_proj"):
            # Be robust to different prefix conventions (module./model./base_model/...).
            suffix = f"layers.{layer_idx}.self_attn.{proj_name}.weight"
            w = None
            # Fast-path: common exact keys
            for k in (
                f"model.{suffix}",
                suffix,
                f"model.model.{suffix}",
                f"base_model.model.{suffix}",
                f"base_model.model.model.{suffix}",
            ):
                if k in indexer_sd:
                    w = indexer_sd[k]
                    break
            # Fallback: any key ending with the suffix
            if w is None:
                for k, v in indexer_sd.items():
                    if k.endswith(suffix):
                        w = v
                        break
            if w is None:
                continue
            # Match dtype with the base Q projection if possible (avoid upcasting WQKV to fp32).
            try:
                ref_dtype = hf_layer.self_attn.q_proj.weight.dtype
                w = w.to(dtype=ref_dtype)
            except Exception:
                pass
            proj = nn.Linear(w.shape[1], w.shape[0], bias=False)
            proj.weight = nn.Parameter(w.contiguous())
            setattr(hf_layer.self_attn, proj_name, proj)
    
    # Official MiniCPM4.1-8B 128K LongRoPE factors from:
    # https://huggingface.co/openbmb/MiniCPM4.1-8B
    _MINICPM41_128K_LONG_FACTOR = [
        0.9982316082870437, 1.033048153422584, 1.0749920956484724,
        1.1255096879436193, 1.1863348602111476, 1.259543828902579,
        1.3476188888731149, 1.4535223827776373, 1.5807816745852985,
        1.7335856049489526, 1.9168922912975785, 2.1365471404135326,
        2.3994084200118646, 2.713475511863602, 3.0880118452194134,
        3.533650295140154, 4.062463396503134, 4.687974098908333,
        5.425075306704039, 6.289818967956352, 7.29902962722721,
        8.6357018163639, 10.210822723989212, 12.053807765671676,
        14.193944598909404, 16.65780676784363, 19.463620727694074,
        22.628311203524586, 26.150106147261315, 30.02526691405111,
        34.23183327975347, 38.73811934094828, 43.502489489729555,
        48.47627117965394, 53.61139491762471, 58.857366522037935,
        64.16798299215064, 69.51359464319125, 74.86555458220285,
        80.21497790341579, 85.55322183307433, 90.89611806932027,
        96.26245306514224, 101.68269304046481, 107.18619510219668,
        112.82253283014026, 118.63764063163615, 119.88866203644656,
        120.9462882391725, 121.837565139014, 122.58663780572562,
        123.2147719894291, 123.74049454862576, 124.17980424685767,
        124.54641761955492, 124.85202548028222, 125.10654406389756,
        125.31835105170659, 125.49450117164764, 125.64091910903052,
        125.76256945356558, 125.86360463815589, 125.94749252260765,
        126.01712561287873,
    ]
    _MINICPM41_128K_SHORT_FACTOR = [
        0.9982316082870437, 1.033048153422584, 1.0749920956484724,
        1.1255096879436193, 1.1863348602111476, 1.259543828902579,
        1.3476188888731149, 1.4535223827776373, 1.5807816745852985,
        1.7335856049489526, 1.9168922912975785, 2.1365471404135326,
        2.3994084200118646, 2.713475511863602, 3.0880118452194134,
        3.533650295140154, 4.062463396503134, 4.687974098908333,
        5.425075306704039, 6.289818967956352, 7.29902962722721,
        8.6357018163639, 10.210822723989212, 12.053807765671676,
        14.193944598909404, 16.65780676784363, 19.463620727694074,
        22.628311203524586, 26.150106147261315, 30.02526691405111,
        34.23183327975347, 38.73811934094828, 43.502489489729555,
        48.47627117965394, 53.61139491762471, 58.857366522037935,
        64.16798299215064, 69.51359464319125, 74.86555458220285,
        80.21497790341579, 85.55322183307433, 90.89611806932027,
        96.26245306514224, 101.68269304046481, 107.18619510219668,
        112.82253283014026, 118.63764063163615, 119.88866203644656,
        120.9462882391725, 121.837565139014, 122.58663780572562,
        123.2147719894291, 123.74049454862576, 124.17980424685767,
        124.54641761955492, 124.85202548028222, 125.10654406389756,
        125.31835105170659, 125.49450117164764, 125.64091910903052,
        125.76256945356558, 125.86360463815589, 125.94749252260765,
        126.01712561287873,
    ]

    @classmethod
    def _apply_minicpm_128k_longrope(cls, rotary_emb):
        """Swap LongRoPE factors to the official 128K-validated set."""
        rotary_emb.long_factor = cls._MINICPM41_128K_LONG_FACTOR
        rotary_emb.short_factor = cls._MINICPM41_128K_SHORT_FACTOR

    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(self.max_length + 1024, device=self.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.float32), emb.sin().to(torch.float32)

    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        if input_len > 1:
            self.past_len = 0
        position_ids = torch.arange(self.past_len, self.past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        if input_len > 1:
            self.past_len = input_len
        else:
            self.past_len += 1
        return position_ids


    @torch.inference_mode()
    def _infinigen_warmup(self, input_ids, warmup_seq_len=2048, max_new_tokens=None):
        """InfiniGen warmup: compute per-layer skewing matrices.

        Request-specific partial indices are refreshed from each prefill rather
        than frozen from the warmup prompt.
        """
        if max_new_tokens is None:
            raise ValueError("max_new_tokens is required for InfiniGen warmup")
        warmup_len = min(input_ids.shape[1], warmup_seq_len)
        input_ids = input_ids[:1, :warmup_len].to(self.device)
        bsz = 1
        position_ids = torch.arange(warmup_len, device=self.device).unsqueeze(0)

        hidden_states = F.embedding(input_ids, self.embed_tokens) * self.scale_emb

        for idx, layer in enumerate(self.layers):
            residual = hidden_states
            hs_normed = _layer_norm(hidden_states, layer.input_layernorm_variance_epsilon,
                                     layer.input_layernorm_weight)
            qkv = F.linear(hs_normed, layer.wqkv)
            q = qkv[:, :, :layer.q_size]
            k = qkv[:, :, layer.q_size:layer.q_size + layer.kv_size]
            v = qkv[:, :, layer.q_size + layer.kv_size:layer.q_size + 2 * layer.kv_size]

            q = q.view(bsz, warmup_len, self.num_heads, self.head_dim)
            k = k.view(bsz, warmup_len, self.num_key_value_heads, self.head_dim)
            v = v.view(bsz, warmup_len, self.num_key_value_heads, self.head_dim)

            q_flat = q.view(-1, self.num_heads * self.head_dim)
            k_flat = k.view(-1, self.num_key_value_heads * self.head_dim)
            apply_rope_with_cos_sin_cache_inplace(
                position_ids.flatten(), q_flat, k_flat,
                self.head_dim, self.cos_sin_cache, True)
            q = q_flat.view(bsz, warmup_len, self.num_heads, self.head_dim)
            k = k_flat.view(bsz, warmup_len, self.num_key_value_heads, self.head_dim)

            q_act = q.transpose(1, 2).detach()
            k_act = k.transpose(1, 2).detach()

            A = compute_skewing_matrix(q_act, k_act, self.num_heads,
                                        self.num_key_value_heads, self.head_dim)
            layer.ig_skewing_matrix = A

            layer.ig_num_channels = self.infinigen_num_channels
            layer.ig_enabled = True
            layer.ig_partial_indices = None
            layer.ig_a_partial = None
            layer.ig_partial_key_cache = None
            layer._ig_cache_max = self.max_length + max_new_tokens

            from flash_attn import flash_attn_func
            out = flash_attn_func(q, k, v, causal=True)
            out = out.view(bsz, warmup_len, -1)
            out = F.linear(out, layer.wo)
            hidden_states = residual + out * layer.residual_scale

            residual = hidden_states
            hs_normed = _layer_norm(hidden_states, layer.post_attention_layernorm_variance_epsilon,
                                     layer.post_attention_layernorm_weight)
            gate_up = F.linear(hs_normed, layer.gate_up_proj)
            dd = gate_up.shape[-1] // 2
            out = torch.empty((*gate_up.shape[:-1], dd), dtype=gate_up.dtype, device=gate_up.device)
            silu_and_mul(gate_up, out)
            hidden_states = residual + F.linear(out, layer.down_proj) * layer.residual_scale

        self._infinigen_initialized = True
        print(f"[InfiniGen MiniCPM] Warmup complete: {self.num_layers} layers, "
              f"num_channels={self.infinigen_num_channels}")

    def _use_dense_inference(self, total_len: int) -> bool:
        return (
            not self.infinigen
            and (
                self.force_dense_inference
                or (self.dense_len > 0 and total_len < self.dense_len)
            )
        )

    def _sync_sparse_runtime_from_full_cache(self, cache_engine, bsz: int):
        seq_len = int(cache_engine.get_seq_length(0))
        if seq_len <= 0:
            return

        cu_seqlens = torch.arange(
            0, (bsz + 1) * seq_len, seq_len, dtype=torch.int32, device=self.device
        )
        for idx, layer in enumerate(self.layers):
            cache_layer = cache_engine.layers[idx]
            engine = cache_layer.cache_engine
            full_k = engine._k_cpu[:bsz, :seq_len, :, :]
            full_v = engine._v_cpu[:bsz, :seq_len, :, :]

            if full_k.device != self.device:
                full_k = full_k.to(self.device, non_blocking=False)
                full_v = full_v.to(self.device, non_blocking=False)
            else:
                full_k = full_k.contiguous()
                full_v = full_v.contiguous()

            kv_bias = torch.empty((bsz, seq_len, self.num_key_value_heads), dtype=full_k.dtype, device=full_k.device)
            cache_engine.prefill_update_kv(full_k, full_v, kv_bias, idx, 0, bsz)
            flat_k = full_k.reshape(bsz * seq_len, self.num_key_value_heads, self.head_dim)
            compressed_k, cu_seqlens_comp = compress_k(
                flat_k,
                cu_seqlens,
                self.num_key_value_heads,
                self.head_dim,
                kernel_size=layer.kernel_size,
                kernel_stride=layer.kernel_stride,
            )
            max_seqlen_comp = int((cu_seqlens_comp[1:] - cu_seqlens_comp[:-1]).max().item())
            cache_engine.update_compress_k_prefill(
                compressed_k,
                idx,
                cu_seqlens=cu_seqlens_comp,
                max_seqlen=max_seqlen_comp,
                current_batch_pos=0,
                total_bsz=bsz,
            )
            no_compress_start = max_seqlen_comp * layer.kernel_stride
            cache_engine.update_no_compress_k_prefill(
                full_k[:, no_compress_start:],
                idx,
                layer.kernel_size,
                layer.kernel_stride,
                current_batch_pos=0,
                total_bsz=bsz,
            )
        self._dense_decode_active = False

    @torch.inference_mode()
    def _prepare_infinigen_request_state(self, batch_size, cache_max):
        """Reset per-request InfiniGen state and preallocate batched partial caches."""
        cache_max = int(cache_max)
        for layer in self.layers:
            if not getattr(layer, "ig_enabled", False):
                continue
            layer._ig_cache_max = cache_max
            layer.ig_partial_indices = None
            layer.ig_a_partial = None
            layer.ig_partial_key_cache = torch.zeros(
                batch_size,
                cache_max,
                self.num_key_value_heads,
                self.infinigen_num_channels,
                dtype=self.dtype,
                device=self.device,
            )

    @torch.inference_mode()
    def _prepare_infinigen_prefill_indices(self, input_ids):
        """Refresh request-specific InfiniGen channels once per request."""
        if not (self.infinigen and self._infinigen_initialized):
            return
        if input_ids.numel() == 0:
            return
        position_ids = torch.arange(
            input_ids.shape[1], device=self.device, dtype=torch.long
        ).unsqueeze(0).repeat(input_ids.shape[0], 1)
        self._refresh_infinigen_partial_indices_from_prefill(input_ids, position_ids)

    @torch.inference_mode()
    def _refresh_infinigen_partial_indices_from_prefill(self, input_ids, position_ids):
        """Recompute request-specific InfiniGen channels from the current prefill.

        The controller currently extracts channels from batch element 0, so we
        refresh the shared selection using the first request in this batch.
        """
        if input_ids.numel() == 0:
            return

        request_input_ids = input_ids[:1].to(self.device)
        request_position_ids = position_ids[:1].to(self.device)
        bsz = 1
        seq_len = request_input_ids.shape[1]
        hidden_states = F.embedding(request_input_ids, self.embed_tokens) * self.scale_emb

        _precompute = get_precompute_a_partial()

        for layer in self.layers:
            residual = hidden_states
            hs_normed = _layer_norm(hidden_states, layer.input_layernorm_variance_epsilon,
                                    layer.input_layernorm_weight)
            qkv = F.linear(hs_normed, layer.wqkv)
            q = qkv[:, :, :layer.q_size]
            k = qkv[:, :, layer.q_size:layer.q_size + layer.kv_size]
            v = qkv[:, :, layer.q_size + layer.kv_size:layer.q_size + 2 * layer.kv_size]

            q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
            k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
            v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

            q_flat = q.view(-1, self.num_heads * self.head_dim)
            k_flat = k.view(-1, self.num_key_value_heads * self.head_dim)
            apply_rope_with_cos_sin_cache_inplace(
                request_position_ids.flatten(), q_flat, k_flat,
                self.head_dim, self.cos_sin_cache, True)
            q = q_flat.view(bsz, seq_len, self.num_heads, self.head_dim)
            k = k_flat.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

            q_act = q.transpose(1, 2).detach()
            k_act = k.transpose(1, 2).detach()

            if layer.ig_skewing_matrix is not None:
                indices = compute_partial_weight_indices(
                    q_act,
                    k_act,
                    layer.ig_skewing_matrix,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    num_channels=self.infinigen_num_channels,
                )
                layer.ig_partial_indices = indices
                layer.ig_a_partial = _precompute(layer.ig_skewing_matrix, indices)

            from flash_attn import flash_attn_func
            out = flash_attn_func(q, k, v, causal=True)
            out = out.view(bsz, seq_len, -1)
            out = F.linear(out, layer.wo)
            hidden_states = residual + out * layer.residual_scale

            residual = hidden_states
            hs_normed = _layer_norm(hidden_states, layer.post_attention_layernorm_variance_epsilon,
                                    layer.post_attention_layernorm_weight)
            gate_up = F.linear(hs_normed, layer.gate_up_proj)
            dd = gate_up.shape[-1] // 2
            out = torch.empty((*gate_up.shape[:-1], dd), dtype=gate_up.dtype, device=gate_up.device)
            silu_and_mul(gate_up, out)
            hidden_states = residual + F.linear(out, layer.down_proj) * layer.residual_scale


    @torch.inference_mode()
    def prefill_inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            cache_engine = None):
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        max_seqlen = seq_len

        use_dense_inference = self._use_dense_inference(seq_len)
        self._dense_decode_active = use_dense_inference

        output_hs = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=self.device)
        T = 1
        for t in tqdm(range(0, bsz, T), desc="prefill"):
            hs = F.embedding(input_ids[t:t+T], self.embed_tokens) * self.scale_emb
            cu_seqlens = torch.arange(0, hs.shape[0] * seq_len + 1, seq_len, dtype=torch.int, device=input_ids.device)
            q_future_for_next = None
            for idx in range(self.num_layers):
                if use_dense_inference:
                    hs = self.layers[idx].dense_prefill_forward(
                        hs, position_ids[t:t+T], self.cos_sin_cache,
                        cache_engine, total_bsz=bsz, current_batch_pos=t)
                else:
                    result = self.layers[idx].prefill_forward(
                        hs, position_ids[t:t+T], self.cos_sin_cache, cu_seqlens, max_seqlen,
                        cache_engine, total_bsz=bsz, current_batch_pos=t,
                        decoupled_mode=self.decoupled, q_future_from_prev=q_future_for_next)
                    if self.decoupled:
                        hs, q_future_for_next = result
                    else:
                        hs = result

                # Build InfiniGen partial key cache from prefill keys
                if (not use_dense_inference
                        and self.infinigen and self._infinigen_initialized
                        and self.layers[idx].ig_enabled
                        and self.layers[idx].ig_partial_indices is not None):
                    layer = self.layers[idx]
                    ce_layer = cache_engine.layers[idx]
                    ce = ce_layer.cache_engine
                    if hasattr(ce, '_k_cpu') and ce._k_cpu is not None:
                        full_k = ce._k_cpu[t]
                        k_len = int(ce.seq_length) if ce.seq_length > 0 else seq_len
                        partial_k = infinigen_build_partial_cache(
                            full_k[:k_len].to(self.device),
                            layer.ig_skewing_matrix, layer.ig_partial_indices)
                        if layer.ig_partial_key_cache is not None:
                            layer.ig_partial_key_cache[t, :k_len] = partial_k

            output_hs[t:t+T].copy_(hs[:, -1:, :], non_blocking=True)

        hidden_states = layer_norm(output_hs, w=self.norm_weight, eps=self.norm_variance_epsilon)

        # muP LM head scaling (MiniCPM-style): logits = lm_head(h / scale)
        if self.lm_head_scale != 1.0:
            hidden_states = hidden_states / self.lm_head_scale
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits
    

    @torch.inference_mode()
    def decode_inference(self,
            input_ids: torch.LongTensor,
            cu_seqlens: torch.Tensor,
            position_ids: torch.LongTensor,
            cache_engine = None):
        hidden_states = F.embedding(input_ids, self.embed_tokens) * self.scale_emb
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        max_seqlen = 1

        total_len = cache_engine.get_seq_length(0) + seq_len
        if self._use_dense_inference(total_len):
            self._dense_decode_active = True
            for idx in range(self.num_layers):
                hidden_states = self.layers[idx].dense_decode_forward(
                    hidden_states, position_ids, self.cos_sin_cache, cache_engine
                )

            hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
            if self.lm_head_scale != 1.0:
                hidden_states = hidden_states / self.lm_head_scale
            logits = F.linear(hidden_states, self.lm_head).float()
            return logits

        if self._dense_decode_active:
            self._sync_sparse_runtime_from_full_cache(cache_engine, bsz)

        cache_lens = cache_engine.get_seq_length(0)
        cache_lens = torch.tensor([cache_lens] * bsz, dtype=torch.int, device=hidden_states.device)
        q_idx = cache_lens // self.block_size

        need_warmup = not self.has_buffers

        if need_warmup:

            
            total_len = cache_engine.get_seq_length(0) + 1
            out_len = (total_len + self.block_size - 1) // self.block_size
            self.pooling_buf_all = torch.empty(2 * self.num_key_value_heads, bsz, out_len, device=hidden_states.device, dtype=hidden_states.dtype)
            self.max_pooling_buf = torch.empty(self.num_key_value_heads, bsz, out_len, device=hidden_states.device, dtype=hidden_states.dtype)
            self.max_pooling_buf_cis = torch.empty(self.num_key_value_heads, bsz, out_len, device=hidden_states.device, dtype=hidden_states.dtype)
            self.topk_buf_val = torch.empty(self.num_key_value_heads, bsz, self.topk, device=hidden_states.device, dtype=hidden_states.dtype)
            self.topk_buf_indices = torch.empty(self.num_key_value_heads, bsz, self.topk, device=hidden_states.device, dtype=torch.int64)
            self.topk_val_buf_q = torch.empty(self.num_key_value_heads, bsz, 33, device=hidden_states.device, dtype=hidden_states.dtype)
            self.topk_idx_buf_q = torch.empty(self.num_key_value_heads, bsz, 33, device=hidden_states.device, dtype=torch.int64)
            self.topk_val_buf = torch.empty(self.num_key_value_heads, bsz, self.topk, device=hidden_states.device, dtype=hidden_states.dtype)
            self.topk_idx_buf = torch.empty(self.num_key_value_heads, bsz, self.topk, device=hidden_states.device, dtype=torch.int64)
            self.mask_buf = torch.empty(self.num_key_value_heads, bsz, out_len, device=hidden_states.device, dtype=torch.bool)

            self.has_buffers = True

            _seed_nooff = getattr(self, '_collect_nooff_topk_stats', False) and not self.offload
            q_future_for_next = None
            for idx in range(self.num_layers):
                result = self.layers[idx].decode_forward_warmup(hidden_states, position_ids, self.cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx, self.pooling_buf_all, self.topk_buf_val, self.topk_buf_indices, self.topk_val_buf_q, self.topk_idx_buf_q, self.topk_val_buf, self.topk_idx_buf, self.mask_buf, decoupled_mode=self.decoupled, q_future_from_prev=q_future_for_next)
                if self.decoupled:
                    hidden_states, q_future_for_next = result
                else:
                    hidden_states = result
                if _seed_nooff:
                    if not hasattr(self, '_nooff_prev_topk'):
                        self._nooff_prev_topk = [None] * self.num_layers
                        self._nooff_miss = [0] * self.num_layers
                        self._nooff_total = [0] * self.num_layers
                        self._nooff_steps = [0] * self.num_layers
                    self._nooff_prev_topk[idx] = self.topk_idx_buf.detach().clone()

            # Initialize InfiniGen partial key caches (graphs already captured by warmup above)
            if self.infinigen and self._infinigen_initialized:
                for _idx in range(self.num_layers):
                    if self.layers[_idx].ig_partial_key_cache is None:
                        _max = self.layers[_idx]._ig_cache_max
                        self.layers[_idx].ig_partial_key_cache = torch.zeros(
                            bsz, _max, self.num_key_value_heads, self.infinigen_num_channels,
                            dtype=self.dtype, device=self.device)

            # Initialize decoupled buffers right after warmup (so decoupled starts at step 2)
            if (
                self.decoupled
                and self.offload
                and not self.decoupled_no_prefetch
                and not self._decoupled_initialized
            ):
                self._init_decoupled_buffers(bsz, cache_engine)
            # Initialize infinigen offload buffers (uses same infrastructure as decoupled)
            if (self.infinigen and self._infinigen_initialized
                    and self.offload and not getattr(self, '_ig_offload_initialized', False)):
                self._init_infinigen_offload_buffers(bsz, cache_engine)
        else:
            # InfiniGen decode loop with per-layer offload + prefetch
            # H2D uses CPU gather + non_blocking transfer (not Triton persistent kernel)
            if (self.infinigen and self._infinigen_initialized
                    and self.offload and getattr(self, '_ig_offload_initialized', False)):

                prefetch_stream = self._prefetch_stream
                gather_worker = self._ig_gather_worker

                layers = self.layers
                num_layers = self.num_layers
                ce_layers = cache_engine.layers
                prep_graph = self._prep_graph
                score_buf_prep_input = self._score_buf_prep_input

                pending_prefetch_done_event = None
                collect_stats = getattr(self, '_collect_h2d_stats', False)
                for idx in range(num_layers):
                    next_layer = layers[idx + 1] if idx + 1 < num_layers else None
                    next_cache_layer = ce_layers[idx + 1] if idx + 1 < num_layers else None

                    # Per-task ready event: created here, passed into the
                    # forward where _infinigen_prep_next_layer records it
                    # RIGHT AFTER PREP (diff_offload) — before wait_event
                    # and stage2.  The BG thread can start the gather as
                    # soon as PREP finishes, overlapping with the rest of
                    # this layer's forward.
                    need_gather = (idx + 1 < num_layers
                                   and next_layer is not None
                                   and getattr(next_layer, 'ig_enabled', False)
                                   and layers[idx].ig_enabled)
                    task_ready_event = torch.cuda.Event() if need_gather else None

                    hidden_states = layers[idx].decode_forward_infinigen(
                        hidden_states, position_ids, self.cos_sin_cache,
                        cu_seqlens, max_seqlen,
                        cache_engine, cache_lens, q_idx,
                        self.pooling_buf_all, self.topk_buf_val, self.topk_buf_indices,
                        self.topk_val_buf_q, self.topk_idx_buf_q, self.topk_val_buf, self.topk_idx_buf, self.mask_buf,
                        skip_wait=(idx == 0),
                        prefetch_done_event=pending_prefetch_done_event,
                        prefetch_ready_event=task_ready_event,
                        next_layer=next_layer,
                        prep_graph=prep_graph,
                        score_buf_prep_input=score_buf_prep_input,
                        next_layer_cache_layer=next_cache_layer,
                    )

                    # Submit gather for next layer. The background worker
                    # gathers into pinned CPU staging and then records the
                    # next layer's H2D/scatter event on the prefetch stream.
                    next_prefetch_done_event = None
                    if need_gather:
                        ce_next = ce_layers[idx + 1].cache_engine
                        if hasattr(ce_next, '_k_cpu') and hasattr(ce_next, '_prefetch_load_mask'):
                            next_prefetch_done_event = torch.cuda.Event()
                            gather_worker.submit(
                                ce_next=ce_next,
                                prefetch_ready_event=task_ready_event,
                                prefetch_done_event=next_prefetch_done_event,
                                record_stats=collect_stats,
                            )
                    pending_prefetch_done_event = next_prefetch_done_event

            elif (self.infinigen and self._infinigen_initialized
                      and not self.offload):
                # InfiniGen without offload — simple forward loop using
                # decode_forward_infinigen with per-layer graphs (no shared
                # prep_graph, no gather worker, no H2D).
                layers = self.layers
                num_layers = self.num_layers
                for idx in range(num_layers):
                    next_layer = layers[idx + 1] if idx + 1 < num_layers else None
                    next_cache_layer = (cache_engine.layers[idx + 1]
                                        if idx + 1 < num_layers else None)
                    hidden_states = layers[idx].decode_forward_infinigen(
                        hidden_states, position_ids, self.cos_sin_cache,
                        cu_seqlens, max_seqlen,
                        cache_engine, cache_lens, q_idx,
                        self.pooling_buf_all, self.topk_buf_val,
                        self.topk_buf_indices,
                        self.topk_val_buf_q, self.topk_idx_buf_q,
                        self.topk_val_buf, self.topk_idx_buf, self.mask_buf,
                        skip_wait=(idx == 0),
                        next_layer=next_layer,
                        next_layer_cache_layer=next_cache_layer,
                    )

            elif self.decoupled and self.offload and self.decoupled_no_prefetch:
                q_future_for_next = None
                for idx in range(self.num_layers):
                    hidden_states, q_future_for_next = self.layers[idx].decode_forward_decoupled_no_offload(
                        hidden_states, position_ids, self.cos_sin_cache, cu_seqlens, max_seqlen,
                        cache_engine, cache_lens, q_idx,
                        self.pooling_buf_all, self.topk_buf_val, self.topk_buf_indices,
                        self.topk_val_buf_q, self.topk_idx_buf_q, self.topk_val_buf, self.topk_idx_buf, self.mask_buf,
                        q_future_from_prev=q_future_for_next,
                    )
            elif self.decoupled and self.offload and self._decoupled_initialized:
                from .flash_cache_engine.flash_h2d_persistent import flash_h2d_kv_persistent

                prefetch_stream = self._prefetch_stream
                prefetch_ready_event = self._prefetch_ready_event

                # Cache attribute lookups to reduce per-iteration Python overhead
                layers = self.layers
                num_layers = self.num_layers
                ce_layers = cache_engine.layers
                cos_sin_cache = self.cos_sin_cache
                prep_graph = self._prep_graph
                score_buf_prep = self._score_buf_prep
                score_buf_prep_input = self._score_buf_prep_input
                block_map_buf = self._block_map_buf
                new_map_buf = self._new_map_buf
                load_mask_buf = self._load_mask_buf
                pooled_q_buf = self._pooled_q_buf
                pooled_cis_dummy_buf = self._pooled_cis_dummy_buf
                pooling_buf_all = self.pooling_buf_all
                topk_buf_val = self.topk_buf_val
                topk_buf_indices = self.topk_buf_indices
                topk_val_buf_q = self.topk_val_buf_q
                topk_idx_buf_q = self.topk_idx_buf_q
                topk_val_buf = self.topk_val_buf
                topk_idx_buf = self.topk_idx_buf
                mask_buf = self.mask_buf

                pending_prefetch_done_event = None
                for idx in range(num_layers):
                    next_cache_layer = ce_layers[idx + 1] if idx + 1 < num_layers else None

                    hidden_states = layers[idx].decode_forward_decoupled(
                        hidden_states, position_ids, cos_sin_cache, cu_seqlens, max_seqlen,
                        cache_engine, cache_lens, q_idx,
                        pooling_buf_all, topk_buf_val, topk_buf_indices,
                        topk_val_buf_q, topk_idx_buf_q, topk_val_buf, topk_idx_buf, mask_buf,
                        skip_wait=(idx == 0),
                        prefetch_done_event=pending_prefetch_done_event,
                        prefetch_ready_event=prefetch_ready_event,
                        prep_graph=prep_graph,
                        score_buf_prep=score_buf_prep,
                        block_map_buf=block_map_buf,
                        new_map_buf=new_map_buf,
                        load_mask_buf=load_mask_buf,
                        pooled_q_buf=pooled_q_buf,
                        pooled_cis_dummy_buf=pooled_cis_dummy_buf,
                        next_layer_cache_layer=next_cache_layer,
                        score_buf_prep_input=score_buf_prep_input,
                        prefetch_stream=prefetch_stream,
                    )

                    next_prefetch_done_event = None
                    # Launch H2D for next layer on prefetch stream (UVA Triton kernel)
                    if idx + 1 < num_layers and layers[idx].has_q_future:
                        ce = ce_layers[idx + 1].cache_engine
                        next_prefetch_done_event = torch.cuda.Event()
                        with torch.cuda.stream(prefetch_stream):
                            nvtx.range_push("h2d_prefetch")
                            prefetch_stream.wait_event(prefetch_ready_event)
                            ce.record_prefetch_stats()
                            flash_h2d_kv_persistent(
                                ce._k_gpu, ce._k_cpu,
                                ce._v_gpu, ce._v_cpu,
                                load_ids=ce._prefetch_load_mask,
                            )
                            next_prefetch_done_event.record(prefetch_stream)
                            nvtx.range_pop()
                    pending_prefetch_done_event = next_prefetch_done_event
            elif self.decoupled and not self.offload:
                # No-offload + decoupled path: thread q_future between layers
                q_future_for_next = None
                _track = getattr(self, '_collect_nooff_topk_stats', False)
                for idx in range(self.num_layers):
                    hidden_states, q_future_for_next = self.layers[idx].decode_forward_decoupled_no_offload(
                        hidden_states, position_ids, self.cos_sin_cache, cu_seqlens, max_seqlen,
                        cache_engine, cache_lens, q_idx,
                        self.pooling_buf_all, self.topk_buf_val, self.topk_buf_indices,
                        self.topk_val_buf_q, self.topk_idx_buf_q, self.topk_val_buf, self.topk_idx_buf, self.mask_buf,
                        q_future_from_prev=q_future_for_next,
                    )
                    if _track:
                        curr = self.topk_idx_buf.detach()
                        if not hasattr(self, '_nooff_prev_topk'):
                            self._nooff_prev_topk = [None] * self.num_layers
                            self._nooff_miss = [0] * self.num_layers
                            self._nooff_total = [0] * self.num_layers
                            self._nooff_steps = [0] * self.num_layers
                        prev = self._nooff_prev_topk[idx]
                        if prev is not None and prev.shape == curr.shape:
                            in_prev = (curr.unsqueeze(-1) == prev.unsqueeze(-2)).any(-1)
                            valid = curr != -1
                            miss = int((valid & ~in_prev).sum().item())
                            self._nooff_miss[idx] += miss
                            self._nooff_total[idx] += curr.numel()
                            self._nooff_steps[idx] += 1
                        self._nooff_prev_topk[idx] = curr.clone()
            else:
                # Non-decoupled path (fallback when decoupled=False)
                _track = getattr(self, '_collect_nooff_topk_stats', False) and not self.offload
                for idx in range(self.num_layers):
                    hidden_states = self.layers[idx].decode_forward(hidden_states, position_ids, self.cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx, self.pooling_buf_all, self.topk_buf_val, self.topk_buf_indices, self.topk_val_buf_q, self.topk_idx_buf_q, self.topk_val_buf, self.topk_idx_buf, self.mask_buf)
                    if _track:
                        curr = self.topk_idx_buf.detach()
                        if not hasattr(self, '_nooff_prev_topk'):
                            self._nooff_prev_topk = [None] * self.num_layers
                            self._nooff_miss = [0] * self.num_layers
                            self._nooff_total = [0] * self.num_layers
                            self._nooff_steps = [0] * self.num_layers
                        prev = self._nooff_prev_topk[idx]
                        if prev is not None and prev.shape == curr.shape:
                            in_prev = (curr.unsqueeze(-1) == prev.unsqueeze(-2)).any(-1)
                            valid = curr != -1
                            miss = int((valid & ~in_prev).sum().item())
                            self._nooff_miss[idx] += miss
                            self._nooff_total[idx] += curr.numel()
                            self._nooff_steps[idx] += 1
                        self._nooff_prev_topk[idx] = curr.clone()
        
        profile_verbose = bool(getattr(self, "_profile_verbose", False))
        if profile_verbose:
            nvtx.range_push("detail: final norm")
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        if profile_verbose:
            nvtx.range_pop()
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        # muP LM head scaling (MiniCPM-style): logits = lm_head(h / scale)
        if profile_verbose:
            nvtx.range_push("detail: lm head")
        if self.lm_head_scale != 1.0:
            hidden_states = hidden_states / self.lm_head_scale
        logits = F.linear(hidden_states, self.lm_head).float()
        if profile_verbose:
            nvtx.range_pop()
        
        return logits

    def _init_infinigen_offload_buffers(self, bsz, cache_engine):
        """Initialize infinigen offload buffers (reuses decoupled buffer infrastructure)."""
        self._init_decoupled_buffers(bsz, cache_engine)
        # _init_decoupled_buffers sets _decoupled_initialized; reset it and use our own flag
        self._decoupled_initialized = False
        self._ig_gather_worker = _InfiniGenGatherWorker(self._prefetch_stream)
        self._ig_offload_initialized = True

    def _init_decoupled_buffers(self, bsz, cache_engine):
        """Initialize decoupled prefetch buffers after warmup's first non-decoupled step."""
        H = self.num_key_value_heads
        device = self.device
        dtype = self.dtype

        self._prefetch_stream = torch.cuda.Stream(device=device, priority=0)
        self._prefetch_ready_event = torch.cuda.Event()

        # score_buf_prep must accommodate max_seqlen_comp (compressed entries),
        # not out_len (blocks), since decoupled_stage1 produces one score per
        # compressed entry.
        max_seqlen_comp = cache_engine.layers[0].cached_compressed_max_seqlen
        out_len = self.pooling_buf_all.shape[-1]
        # Use padded pitch (same as warmup score_buf when available) to keep
        # pooling memory access aligned; then take a width-C view.
        prep_pitch = max_seqlen_comp
        if hasattr(self, "score_buf") and self.score_buf is not None:
            prep_pitch = max(prep_pitch, int(self.score_buf.shape[-1]))
        self._score_buf_prep_storage = torch.full(
            (H, bsz, prep_pitch), -float('inf'), dtype=dtype, device=device
        )
        self._score_buf_prep = self._score_buf_prep_storage[:, :, :max_seqlen_comp]
        # Contiguous staging buffer fed by decoupled_stage1 output.
        self._score_buf_prep_input = torch.empty(
            (H, bsz, max_seqlen_comp), dtype=dtype, device=device
        )
        self._block_map_buf = torch.empty(H, bsz, self.topk, dtype=torch.int64, device=device)
        self._new_map_buf = torch.empty(H, bsz, self.topk, dtype=torch.int64, device=device)
        self._load_mask_buf = torch.empty(H, bsz, self.topk, dtype=torch.int64, device=device)
        self._pooled_q_buf = torch.empty(H, bsz, out_len, dtype=dtype, device=device)
        self._pooled_cis_dummy_buf = torch.empty(H, bsz, out_len, dtype=dtype, device=device)

        # --- prep_graph capture (pooling + topk) ---
        # NOTE: diff_offload is a custom C++ extension that is NOT CUDA-graph-safe
        # (its kernel launch is silently skipped during graph replay).
        # We capture only pooling + topk, then run diff_offload inline after replay.
        # Warmup run (triggers JIT compilation for nosa_pooling)
        self._score_buf_prep_input.zero_()
        self._score_buf_prep.copy_(self._score_buf_prep_input)
        nosa_pooling(self._score_buf_prep, self._score_buf_prep,
                     max_seqlen_comp,
                     self._pooled_q_buf, self._pooled_cis_dummy_buf,
                     stride=self.kernel_stride, block_size=self.block_size, local_blocks=self.local_blocks, init_blocks=self.init_blocks)
        torch.topk(self._pooled_q_buf, self.topk, dim=-1, sorted=False,
                    out=(self.topk_val_buf, self.topk_idx_buf))
        # Capture (without diff_offload)
        self._prep_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._prep_graph):
            self._score_buf_prep.copy_(self._score_buf_prep_input)
            nosa_pooling(self._score_buf_prep, self._score_buf_prep,
                         max_seqlen_comp,
                         self._pooled_q_buf, self._pooled_cis_dummy_buf,
                         stride=self.kernel_stride, block_size=self.block_size, local_blocks=self.local_blocks, init_blocks=self.init_blocks)
            torch.topk(self._pooled_q_buf, self.topk, dim=-1, sorted=False,
                        out=(self.topk_val_buf, self.topk_idx_buf))

        self._decoupled_initialized = True

    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, cache_engine=None):
        batch_size = input_ids.size(0)
        if cache_engine is not None and hasattr(cache_engine, "prepare_prefill"):
            cache_engine.prepare_prefill(batch_size, input_ids.shape[-1])
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)
        position_ids = torch.zeros(batch_size, input_ids.shape[-1], device=self.device, dtype=torch.int)
        T = 10000  # we do not chunk here
        for bsz in range(0, batch_size, T):
            req_input_ids = input_ids[bsz:bsz+T]
            pos_ids = self.get_ctx(req_input_ids)
            logits[bsz:bsz+T].copy_(self.prefill_inference(input_ids=req_input_ids, position_ids=pos_ids, cache_engine=cache_engine))
            position_ids[bsz:bsz+T].copy_(pos_ids)
        if cache_engine is not None and hasattr(cache_engine, "wait_prefill_offload"):
            cache_engine.wait_prefill_offload()
        return logits, position_ids

    def _use_layer0_resident_cache(self) -> bool:
        return (
            self.offload
            and (self.decoupled or self.infinigen)
            and not self.offload_layer0
        )

    def batch_generate(self, input_ids, max_new_tokens=4):
        if self.infinigen:
            if not self._infinigen_initialized:
                self._infinigen_warmup(input_ids, max_new_tokens=max_new_tokens)
            requested_cache_max = int(input_ids.shape[1]) + int(max_new_tokens)
            self._prepare_infinigen_request_state(input_ids.shape[0], requested_cache_max)
            self._prepare_infinigen_prefill_indices(input_ids)
        if hasattr(self, '_ig_gather_worker'):
            self._ig_gather_worker.shutdown()
            del self._ig_gather_worker

        cache_cls = InfLLMv2Cache if self.offload else InfLLMv2CacheNoOffload
        l0_res = self._use_layer0_resident_cache()
        self._last_cache_engine = None
        self.has_buffers = False
        self._decoupled_initialized = False
        self._ig_offload_initialized = False
        self._dense_decode_active = False
        if hasattr(self, '_nooff_prev_topk'):
            del self._nooff_prev_topk, self._nooff_miss, self._nooff_total, self._nooff_steps
        self._nooff_step_count = 0
        cache_engine = cache_cls(config=self.config, num_hidden_layers=self.config.num_hidden_layers, has_kv_bias=False,
                                 max_gen_len=max_new_tokens,
                                 **({"l0_resident": True} if l0_res else {}))
        if hasattr(cache_engine, "prepare_prefill"):
            cache_engine.prepare_prefill(input_ids.shape[0], input_ids.shape[1])
        logits, position_ids = self.batch_prefill(input_ids, cache_engine)

        next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = [next_ids]
        position_ids = position_ids[:, -1:] + 1
        profile_verbose = bool(getattr(self, "_profile_verbose", False))

        cu_seqlens_de = torch.arange(0, next_ids.shape[0] + 1, dtype=torch.int, device=input_ids.device)

        for idx in range(max_new_tokens - 1):
            logits = self.decode_inference(next_ids, cu_seqlens_de, position_ids, cache_engine)

            next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(next_ids)
            position_ids = position_ids[:, -1:] + 1
        gen_ids = torch.cat(gen_ids, dim=-1)
        return gen_ids

    def batch_generate_benchmark(self, input_ids, max_new_tokens=4):
        if self.infinigen:
            if not self._infinigen_initialized:
                self._infinigen_warmup(input_ids, max_new_tokens=max_new_tokens)
            requested_cache_max = int(input_ids.shape[1]) + int(max_new_tokens)

        # Free GPU memory from the previous call before allocating a new cache
        self._last_cache_engine = None
        for attr in ('pooling_buf_all', 'max_pooling_buf', 'max_pooling_buf_cis',
                      'topk_buf_val', 'topk_buf_indices', 'topk_val_buf_q',
                      'topk_idx_buf_q', 'topk_val_buf', 'topk_idx_buf', 'mask_buf',
                      '_score_buf_prep_storage', '_score_buf_prep', '_score_buf_prep_input',
                      '_block_map_buf', '_new_map_buf', '_load_mask_buf',
                      '_pooled_q_buf', '_pooled_cis_dummy_buf', '_prep_graph',
                      '_prefetch_stream', '_prefetch_ready_event'):
            if hasattr(self, attr):
                delattr(self, attr)
        for layer in getattr(self, 'layers', []):
            for attr in ('after_pooling_graph', 'score_buf', 'compressed_cis_buf', 'score_copy_dst'):
                if hasattr(layer, attr):
                    delattr(layer, attr)
            if hasattr(layer, 'ig_partial_key_cache'):
                layer.ig_partial_key_cache = None
        if hasattr(self, '_ig_gather_worker'):
            self._ig_gather_worker.shutdown()
            del self._ig_gather_worker
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()

        if self.infinigen:
            self._prepare_infinigen_request_state(input_ids.shape[0], requested_cache_max)
            self._prepare_infinigen_prefill_indices(input_ids)

        cache_cls = InfLLMv2Cache if self.offload else InfLLMv2CacheNoOffload
        l0_res = self._use_layer0_resident_cache()
        cache_engine = cache_cls(config=self.config, num_hidden_layers=self.config.num_hidden_layers, has_kv_bias=False,
                                 max_gen_len=max_new_tokens,
                                 **({"l0_resident": True} if l0_res else {}))
        if hasattr(cache_engine, "prepare_prefill"):
            cache_engine.prepare_prefill(input_ids.shape[0], input_ids.shape[1])
        # Each call creates a new cache_engine, so CUDA graphs captured during
        # the previous call's warmup reference stale GPU memory.  Reset so the
        # first decode step re-does warmup + graph capture.
        self.has_buffers = False
        self._decoupled_initialized = False
        self._ig_offload_initialized = False
        self._dense_decode_active = False
        # Reset no-offload topk tracker
        if hasattr(self, '_nooff_prev_topk'):
            del self._nooff_prev_topk, self._nooff_miss, self._nooff_total, self._nooff_steps
        self._nooff_step_count = 0
        profile_verbose = bool(getattr(self, "_profile_verbose", False))
        collect_split_timing = bool(getattr(self, "_collect_prefill_decode_timings", False))
        self._last_prefill_time_s = None
        self._last_decode_time_s = None
        if collect_split_timing:
            torch.cuda.synchronize()
            prefill_beg = time.time()
        nvtx.range_push("__prefill_timed__")
        logits, position_ids = self.batch_prefill(input_ids, cache_engine)
        if profile_verbose:
            nvtx.range_push("detail: token select")
        next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if profile_verbose:
            nvtx.range_pop()
        nvtx.range_pop()
        if collect_split_timing:
            torch.cuda.synchronize()
            self._last_prefill_time_s = max(0.0, time.time() - prefill_beg)

        gen_ids = [next_ids]
        position_ids = position_ids[:, -1:] + 1

        cu_seqlens_de = torch.arange(0, next_ids.shape[0] + 1, dtype=torch.int, device=input_ids.device)

        for it in range(max_new_tokens - 1):
            logits = self.decode_inference(next_ids, cu_seqlens_de, position_ids, cache_engine)
            # Drain BG gather queue before the next step overwrites
            # _prefetch_load_mask during PREP.
            if hasattr(self, '_ig_gather_worker'):
                if profile_verbose:
                    nvtx.range_push("detail: gather drain")
                self._ig_gather_worker.drain()
                if profile_verbose:
                    nvtx.range_pop()

            if profile_verbose:
                nvtx.range_push("detail: token select")
            next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(next_ids)
            position_ids = position_ids[:, -1:] + 1
            if profile_verbose:
                nvtx.range_pop()
            if it == 0:
                # Enable H2D stats for steady-state decode if profiler requested
                if getattr(self, '_collect_h2d_stats', False) and hasattr(cache_engine, 'enable_h2d_stats'):
                    cache_engine.enable_h2d_stats()
                if getattr(self, '_collect_h2d_stats', False) and hasattr(self, '_ig_gather_worker'):
                    self._ig_gather_worker.reset_breakdown()
                    self._ig_gather_worker.enable_breakdown(True)
                torch.cuda.synchronize()
                beg = time.time()
                nvtx.range_push("__decode_timed__")
        nvtx.range_pop()  # __decode_timed__
        if hasattr(self, '_ig_gather_worker'):
            self._ig_gather_worker.drain()
        torch.cuda.synchronize()
        end = time.time()
        if collect_split_timing:
            self._last_decode_time_s = max(0.0, end - beg)
        if hasattr(self, '_ig_gather_worker'):
            self._ig_gather_worker.enable_breakdown(False)
        gen_ids = torch.cat(gen_ids, dim=-1)
        # Keep cache engine accessible for H2D stats retrieval
        self._last_cache_engine = cache_engine
        return gen_ids, gen_ids.shape[0] * (max_new_tokens - 2) / (end - beg)

    def get_ig_h2d_breakdown(self):
        """Return H2D phase breakdown dict, or None if not available."""
        if hasattr(self, '_ig_gather_worker'):
            return self._ig_gather_worker.get_breakdown()
        return None

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
