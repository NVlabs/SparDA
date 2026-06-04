# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This code is based on the original HuggingFace MiniCPM implementation with
# InfLLM V2 future prediction integration for sparse attention.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniCPM model with InfLLM V2 future prediction integration."""

import math
import contextlib
import warnings
import os
import sys
from typing import Any, List, Optional, Tuple, Union, Dict
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
import re

# Try to import the original MiniCPM configuration
try:
    from .configuration_minicpm import MiniCPMConfig
except ImportError:
    try:
        from configuration_minicpm import MiniCPMConfig
    except ImportError:
        from transformers import PretrainedConfig
        
        class MiniCPMConfig(PretrainedConfig):
            """Minimal MiniCPM4.1-8B config placeholder - will be replaced at load time."""
            model_type = "minicpm"
            keys_to_ignore_at_inference = ["past_key_values"]
            
            def __init__(
                self,
                vocab_size=73448,
                hidden_size=4096,
                intermediate_size=16384,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=2,
                hidden_act="silu",
                max_position_embeddings=65536,
                initializer_range=0.1,
                rms_norm_eps=1e-6,
                use_cache=True,
                pad_token_id=2,
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
                rope_theta=10000.0,
                rope_scaling=None,
                attention_dropout=0.0,
                scale_emb=12,
                dim_model_base=256,
                scale_depth=1.4,
                mup_denominator=32,
                attention_bias=False,
                pretraining_tp=1,
                sparse_config=None,
                **kwargs,
            ):
                super().__init__(
                    pad_token_id=pad_token_id,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    tie_word_embeddings=tie_word_embeddings,
                    **kwargs,
                )
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.intermediate_size = intermediate_size
                self.num_hidden_layers = num_hidden_layers
                self.num_attention_heads = num_attention_heads
                self.num_key_value_heads = num_key_value_heads
                self.hidden_act = hidden_act
                self.max_position_embeddings = max_position_embeddings
                self.initializer_range = initializer_range
                self.rms_norm_eps = rms_norm_eps
                self.mup_denominator = mup_denominator
                self.sparse_config = sparse_config
                self.use_cache = use_cache
                self.rope_theta = rope_theta
                self.rope_scaling = rope_scaling
                self.attention_dropout = attention_dropout
                self.scale_emb = scale_emb
                self.dim_model_base = dim_model_base
                self.scale_depth = scale_depth
                self.attention_bias = attention_bias
                self.pretraining_tp = pretraining_tp

# Flash Attention imports
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    flash_attn_available = True
except ImportError:
    flash_attn_available = False
    flash_attn_func = None
    flash_attn_varlen_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None

# InfLLM V2 kernel imports
try:
    from infllm_v2 import (
        infllmv2_attn_stage1,
        infllmv2_attn_varlen_func,
        infllmv2_attn_with_kvcache,
        max_pooling_1d,
        max_pooling_1d_varlen
    )
    try:
        from infllm_v2 import infllmv2_attn_stage1_kl_teacher
    except ImportError:
        infllmv2_attn_stage1_kl_teacher = None
    infllm_ops_available = True
except ImportError:
    infllmv2_attn_varlen_func = None
    infllmv2_attn_stage1 = None
    infllmv2_attn_stage1_kl_teacher = None
    infllmv2_attn_with_kvcache = None
    max_pooling_1d = None
    max_pooling_1d_varlen = None
    infllm_ops_available = False

# InfiniGen controller imports (used only when the standalone MiniCPM
# InfiniGen adapter enables the relevant runtime state on attention layers).
_controllers_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'NOSA', 'nosi', 'nosi')
if _controllers_dir not in sys.path:
    sys.path.insert(0, _controllers_dir)

try:
    from infinigen_controllers import (
        infinigen_token_scores_hf,
        infinigen_build_partial_cache,
        infinigen_update_partial_cache,
    )
    _INFINIGEN_AVAILABLE = True
except ImportError:
    _INFINIGEN_AVAILABLE = False

if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "MiniCPMConfig"

# Global flag to skip KL loss during gradient checkpointing recompute
# This saves ~1GB+ memory during backward pass. Set via set_skip_kl_loss().
_SKIP_KL_LOSS = False

def set_skip_kl_loss(skip: bool):
    """Set global flag to skip KL loss computation (used during grad checkpoint recompute)."""
    global _SKIP_KL_LOSS
    _SKIP_KL_LOSS = skip


@contextlib.contextmanager
def skip_kl_loss(enabled: bool = True):
    """Context manager to temporarily toggle KL-loss computation.

    This avoids state leaks when exceptions occur and keeps callsites concise.
    """
    global _SKIP_KL_LOSS
    prev = _SKIP_KL_LOSS
    _SKIP_KL_LOSS = enabled
    try:
        yield
    finally:
        _SKIP_KL_LOSS = prev


# =============================================================================
# HELPER FUNCTIONS FOR SPARSE ATTENTION
# =============================================================================

def _call_infllmv2_stage1_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_v: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> torch.Tensor:
    """Call stage1 with the repo API and fail fast on stale infllm_v2 builds."""
    try:
        return infllmv2_attn_stage1(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_v,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )
    except TypeError as exc:
        if "cu_seqlens_v" not in str(exc):
            raise
        raise RuntimeError(
            "MiniCPM training requires the repo infllm_v2 extension with cu_seqlens_v support. "
            "Launch training via training/run_train.sh so the repo "
            "extension is built and imported instead of the old package."
        ) from exc


def _call_infllmv2_stage1_kl_teacher_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_v: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    causal_stride: int,
    query_offset: int,
) -> torch.Tensor:
    """Call the training-only KL-teacher stage1 op and fail fast on stale builds."""
    if infllmv2_attn_stage1_kl_teacher is None:
        raise RuntimeError(
            "MiniCPM KL-teacher training requires the repo infllm_v2 extension with "
            "infllmv2_attn_stage1_kl_teacher support. Rebuild the repo extension via "
            "`cd infllmv2_cuda_impl && pip install -e .`."
        )
    try:
        return infllmv2_attn_stage1_kl_teacher(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_v,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal_stride=causal_stride,
            query_offset=query_offset,
            causal=causal,
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "MiniCPM KL-teacher training requires the rebuilt repo infllm_v2 extension with "
            "the training-only stage1 KL-teacher op. Rebuild via "
            "`cd infllmv2_cuda_impl && pip install -e .`."
        ) from exc


def _pool_kl_target_alignment_1d(
    values: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    mode: str,
) -> torch.Tensor:
    """Pool finer KL targets onto the pred compressed grid."""
    if mode == "max":
        return F.max_pool1d(values, kernel_size=kernel_size, stride=stride, padding=padding)
    if mode == "mean":
        return F.avg_pool1d(values, kernel_size=kernel_size, stride=stride, padding=padding)
    raise ValueError(f"Unsupported kl_target_align_pooling={mode!r}")


def _align_probabilities_to_target_grid_1d(
    values: torch.Tensor,
    *,
    source_stride: int,
    target_stride: int,
    target_length: int,
    mode: str,
) -> torch.Tensor:
    """Align finer probability tensors onto a coarser compressed grid.

    Intentionally uses the legacy stride-based heuristic (`ratio + 1`, pad=1)
    instead of exact kernel-support alignment because that matched better
    training/eval accuracy for the deployed 32/16 routing grid.
    """
    if target_stride % source_stride != 0:
        raise ValueError(
            f"Cannot align source stride {source_stride} to target stride {target_stride}: "
            "target stride must be divisible by source stride."
        )
    ratio = target_stride // source_stride
    if ratio == 1:
        aligned = values
    else:
        aligned = _pool_kl_target_alignment_1d(
            values,
            kernel_size=ratio + 1,
            stride=ratio,
            padding=1,
            mode=mode,
        )
    pooled_len = aligned.shape[-1]
    if pooled_len > target_length:
        aligned = aligned[..., :target_length]
    elif pooled_len < target_length:
        aligned = F.pad(aligned, (0, target_length - pooled_len), value=0.0)
    return aligned

def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
    # Optional: pre-computed score tensor (for indexer scoring)
    precomputed_score: torch.Tensor = None,
    # Optional: override stride for max_pooling_1d_varlen (default: kernel_stride)
    # Use score_stride=1 when precomputed_score is at token level (e.g., InfiniGen)
    score_stride: int = None,
) -> torch.Tensor:
    """Compute compressed attention scores and return top-k block indices.
    
    Supports two modes:
    1. Standard: Compute scores using infllmv2_attn_stage1(q, k, k2)
    2. Precomputed: Use pre-computed score tensor (e.g., from an indexer q_future @ k.T)
    
    Args:
        q, k, k2: Query and compressed key tensors (used if precomputed_score is None)
        precomputed_score: Optional pre-computed score tensor [num_heads, total_q, total_k]
                          If provided, q/k/k2 are ignored for score computation
        ... (other args same as before)
    
    Returns:
        topk_idx: [num_heads, total_q, topk] top-k block indices
    """
    with torch.no_grad():
        device = q.device if precomputed_score is None else precomputed_score.device
        batch_size = cu_seqlens_q.shape[0] - 1
        
        # Check if it's prefilling stage
        is_prefilling = cache_lens is None or (cache_lens == 0).all().item()
        
        if is_prefilling:  # prefilling stage
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=device) 
            q_idx = torch.cat([
                (torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=device) + 
                 max_seqlen_q - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])) // block_size
                for i in range(batch_size)
            ], dim=0)  # shape: [total_q_len]
        else:  # decoding stage
            q_idx = cache_lens // block_size  # shape: [batch_size] = [total_q_len] in decoding
        
        # Compute or use pre-computed score
        if precomputed_score is not None:
            # Precomputed-score mode: use the provided score
            score = precomputed_score
        else:
            # Standard mode: compute score with CUDA kernel
            torch.cuda.nvtx.range_push("stage 1")
            score = _call_infllmv2_stage1_compat(
                q.contiguous(),
                k.contiguous(),
                k2.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_v=cu_seqlens_k2,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=is_prefilling
            )
            torch.cuda.nvtx.range_pop()  # stage 1
        
        score = score[:, :q_idx.shape[0], :]  # [num_heads, total_q_len, total_k]
        
        # Ensure score is in bf16 before max_pooling_1d_varlen (CUDA kernel requirement)
        if score.dtype != torch.bfloat16:
            score = score.to(torch.bfloat16)
        
        torch.cuda.nvtx.range_push("pooling")
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=score_stride if score_stride is not None else kernel_stride
        )  # shape: [num_heads, total_q_len, num_blocks]
        
        # get topk
        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)
        torch.cuda.nvtx.range_pop()  # pooling
        
    return topk_idx


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
    
    # Handle edge case: sequence too short for even one chunk (e.g., during decode)
    if max_seq_len < chunk_size:
        # Return empty indices and cu_seqlens with no compressed tokens
        empty_indices = torch.tensor([], dtype=torch.long, device=cu_seqlen.device)
        cu_seqlens_compressed = torch.zeros(len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device)
        return empty_indices, cu_seqlens_compressed
    
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
    chunk_indices = torch.arange(0, chunk_size, device=cu_seqlen.device)[None, :]  # [1, chunk_size]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, chunk_size]
    filtered_indices = filtered_indices.view(-1)  # Flatten to 1D indices
    
    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)  # Number of valid chunks per batch
    cu_seqlens_compressed = torch.zeros(len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device)
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    
    del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, chunk_indices
    return filtered_indices, cu_seqlens_compressed


def _unpad_one_tensor(hidden_states, attention_mask):
    """Unpad a single tensor using attention mask."""
    indices, cu_seqlens, max_seqlen_in_batch = _get_unpad_data(attention_mask)
    batch_size, seq_len = hidden_states.shape[:2]
    
    # Get the remaining dimensions
    remaining_dims = hidden_states.shape[2:]
    
    # Reshape to (batch_size * seq_len, *remaining_dims)
    reshaped_states = hidden_states.reshape(batch_size * seq_len, *remaining_dims)
    
    # Apply unpadding using indices
    unpadded_states = index_first_axis(reshaped_states, indices)
    
    return unpadded_states, indices, cu_seqlens, max_seqlen_in_batch


# =============================================================================
# REUSED FROM ORIGINAL HUGGINGFACE modeling_minicpm.py
# =============================================================================

def _get_unpad_data(attention_mask):
    """Get unpadding data for variable length sequences."""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with LongRoPE scaling for extended context."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, 
                 short_factor=None, long_factor=None, original_max_position_embeddings=None):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = (max_position_embeddings / self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)
        
        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype)
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistent=False)


_MINICPM41_128K_LONGROPE_FACTOR = [
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


def apply_minicpm41_128k_longrope(config):
    """Mutate config to use official MiniCPM4.1-8B 128K LongRoPE factors."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return config
    scaling_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if scaling_type != "longrope":
        return config

    rope_scaling["rope_type"] = "longrope"
    rope_scaling["long_factor"] = list(_MINICPM41_128K_LONGROPE_FACTOR)
    rope_scaling["short_factor"] = list(_MINICPM41_128K_LONGROPE_FACTOR)
    rope_scaling["original_max_position_embeddings"] = 65536
    config.rope_scaling = rope_scaling
    return config


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


def apply_rotary_pos_emb_q_only(q, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to query tensor only."""
    orig_dtype = q.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    return q_embed.to(dtype=orig_dtype)


class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (for GQA/MQA)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# =============================================================================
# COMPRESS KEY MODULE
# =============================================================================

class CompressK(torch.nn.Module):
    """Module for compressing key (K) representations using mean pooling."""
    
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """
        Args:
            head_num_k (int): Number of key attention heads.
            head_dim (int): Dimension of each attention head.
            kernel_size (int): Size of each chunk used for compression.
            kernel_stride (int, optional): Stride used when dividing input into chunks. Default is 16.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(self, k: torch.Tensor, cu_seqlens):
        """
        Forward pass for compressing the key (K) tensor.
        
        Args:
            k (torch.Tensor): Input key tensor of shape (total_seq_len, num_heads, head_dim).
            cu_seqlens (torch.Tensor): Cumulative sequence lengths for each sample in the batch.
            
        Returns:
            compress_k (torch.Tensor): Compressed key tensor.
            cu_seqlens_compressed (torch.Tensor): Updated cumulative sequence lengths after compression.
        """
        # Compute chunk-related metadata, with stride support
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )
        
        # Handle edge case: no valid chunks (e.g., during decode with short sequences)
        if filtered_k_indices.numel() == 0:
            # Return empty compressed tensor with correct shape
            compressed_k = k.new_empty(0, self.head_num_k, self.head_dim)
            return compressed_k, cu_seqlens_compressed
        
        # Extract filtered key vectors
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        
        # split
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size, self.kernel_size, 
            self.head_num_k, self.head_dim
        )  # [l, block_size, h, d]
        compressed_k = filtered_k.mean(dim=1)
        
        return compressed_k, cu_seqlens_compressed


# =============================================================================
# INDEXER SUPPORT FOR SPARSE ATTENTION
# =============================================================================

# =============================================================================
# INFLLM V2 CACHE CLASSES
# =============================================================================

class InfLLMv2CacheLayer(DynamicLayer):
    """Cache layer for InfLLM V2 sparse attention with compressed key storage."""
    
    def __init__(self):
        super().__init__()
        self.no_rope_keys = torch.tensor([], dtype=torch.float32)

        # Each slot stores: (cache_list, cu_seqlens, varlen_tensor, no_compress_cache)
        # Slots: compress_k, compress_k2, compress_k_kl_target, compress_k2_kl_target
        self.compress_k_cache = []
        self.cached_compressed_cu_seqlens = torch.tensor([], dtype=torch.int32)
        self.compress_k_cache_varlen = torch.tensor([], dtype=torch.float32)
        self.no_compress_k_cache = []

        self.compress_k2_cache = []
        self.cached_compressed_cu_seqlens2 = torch.tensor([], dtype=torch.int32)
        self.compress_k2_cache_varlen = torch.tensor([], dtype=torch.float32)
        self.no_compress_k2_cache = []

        self.compress_k_cache_kl_target = []
        self.cached_compressed_cu_seqlens_kl_target = torch.tensor([], dtype=torch.int32)
        self.compress_k_cache_varlen_kl_target = torch.tensor([], dtype=torch.float32)
        self.no_compress_k_cache_kl_target = []

        self.compress_k2_cache_kl_target = []
        self.cached_compressed_cu_seqlens2_kl_target = torch.tensor([], dtype=torch.int32)
        self.compress_k2_cache_varlen_kl_target = torch.tensor([], dtype=torch.float32)
        self.no_compress_k2_cache_kl_target = []

    def _infer_empty_tensor_shape(self, key_states):
        """Infer device/dtype/shape from key_states or KV cache for creating empty tensors."""
        device, dtype, num_kv_heads, head_dim = None, None, None, None
        keys = getattr(self, "keys", None)
        if isinstance(keys, torch.Tensor) and keys.numel() > 0 and keys.dim() == 4:
            device, dtype = keys.device, keys.dtype
            num_kv_heads, head_dim = keys.shape[1], keys.shape[-1]
        if device is None:
            for k in (key_states if isinstance(key_states, (list, tuple)) else [key_states]):
                if isinstance(k, torch.Tensor):
                    device, dtype = k.device, k.dtype
                    num_kv_heads, head_dim = k.shape[-2], k.shape[-1]
                    break
        if device is None:
            device = keys.device if isinstance(keys, torch.Tensor) else torch.device("cpu")
            dtype = keys.dtype if isinstance(keys, torch.Tensor) else torch.float32
            num_kv_heads, head_dim = 0, 0
        return device, dtype, num_kv_heads, head_dim

    def _update_compress_cache(self, key_states, cu_seqlens, cache_list, cu_attr, varlen_attr):
        """Generic compressed key cache update (used by all compress_k variants).
        
        Returns (varlen_tensor, cu_seqlens_tensor).
        """
        if len(cache_list) == 0:
            if cu_seqlens is not None:
                setattr(self, cu_attr, cu_seqlens.clone())
                setattr(self, varlen_attr, key_states)
                split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                cache_list[:] = list(torch.split(key_states, split_sizes))
            elif isinstance(key_states, (list, tuple)):
                batch_size = len(key_states)
                device, dtype, num_kv_heads, head_dim = self._infer_empty_tensor_shape(key_states)
                empty = torch.empty((0, num_kv_heads, head_dim), device=device, dtype=dtype)
                cache_list[:] = [(k if isinstance(k, torch.Tensor) else empty) for k in key_states]
                setattr(self, varlen_attr, torch.cat(cache_list, dim=0) if batch_size > 0 else empty)
                seq_lens = torch.tensor([t.shape[0] for t in cache_list], dtype=torch.int32, device=device)
                cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
                if batch_size > 0:
                    cu[1:] = torch.cumsum(seq_lens, dim=0)
                setattr(self, cu_attr, cu)
            else:
                if not isinstance(key_states, torch.Tensor):
                    raise TypeError(f"Unexpected key_states type: {type(key_states)}")
                device = key_states.device
                cu = torch.tensor([0, key_states.shape[0]], dtype=torch.int32, device=device)
                setattr(self, cu_attr, cu)
                setattr(self, varlen_attr, key_states)
                cache_list[:] = [key_states]
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    cache_list[index] = torch.cat([cache_list[index], k], dim=0)
            new_seq_lens = torch.tensor([t.shape[0] for t in cache_list], dtype=torch.int32)
            new_cumsum = torch.cumsum(new_seq_lens, dim=0, dtype=torch.int32)
            varlen = torch.cat(cache_list, dim=0)
            setattr(self, varlen_attr, varlen)
            setattr(self, cu_attr, torch.cat(
                [torch.tensor([0], dtype=torch.int32), new_cumsum]
            ).to(varlen.device))
        return getattr(self, varlen_attr), getattr(self, cu_attr)

    @staticmethod
    def _update_no_compress_cache(key_states, no_compress_cache, kernel_size, kernel_stride):
        """Generic uncompressed key buffer update for incremental compression."""
        k_chunk_list = []
        for index, k in enumerate(key_states):
            if len(no_compress_cache) <= index:
                no_compress_cache.append(k)
            else:
                no_compress_cache[index] = torch.cat([no_compress_cache[index], k], dim=0)
                current_len = no_compress_cache[index].shape[0]
                if current_len >= kernel_size:
                    k_chunk_list.append(no_compress_cache[index][:kernel_size])
                    no_compress_cache[index] = no_compress_cache[index][kernel_stride:]
                else:
                    k_chunk_list.append(None)
        return k_chunk_list

    def update_no_rope_key(self, key_states):
        """Update the no-rope key cache."""
        if self.no_rope_keys.numel() == 0:
            self.no_rope_keys = key_states
        else:
            self.no_rope_keys = torch.cat([self.no_rope_keys, key_states], dim=1)
        return self.no_rope_keys

    def update_compress_k(self, key_states, cu_seqlens=None):
        return self._update_compress_cache(
            key_states, cu_seqlens, self.compress_k_cache,
            "cached_compressed_cu_seqlens", "compress_k_cache_varlen")

    def update_compress_k_kl_target(self, key_states, cu_seqlens=None):
        return self._update_compress_cache(
            key_states, cu_seqlens, self.compress_k_cache_kl_target,
            "cached_compressed_cu_seqlens_kl_target", "compress_k_cache_varlen_kl_target")

    def update_compress_k2(self, key_states, cu_seqlens=None):
        return self._update_compress_cache(
            key_states, cu_seqlens, self.compress_k2_cache,
            "cached_compressed_cu_seqlens2", "compress_k2_cache_varlen")

    def update_compress_k2_kl_target(self, key_states, cu_seqlens=None):
        return self._update_compress_cache(
            key_states, cu_seqlens, self.compress_k2_cache_kl_target,
            "cached_compressed_cu_seqlens2_kl_target", "compress_k2_cache_varlen_kl_target")

    def update_no_compress_k(self, key_states, kernel_size=32, kernel_stride=16):
        return self._update_no_compress_cache(key_states, self.no_compress_k_cache, kernel_size, kernel_stride)

    def update_no_compress_k_kl_target(self, key_states, kernel_size=32, kernel_stride=16):
        return self._update_no_compress_cache(key_states, self.no_compress_k_cache_kl_target, kernel_size, kernel_stride)

    def update_no_compress_k2(self, key_states, kernel_size=128, kernel_stride=64):
        return self._update_no_compress_cache(key_states, self.no_compress_k2_cache, kernel_size, kernel_stride)

    def update_no_compress_k2_kl_target(self, key_states, kernel_size=128, kernel_stride=64):
        return self._update_no_compress_cache(key_states, self.no_compress_k2_cache_kl_target, kernel_size, kernel_stride)



class InfLLMv2Cache(DynamicCache):
    """Dynamic cache for InfLLM V2 with compressed key storage per layer."""
    
    def __init__(self, config=None, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self.layers = [InfLLMv2CacheLayer() for _ in range(num_hidden_layers)] if num_hidden_layers else []
        self._seen_tokens = 0
        
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def update_no_rope_key(self, key_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update_no_rope_key(key_states)

    def update_compress_k(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k(key_states, cu_seqlens)

    def update_no_compress_k(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None):
        return self.layers[layer_idx].update_no_compress_k(key_states, kernel_size, kernel_stride)

    def update_compress_k2(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k2(key_states, cu_seqlens)

    def update_no_compress_k2(self, key_states, layer_idx, kernel_size=128, kernel_stride=64, cache_kwargs=None):
        return self.layers[layer_idx].update_no_compress_k2(key_states, kernel_size, kernel_stride)

    # --- Optional KL-target compressed-K caches (second compression scale) ---
    def update_compress_k_kl_target(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k_kl_target(key_states, cu_seqlens)

    def update_no_compress_k_kl_target(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None):
        return self.layers[layer_idx].update_no_compress_k_kl_target(key_states, kernel_size, kernel_stride)

    def update_compress_k2_kl_target(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k2_kl_target(key_states, cu_seqlens)

    def update_no_compress_k2_kl_target(self, key_states, layer_idx, kernel_size=128, kernel_stride=64, cache_kwargs=None):
        return self.layers[layer_idx].update_no_compress_k2_kl_target(key_states, kernel_size, kernel_stride)

    def crop(self, max_length):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)


# =============================================================================
# BASE ATTENTION CLASS (from original HuggingFace)
# =============================================================================

class MiniCPMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # Support both 'type' and 'rope_type' keys for compatibility
            scaling_type = self.config.rope_scaling.get("rope_type", 
                           self.config.rope_scaling.get("type", "linear"))
            scaling_factor = self.config.rope_scaling.get("factor", None)
            
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor=self.config.rope_scaling["short_factor"],
                    long_factor=self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"]
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated.")

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, 'get_usable_length'):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            elif hasattr(past_key_value, 'get_seq_length'):
                kv_seq_len += past_key_value.get_seq_length()
        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class MiniCPMFlashAttention2(MiniCPMAttention):
    """MiniCPM flash attention module with proper unpadding support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated.")
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, 'get_usable_length'):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            elif hasattr(past_key_value, 'get_seq_length'):
                kv_seq_len += past_key_value.get_seq_length()
        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32. We will cast back in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """Flash Attention with proper unpadding for variable length sequences."""
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MiniCPMSdpaAttention(MiniCPMAttention):
    """MiniCPM attention using torch.nn.functional.scaled_dot_product_attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "MiniCPMModel is using MiniCPMSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` "
                "does not support `output_attentions=True`. Falling back to the manual attention implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, 'get_usable_length'):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            elif hasattr(past_key_value, 'get_seq_length'):
                kv_seq_len += past_key_value.get_seq_length()
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# =============================================================================
# INFLLM V2 INTEGRATION - Sparse Attention with q_future Training Support
# =============================================================================

class MiniCPMInfLLMv2Attention(MiniCPMAttention):
    """
    MiniCPM attention using InfLLM V2 kernels with q_future prediction training support.
    
    This module supports both:
    1. Inference with full InfLLM V2 sparse attention (dense_len switching, caching, etc.)
    2. Single-stage indexer training with q_future prediction (KL loss trains the indexer)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.config._attn_implementation == 'flash_attention_2', \
            'Only flash_attention_2 is supported for sparse attention'
        
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        
        # Get sparse config from model config, with defaults
        sparse_config = getattr(self.config, 'sparse_config', None) or {}
        self.sparse_config = sparse_config  # Store for later access
        
        # Sparse attention hyperparameters
        self.kernel_size = sparse_config.get('kernel_size', 32)
        self.kernel_stride = sparse_config.get('kernel_stride', 16)
        self.init_blocks = sparse_config.get('init_blocks', 1)
        self.block_size = sparse_config.get('block_size', 64)
        self.window_size = sparse_config.get('window_size', 2048)
        self.dense_len = sparse_config.get('dense_len', 8192)
        self.local_blocks = self.window_size // self.block_size
        self.topk = sparse_config.get('topk', 64) + (self.window_size // self.block_size)
        self.use_nope = sparse_config.get('use_nope', False)
        # Whether to use q_future (predicted query from indexer) for top-k selection during inference
        # True = use predicted q_future (indexer), False = use actual query (original mode)
        self.use_q_future_for_topk = sparse_config.get('use_q_future_for_topk', True)
        # Whether to use q_future only during decode phase (prefill uses actual query)
        self.use_q_future_decode_only = sparse_config.get('use_q_future_decode_only', False)

        # Optional: KL target scoring can use a different compression scale than pred scoring.
        # This is useful when you want a "finer" target distribution (smaller stride) and then
        # max-pool it down to match the pred distribution dimension before KL.
        _tr_ks = sparse_config.get('training_kernel_size', None)
        _tr_kst = sparse_config.get('training_kernel_stride', None)
        self.training_kernel_size = int(self.kernel_size if _tr_ks is None else _tr_ks)
        self.training_kernel_stride = int(self.kernel_stride if _tr_kst is None else _tr_kst)
        self.kl_target_align_pooling = str(sparse_config.get("kl_target_align_pooling", "max")).lower()
        if self.kl_target_align_pooling not in {"max", "mean"}:
            raise ValueError(f"Unsupported kl_target_align_pooling={self.kl_target_align_pooling!r}")
        self._kl_target_compress_differs = (
            self.training_kernel_size != self.kernel_size or
            self.training_kernel_stride != self.kernel_stride
        )
        # Key compressors for two-level compression
        self.compress_k = CompressK(
            self.num_key_value_heads, self.head_dim, 
            kernel_size=self.kernel_size, kernel_stride=self.kernel_stride
        )
        self.compress_k2 = CompressK(
            self.num_key_value_heads, self.head_dim,
            kernel_size=self.kernel_size * 4, kernel_stride=self.kernel_stride * 4
        )

        # KL-target key compressors (may share with main compressors if scale matches)
        if self._kl_target_compress_differs:
            self.compress_k_kl_target = CompressK(
                self.num_key_value_heads, self.head_dim,
                kernel_size=self.training_kernel_size, kernel_stride=self.training_kernel_stride
            )
            self.compress_k2_kl_target = CompressK(
                self.num_key_value_heads, self.head_dim,
                kernel_size=self.training_kernel_size * 4, kernel_stride=self.training_kernel_stride * 4
            )
        else:
            self.compress_k_kl_target = self.compress_k
            self.compress_k2_kl_target = self.compress_k2
        
        # Indexer config - integrated into attention module
        # Indexer starts with full attention heads per GQA group during training,
        # then gradually reduces to 1 head per group through head merging.
        # This progressive head reduction helps learn better representations.
        
        # Q projections for indexer (integrated into attention)
        # q_future_proj: predicts query for NEXT layer's scoring
        # q_curr_proj: predicts query for THIS layer's scoring (layer 0 only)
        create_indexer = sparse_config.get('create_indexer', True)
        
        if create_indexer:
            # One Q head per KV head for indexing (simple design)
            self.q_future_proj = nn.Linear(
                self.config.hidden_size, 
                self.num_key_value_heads * self.head_dim, 
                bias=False
            )
            self.q_future_proj._is_indexer = True  # Mark for Kaiming init
            # Layer 0 only: separate projection for this layer's scoring
            if self.layer_idx == 0:
                self.q_curr_proj = nn.Linear(
                    self.config.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=False
                )
                self.q_curr_proj._is_indexer = True  # Mark for Kaiming init
        else:
            self.q_future_proj = None
            self.q_curr_proj = None

        # InfiniGen state mirrors the NOSA path: warmup seeds the skewing matrix
        # once, while request-specific partial indices/cache are reset per call.
        self.infinigen_enabled = False
        self.infinigen_num_channels = 32
        self._ig_skewing_matrix = None
        self._ig_partial_indices = None
        self._ig_partial_key_cache = None
        self._ig_warmed_up = False
        self._ig_prev_hidden_states = None
        self._ig_attn_input = None
        self._ig_max_new_tokens = None
        self._path_infinigen = 0
        self._path_fallback = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        q_future_from_prev: Optional[torch.Tensor] = None,
        training_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[Tuple],
               Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with InfLLM V2 sparse attention and integrated indexer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask for padding
            position_ids: Position IDs for RoPE
            past_key_value: KV cache (InfLLMv2Cache for inference)
            output_attentions: Whether to return attention weights
            use_cache: Whether to use KV cache
            q_future_from_prev: Predicted query from previous layer [batch, seq, num_kv_heads, head_dim]
            training_stage: 0=Inference, 1=Training
            
        Returns:
            attn_output: Attention output
            attn_weights: None (not supported for sparse attention)
            past_key_value: Updated KV cache
            aux_scores: Tuple of (scores_pred, scores_target) for KL loss (training only)
            q_future: Predicted query for NEXT layer [batch, seq, num_kv_heads, head_dim]
            q_curr: Predicted query for THIS layer (layer 0 only) [batch, seq, num_kv_heads, head_dim]
            key_states_for_indexer: K for indexer with RoPE [batch, seq, num_kv_heads, head_dim]
        """
        if 'padding_mask' in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated.")
            attention_mask = kwargs.pop('padding_mask')
        
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        # Capture attention input for InfiniGen cross-layer speculation.
        if getattr(self, 'infinigen_enabled', False):
            self._ig_attn_input = hidden_states.detach()

        # 1. QKV Projection
        torch.cuda.nvtx.range_push("linear")
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Compute q_future (for next layer) and q_curr (for this layer, layer 0 only)
        # One Q head per KV head for simple block selection
        # q_future shape: [bsz, q_len, num_kv_heads, head_dim]
        q_future = None
        q_curr = None
        # Skip indexer projections entirely when InfiniGen is enabled.
        if self.q_future_proj is not None and not getattr(self, 'infinigen_enabled', False):
            # Detach hidden_states during training to prevent gradient flow through indexer
            indexer_input = hidden_states.detach() if training_stage == 1 else hidden_states
            q_future = self.q_future_proj(indexer_input)
            q_future = q_future.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            
            if getattr(self, "q_curr_proj", None) is not None:
                q_curr = self.q_curr_proj(indexer_input)
                q_curr = q_curr.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Save no-rope keys if use_nope is enabled
        if self.use_nope:
            query_states_no_rope = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            key_states_no_rope = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        torch.cuda.nvtx.range_pop()  # linear

        # 2. Apply RoPE
        torch.cuda.nvtx.range_push("rope")
        # Avoid device->host sync from `position_ids.max().item()` on long sequences.
        # The required rotary length is the current KV length (including cached KV if present).
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # Follow the same cache-length conventions used by other attention modules in this file.
            if isinstance(past_key_value, Cache):
                kv_seq_len += past_key_value.get_seq_length()
            elif hasattr(past_key_value, "get_seq_length"):
                kv_seq_len += past_key_value.get_seq_length()
        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Apply RoPE to q_future and q_curr
        # q_future shape: [batch, seq, num_kv_heads, head_dim]
        if q_future is not None:
            q_future_t = q_future.transpose(1, 2)  # [batch, num_kv_heads, seq, head_dim]
            q_future_t = apply_rotary_pos_emb_q_only(q_future_t, cos, sin, position_ids)
            q_future = q_future_t.transpose(1, 2)  # [batch, seq, num_kv_heads, head_dim]
        if q_curr is not None:
            q_curr_t = q_curr.transpose(1, 2)
            q_curr_t = apply_rotary_pos_emb_q_only(q_curr_t, cos, sin, position_ids)
            q_curr = q_curr_t.transpose(1, 2)

        if (getattr(self, 'infinigen_enabled', False)
                and getattr(self, '_ig_warmed_up', False)
                and q_len > 1
                and getattr(self, '_ig_skewing_matrix', None) is not None
                and getattr(self, '_ig_partial_key_cache', None) is None
                and getattr(self, '_ig_partial_indices', None) is None
                and hasattr(self, '_refresh_partial_indices_from_prefill')):
            self._refresh_partial_indices_from_prefill(query_states, key_states)

        # Compute K for indexer (before cache update adds old keys)
        # K per GQA group, matching indexer Q heads exactly (one Q head per K head)
        key_states_for_indexer = key_states.transpose(1, 2)  # [batch, seq, kv_heads, head_dim]
        torch.cuda.nvtx.range_pop()  # rope

        # 3. Update KV cache
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Transpose for flash attention format [batch, seq, heads, dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Update no-rope key cache if needed
        if self.use_nope and past_key_value is not None and isinstance(past_key_value, InfLLMv2Cache):
            key_states_no_rope = past_key_value.update_no_rope_key(key_states_no_rope, self.layer_idx)
            no_rope_param = {
                'key_states_no_rope': key_states_no_rope,
                'query_states_no_rope': query_states_no_rope,
            }
        else:
            no_rope_param = None

        dropout_rate = self.attention_dropout if self.training else 0.0
        kv_seq_len_current = key_states.shape[1]
        ig_token_scores = None

        if (q_len == 1
                and getattr(self, 'infinigen_enabled', False)
                and getattr(self, '_ig_warmed_up', False)
                and self.layer_idx > 0
                and getattr(self, '_ig_prev_hidden_states', None) is not None
                and getattr(self, '_ig_partial_indices', None) is not None
                and getattr(self, '_ig_partial_key_cache', None) is not None
                and _INFINIGEN_AVAILABLE):
            ig_token_scores = infinigen_token_scores_hf(
                self._ig_prev_hidden_states,
                self.q_proj.weight.data,
                self._ig_skewing_matrix,
                self._ig_partial_indices,
                self._ig_partial_key_cache[:, :kv_seq_len_current],
                cos, sin, position_ids,
                self.num_heads, self.num_key_value_heads, self.head_dim,
                self.infinigen_num_channels,
                apply_rotary_pos_emb,
            )

        if (getattr(self, 'infinigen_enabled', False)
                and getattr(self, '_ig_warmed_up', False)
                and getattr(self, '_ig_skewing_matrix', None) is not None
                and getattr(self, '_ig_partial_indices', None) is not None):
            if q_len > 1 and getattr(self, '_ig_partial_key_cache', None) is None:
                max_cache_len = kv_seq_len_current + int(getattr(self, '_ig_max_new_tokens', 0) or 0)
                self._ig_partial_key_cache = torch.zeros(
                    bsz, max_cache_len, self.num_key_value_heads, self.infinigen_num_channels,
                    dtype=key_states.dtype, device=key_states.device,
                )
            if getattr(self, '_ig_partial_key_cache', None) is not None:
                if q_len == 1:
                    new_k = key_states[:, -1, :, :]
                    infinigen_update_partial_cache(
                        self._ig_partial_key_cache, new_k, kv_seq_len_current - 1,
                        self._ig_skewing_matrix, self._ig_partial_indices,
                    )
                elif q_len > 1:
                    partial_k = infinigen_build_partial_cache(
                        key_states, self._ig_skewing_matrix, self._ig_partial_indices,
                    )
                    write_len = min(kv_seq_len_current, self._ig_partial_key_cache.shape[1])
                    self._ig_partial_key_cache[:, :write_len] = partial_k[:, :write_len]

        # Handle dtype casting
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f'The input hidden states seems to be silently casted in float32. We will cast back in {target_dtype}.'
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 4. Choose attention path based on sequence length and training stage
        aux_scores = (None, None)
        
        if training_stage not in (0, 1):
            raise ValueError(f"Unsupported training_stage={training_stage}. Expected 0 (inference) or 1 (training).")

        if (training_stage == 0
                and not getattr(self, 'infinigen_enabled', False)
                and kv_seq_len_current < self.dense_len):
            # Short sequences during inference: use dense flash attention
            attn_output = self._flash_attention_forward_dense(
                query_states, key_states, value_states, 
                attention_mask, q_len, dropout=dropout_rate
            )
        else:
            # Long sequences (inference) or training: use sparse attention
            attn_output, aux_scores = self._sparse_attention_forward(
                query_states, key_states, value_states,
                attention_mask, q_len, dropout=dropout_rate,
                no_rope_param=no_rope_param,
                past_key_value=past_key_value,
                q_future_from_prev=q_future_from_prev,
                q_curr=q_curr,
                key_states_for_indexer=key_states_for_indexer,
                training_stage=training_stage,
                ig_token_scores=ig_token_scores,
            )

        torch.cuda.nvtx.range_push("o linear")
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()  # o linear

        # Return includes q_future, q_curr, and key_states_for_indexer for decoder layer
        return attn_output, None, past_key_value, aux_scores, q_future, q_curr, key_states_for_indexer

    def _sparse_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor, 
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        dropout: float = 0.0,
        softmax_scale: float = None,
        no_rope_param: dict = None,
        past_key_value: Optional[Cache] = None,
        q_future_from_prev: Optional[torch.Tensor] = None,
        q_curr: Optional[torch.Tensor] = None,
        key_states_for_indexer: Optional[torch.Tensor] = None,
        training_stage: int = 0,
        ig_token_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """Sparse attention forward pass with integrated indexer support."""
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        aux_scores = (None, None)
        
        if attention_mask is None:
            raise ValueError('Need attention_mask for sparse attention')

        batch_size = query_states.shape[0]
        
        # KL-target compressed-K is only needed when KL loss is actually computed (training only).
        # Do NOT store extra KL-target compressed-K in KV cache during inference.
        q_pred_for_kl_pre = q_future_from_prev if q_future_from_prev is not None else q_curr
        enable_kl_target_cache = (
            (training_stage == 1) and
            (q_pred_for_kl_pre is not None) and
            (not _SKIP_KL_LOSS)
        )
        should_compute_kl = enable_kl_target_cache

        # Get compressed keys
        if past_key_value is not None and isinstance(past_key_value, InfLLMv2Cache):
            torch.cuda.nvtx.range_push("compress")
            compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2 = self.get_compress_k(
                key_states=key_states if not self.use_nope else no_rope_param['key_states_no_rope'],
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                enable_kl_target_cache=enable_kl_target_cache,
            )
            # Default: KL target uses the same compression as pred unless configured otherwise.
            compressed_k_kl_target = compressed_k
            compressed_cu_seqlens_kl_target = compressed_cu_seqlens
            compressed_k2_kl_target = compressed_k2
            compressed_cu_seqlens2_kl_target = compressed_cu_seqlens2
            if enable_kl_target_cache and self._kl_target_compress_differs:
                layer_cache = past_key_value.layers[self.layer_idx]
                if layer_cache.compress_k_cache_varlen_kl_target.numel() > 0:
                    compressed_k_kl_target = layer_cache.compress_k_cache_varlen_kl_target
                    compressed_cu_seqlens_kl_target = layer_cache.cached_compressed_cu_seqlens_kl_target
                if layer_cache.compress_k2_cache_varlen_kl_target.numel() > 0:
                    compressed_k2_kl_target = layer_cache.compress_k2_cache_varlen_kl_target
                    compressed_cu_seqlens2_kl_target = layer_cache.cached_compressed_cu_seqlens2_kl_target
            torch.cuda.nvtx.range_pop()  # compress

        # Unpad inputs
        query_states_unpad, key_states_unpad, value_states_unpad, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        # Handle no_rope query unpadding
        if no_rope_param is not None:
            if max_seqlen_in_batch_q == 1:
                no_rope_param['query_states_no_rope'] = no_rope_param['query_states_no_rope'].squeeze(1)
            else:
                no_rope_param['query_states_no_rope'], _, _, _ = _unpad_one_tensor(
                    no_rope_param['query_states_no_rope'], attention_mask=attention_mask
                )

        # Get compressed keys for prefill case (no cache)
        if past_key_value is None or not isinstance(past_key_value, InfLLMv2Cache):
            torch.cuda.nvtx.range_push("compress")
            compressed_k, compressed_cu_seqlens = self.compress_k(key_states_unpad, cu_seqlens_k)
            compressed_k2, compressed_cu_seqlens2 = self.compress_k2(key_states_unpad, cu_seqlens_k)
            compressed_k_kl_target = compressed_k
            compressed_cu_seqlens_kl_target = compressed_cu_seqlens
            compressed_k2_kl_target = compressed_k2
            compressed_cu_seqlens2_kl_target = compressed_cu_seqlens2
            if enable_kl_target_cache and self._kl_target_compress_differs:
                compressed_k_kl_target, compressed_cu_seqlens_kl_target = self.compress_k_kl_target(key_states_unpad, cu_seqlens_k)
                compressed_k2_kl_target, compressed_cu_seqlens2_kl_target = self.compress_k2_kl_target(key_states_unpad, cu_seqlens_k)
            torch.cuda.nvtx.range_pop()  # compress

        # The actual query for scoring (without RoPE if using nope mode).
        # Indexer uses q_future for inference top-k selection; training uses q_actual for block selection.
        q_actual = query_states_unpad if no_rope_param is None else no_rope_param['query_states_no_rope']
        
        # Compute cache_lens for decoding (critical for correct block selection)
        cache_lens = None
        if max_seqlen_in_batch_q == 1 and max_seqlen_in_batch_k > 1:  # decoding
            seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            cache_lens = seq_lens_k - 1
        
        # Compressed K for indexer: one K head per GQA group, matching indexer Q heads exactly
        # DETACH to prevent KL loss gradients flowing to model's K projection
        # KL loss should only train indexer (q_future_proj, q_curr_proj), not model weights
        compressed_k_for_indexer = compressed_k.detach()  # [total_k, num_kv_heads, head_dim]
        
        # Determine topk selection strategy:
        # - Inference: optionally use q_future (indexer) for top-k selection if configured
        # - Training: stage 1 uses teacher-guided block selection
        
        use_pred_scores_for_topk = False
        
        if training_stage == 0:
            # Inference: use q_future if configured
            # Layer 0: use q_curr (no previous layer's q_future)
            # Layer N>0: use q_future_from_prev
            q_pred_for_topk = q_future_from_prev if q_future_from_prev is not None else q_curr
            use_pred_scores_for_topk = (
                self.use_q_future_for_topk and
                q_pred_for_topk is not None and 
                compressed_k_for_indexer is not None and
                compressed_k_for_indexer.numel() > 0
            )
            # Handle decode_only mode: use q_actual for prefill, q_future for decode
            if self.use_q_future_decode_only:
                is_decode = cache_lens is not None
                use_pred_scores_for_topk = use_pred_scores_for_topk and is_decode
        elif training_stage == 1:
            # Training: use target scores (q_actual) for top-k selection
            use_pred_scores_for_topk = False
        
        # Stage 1 uses the aligned teacher distribution for routing.
        use_aligned_training_topk = training_stage == 1 and self._kl_target_compress_differs

        # InfiniGen token-level scores (overrides both pred and standard paths)
        if ig_token_scores is not None:
            # InfiniGen provides token-level scores (H_kv, B, seq_len).
            # Pool to block level using original cu_seqlens_k (not compressed) with stride=1.
            topk_idx = compressed_attention(
                q_actual,  # needed for q_idx computation
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                cu_seqlens_k,  # original (not compressed) seqlens
                cu_seqlens_k,  # not used when precomputed_score is set
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,  # original max_seqlen
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
                precomputed_score=ig_token_scores,
                score_stride=1,  # token-level scores, stride=1
            )
        elif use_aligned_training_topk:
            q_topk_unpad = q_actual
            if use_pred_scores_for_topk:
                q_topk_unpad = self._unpad_indexer_query(q_pred_for_topk, attention_mask)
            topk_idx = self._compute_training_aligned_topk_idx(
                q_topk_unpad,
                compressed_k_kl_target.detach(),
                cu_seqlens_q,
                compressed_cu_seqlens_kl_target,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                cache_lens,
                causal,
            )
        elif use_pred_scores_for_topk:
            # Indexer scoring: use predicted q for top-k block selection
            # (q_curr for layer 0, q_future_from_prev for layer N>0)
            topk_idx = self._indexer_compressed_attention(
                q_pred_for_topk,
                compressed_k_for_indexer,
                attention_mask,
                cu_seqlens_q,
                compressed_cu_seqlens,  # Same cu_seqlens as compressed_k
                max_seqlen_in_batch_q,
                cache_lens=cache_lens,
            )
        else:
            # Standard scoring with q_actual (target scores)
            compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
            max_seqlen_compressed = compressed_seqlens.max().item()
            topk_idx = compressed_attention(
                q_actual,
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                compressed_cu_seqlens2,
                max_seqlen_in_batch_q,
                max_seqlen_compressed,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens
            )

        # Store topk_idx for external block stats collection (inference profiling)
        if getattr(self, '_collect_block_stats', False):
            self._last_topk_idx = topk_idx.detach()

        # Compute KL loss scores for training
        # Skip KL loss during gradient checkpointing recompute to save memory
        # Layer 0: use q_curr (no previous layer's q_future)
        # Layer N>0: use q_future_from_prev
        q_pred_for_kl = q_pred_for_kl_pre
        
        if should_compute_kl:
            # topk_idx: [num_kv_heads, total_q, topk] from both paths:
            # - Standard: from infllmv2_attn_stage1 which handles GQA internally
            # - Pred-score: from _indexer_compressed_attention without expansion
            loss_topk_idx = topk_idx
            
            scores_pred, scores_target = self._compute_indexer_kl_loss(
                q_actual,
                q_pred_for_kl,  # q_curr for layer 0, q_future_from_prev for others
                compressed_k_kl_target,  # training-grid compressed K for TARGET
                compressed_k_for_indexer,
                cu_seqlens_q,
                compressed_cu_seqlens_kl_target,
                compressed_cu_seqlens,
                compressed_cu_seqlens,
                attention_mask,
                causal=causal,
                topk_idx=loss_topk_idx,
                compressed_k2=compressed_k2_kl_target,
                compressed_cu_seqlens2=compressed_cu_seqlens2_kl_target,
            )
            aux_scores = (scores_pred, scores_target)

        # Sparse attention with top-k block selection (reuse pre-computed topk_idx)
        torch.cuda.nvtx.range_push("stage 2")
        attn_output_unpad = self.sparse_forward(
            query_states_unpad,
            key_states_unpad,
            value_states_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            no_rope_param=no_rope_param,
            topk_idx=topk_idx,
        )
        torch.cuda.nvtx.range_pop()  # stage 2
        
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        return attn_output, aux_scores

    def get_compress_k(
        self,
        key_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: InfLLMv2Cache,
        enable_kl_target_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get compressed key states, handling both prefilling and decoding."""
        # Initialize compressed caches the first time we need them.
        # Do NOT gate this on `dense_len`: reaching this function already means we're in the sparse path.
        is_prefilling = not past_key_value.layers[self.layer_idx].compress_k_cache
        
        if is_prefilling:
            unpadded_key_states, indices, cu_seqlens, max_seqlen_in_batch = _unpad_one_tensor(
                key_states, attention_mask=attention_mask
            )
            
            # Compress the keys (two levels)
            compressed_k, compressed_cu_seqlens = self.compress_k(unpadded_key_states, cu_seqlens)
            compressed_k2, compressed_cu_seqlens2 = self.compress_k2(unpadded_key_states, cu_seqlens)
            
            # Update cache
            past_key_value.update_compress_k(compressed_k, self.layer_idx, compressed_cu_seqlens)
            past_key_value.update_compress_k2(compressed_k2, self.layer_idx, compressed_cu_seqlens2)

            # Optional: KL-target compressed keys (different scale)
            if enable_kl_target_cache and self._kl_target_compress_differs:
                compressed_k_t, compressed_cu_seqlens_t = self.compress_k_kl_target(unpadded_key_states, cu_seqlens)
                compressed_k2_t, compressed_cu_seqlens2_t = self.compress_k2_kl_target(unpadded_key_states, cu_seqlens)
                past_key_value.update_compress_k_kl_target(compressed_k_t, self.layer_idx, compressed_cu_seqlens_t)
                past_key_value.update_compress_k2_kl_target(compressed_k2_t, self.layer_idx, compressed_cu_seqlens2_t)
            
            # Update no_compress_k buffer
            no_compress_k_list = []
            for i in range(len(compressed_cu_seqlens) - 1):
                no_compress_k_start = (compressed_cu_seqlens[i + 1] - compressed_cu_seqlens[i]) * self.kernel_stride
                no_compress_k_list.append(
                    unpadded_key_states[cu_seqlens[i] + no_compress_k_start:cu_seqlens[i + 1]].clone()
                )
            past_key_value.update_no_compress_k(
                no_compress_k_list, self.layer_idx, 
                kernel_stride=self.kernel_stride, kernel_size=self.kernel_size
            )

            if enable_kl_target_cache and self._kl_target_compress_differs:
                no_compress_k_list_t = []
                # Use KL-target compressed cu_seqlens to compute the remaining uncompressed tail.
                compressed_cu_seqlens_t = past_key_value.layers[self.layer_idx].cached_compressed_cu_seqlens_kl_target
                for i in range(len(compressed_cu_seqlens_t) - 1):
                    no_compress_k_start_t = (compressed_cu_seqlens_t[i + 1] - compressed_cu_seqlens_t[i]) * self.training_kernel_stride
                    no_compress_k_list_t.append(
                        unpadded_key_states[cu_seqlens[i] + no_compress_k_start_t:cu_seqlens[i + 1]].clone()
                    )
                past_key_value.update_no_compress_k_kl_target(
                    no_compress_k_list_t, self.layer_idx,
                    kernel_stride=self.training_kernel_stride, kernel_size=self.training_kernel_size
                )
            
            # Update no_compress_k2 buffer
            no_compress_k2_list = []
            for i in range(len(compressed_cu_seqlens2) - 1):
                no_compress_k2_start = (compressed_cu_seqlens2[i + 1] - compressed_cu_seqlens2[i]) * self.kernel_stride * 4
                no_compress_k2_list.append(
                    unpadded_key_states[cu_seqlens[i] + no_compress_k2_start:cu_seqlens[i + 1]].clone()
                )
            past_key_value.update_no_compress_k2(
                no_compress_k2_list, self.layer_idx,
                kernel_stride=self.kernel_stride * 4, kernel_size=self.kernel_size * 4
            )

            if enable_kl_target_cache and self._kl_target_compress_differs:
                no_compress_k2_list_t = []
                compressed_cu_seqlens2_t = past_key_value.layers[self.layer_idx].cached_compressed_cu_seqlens2_kl_target
                for i in range(len(compressed_cu_seqlens2_t) - 1):
                    no_compress_k2_start_t = (compressed_cu_seqlens2_t[i + 1] - compressed_cu_seqlens2_t[i]) * self.training_kernel_stride * 4
                    no_compress_k2_list_t.append(
                        unpadded_key_states[cu_seqlens[i] + no_compress_k2_start_t:cu_seqlens[i + 1]].clone()
                    )
                past_key_value.update_no_compress_k2_kl_target(
                    no_compress_k2_list_t, self.layer_idx,
                    kernel_stride=self.training_kernel_stride * 4, kernel_size=self.training_kernel_size * 4
                )
        else:
            # Decode: incremental update
            batch_size = key_states.shape[0]
            key_states_split = list(torch.split(
                key_states[:, -1:].squeeze(1), [1] * batch_size, dim=0
            ))
            
            # Update compress_k
            no_compress_k_list = past_key_value.update_no_compress_k(
                key_states_split, self.layer_idx,
                kernel_stride=self.kernel_stride, kernel_size=self.kernel_size
            )
            new_compressed_k_list = []
            for no_compress_k in no_compress_k_list:
                if no_compress_k is not None:
                    new_compressed_k = no_compress_k.mean(dim=0, keepdim=True)
                    new_compressed_k_list.append(new_compressed_k)
                else:
                    new_compressed_k_list.append(None)
            compressed_k, compressed_cu_seqlens = past_key_value.update_compress_k(
                new_compressed_k_list, self.layer_idx
            )

            if enable_kl_target_cache and self._kl_target_compress_differs:
                no_compress_k_list_t = past_key_value.update_no_compress_k_kl_target(
                    key_states_split, self.layer_idx,
                    kernel_stride=self.training_kernel_stride, kernel_size=self.training_kernel_size
                )
                new_compressed_k_list_t = []
                for no_compress_k in no_compress_k_list_t:
                    if no_compress_k is not None:
                        new_compressed_k = no_compress_k.mean(dim=0, keepdim=True)
                        new_compressed_k_list_t.append(new_compressed_k)
                    else:
                        new_compressed_k_list_t.append(None)
                past_key_value.update_compress_k_kl_target(new_compressed_k_list_t, self.layer_idx)
            
            # Update compress_k2
            no_compress_k2_list = past_key_value.update_no_compress_k2(
                key_states_split, self.layer_idx,
                kernel_stride=self.kernel_stride * 4, kernel_size=self.kernel_size * 4
            )
            new_compressed_k2_list = []
            for no_compress_k2 in no_compress_k2_list:
                if no_compress_k2 is not None:
                    new_compressed_k2 = no_compress_k2.mean(dim=0, keepdim=True)
                    new_compressed_k2_list.append(new_compressed_k2)
                else:
                    new_compressed_k2_list.append(None)
            compressed_k2, compressed_cu_seqlens2 = past_key_value.update_compress_k2(
                new_compressed_k2_list, self.layer_idx
            )

            if enable_kl_target_cache and self._kl_target_compress_differs:
                no_compress_k2_list_t = past_key_value.update_no_compress_k2_kl_target(
                    key_states_split, self.layer_idx,
                    kernel_stride=self.training_kernel_stride * 4, kernel_size=self.training_kernel_size * 4
                )
                new_compressed_k2_list_t = []
                for no_compress_k2 in no_compress_k2_list_t:
                    if no_compress_k2 is not None:
                        new_compressed_k2 = no_compress_k2.mean(dim=0, keepdim=True)
                        new_compressed_k2_list_t.append(new_compressed_k2)
                    else:
                        new_compressed_k2_list_t.append(None)
                past_key_value.update_compress_k2_kl_target(new_compressed_k2_list_t, self.layer_idx)
        
        return compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2

    def _get_indexer_scale(self) -> float:
        """Get softmax scaling factor.
        
        Returns:
            softmax_scale: 1/sqrt(head_dim) for Q@K variance control
        """
        return 1.0 / math.sqrt(self.head_dim)
    
    def _compute_indexer_block_scores(
        self,
        q_future: torch.Tensor,
        k_compressed: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute Q @ K scores for inference top-k block selection.
        
        One indexer head per KV group, directly computing raw scores.
        
        Args:
            q_future: [batch, seq, num_kv_heads, head_dim] predicted query
            k_compressed: [total_k, num_kv_heads, head_dim] compressed keys
            attention_mask: Attention mask (used only for prefill unpadding)
            cu_seqlens_q: Cumulative query sequence lengths [batch_size + 1]
            cu_seqlens_k: Cumulative compressed key sequence lengths [batch_size + 1]
            
        Returns:
            score: [num_kv_heads, total_q, max_seqlen_k] raw scores (one head per KV group)
        """
        num_kv_heads = self.num_key_value_heads
        batch_size = q_future.shape[0]
        seq_len_q = q_future.shape[1]
        head_dim = self.head_dim
        device = q_future.device
        dtype = q_future.dtype
        is_decode = seq_len_q == 1
        
        # Get K dimensions
        k_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_k = k_seqlens.max().item()
        
        if is_decode:
            # === DECODE MODE: Fully vectorized using batched einsum ===
            # q_future: [batch, 1, num_kv_heads, head_dim]
            q = q_future.squeeze(1)  # [batch, num_kv_heads, head_dim]
            
            # Pad K for batched computation
            k_padded = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, 
                                   device=device, dtype=dtype)
            for b in range(batch_size):
                k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
                k_len = k_end - k_start
                if k_len > 0:
                    k_padded[b, :k_len] = k_compressed[k_start:k_end]
            
            # Compute Q @ K scores: [batch, num_kv_heads, max_k]
            # q: [batch, num_kv_heads, head_dim]
            # k_padded: [batch, max_k, num_kv_heads, head_dim]
            k_t = k_padded.permute(0, 2, 3, 1)  # [batch, num_kv_heads, head_dim, max_k]
            
            # Use bf16 Tensor Cores with FP32 accumulation (like Flash Attention)
            # allow_bf16_reduced_precision_reduction=False forces FP32 accumulation
            q_bmm = q.view(batch_size * num_kv_heads, 1, head_dim)  # [batch*heads, 1, head_dim]
            k_bmm = k_t.reshape(batch_size * num_kv_heads, head_dim, max_seqlen_k)  # [batch*heads, head_dim, max_k]
            prev_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            batch_scores = torch.bmm(q_bmm, k_bmm).view(batch_size, num_kv_heads, max_seqlen_k)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev_reduced
            
            score = batch_scores.permute(1, 0, 2)  # [num_kv_heads, batch, max_k]
            
        else:
            # === PREFILL MODE: Per-batch computation ===
            indices_q, _, _ = _get_unpad_data(attention_mask)
            # Flatten for unpadding: [batch * seq, num_kv_heads, head_dim]
            q_future_flat = q_future.reshape(-1, num_kv_heads, head_dim)
            q_unpad = index_first_axis(q_future_flat, indices_q)  # [total_q, num_kv_heads, head_dim]
            total_q = q_unpad.shape[0]
            
            # Initialize with -inf for padded positions
            score = torch.full((num_kv_heads, total_q, max_seqlen_k), float('-inf'), 
                               device=device, dtype=dtype)
            
            for b in range(batch_size):
                q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
                k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
                q_len, k_len = q_end - q_start, k_end - k_start
                
                if q_len == 0 or k_len == 0:
                    continue
                
                q_b = q_unpad[q_start:q_end]  # [q_len, num_kv_heads, head_dim]
                k_b = k_compressed[k_start:k_end].permute(1, 0, 2)  # [num_kv_heads, k_len, head_dim]
                
                # Compute Q @ K scores: [num_kv_heads, q_len, k_len]
                # Use bf16 Tensor Cores with FP32 accumulation (like Flash Attention)
                q_b_t = q_b.permute(1, 0, 2)  # [num_kv_heads, q_len, head_dim]
                k_b_t = k_b.transpose(1, 2)  # [num_kv_heads, head_dim, k_len]
                prev_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                raw_scores = torch.bmm(q_b_t, k_b_t)  # [num_kv_heads, q_len, k_len]
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev_reduced
                
                # Apply causal mask
                q_positions = torch.arange(q_len, device=device).unsqueeze(1)
                k_positions = torch.arange(k_len, device=device).unsqueeze(0) * self.kernel_stride
                causal_mask = k_positions > q_positions  # [q_len, k_len]
                raw_scores = raw_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
                
                score[:, q_start:q_end, :k_len] = raw_scores
        
        return score

    def _indexer_compressed_attention(
        self,
        q_future: torch.Tensor,
        k_compressed: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        cache_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Indexer compressed attention for inference top-k block selection.
        
        Computes indexer scores and calls unified compressed_attention with precomputed_score.
        
        Note: Caller must ensure k_compressed is not empty (use_pred_scores_for_topk guard).
        
        Args:
            q_future: [batch, seq, num_kv_heads, head_dim] predicted query
            k_compressed: [total_k, num_kv_heads, head_dim] compressed keys
            attention_mask: Attention mask for unpadding
            cu_seqlens_q: Cumulative query sequence lengths
            cu_seqlens_k: Cumulative compressed key sequence lengths
            max_seqlen_q: Maximum query sequence length
            cache_lens: Cache lengths for decode mode
            
        Returns:
            topk_idx: [num_heads, total_q, topk] top-k block indices
        """
        # Compute indexer block scores for top-k selection
        # Returns shape: [num_kv_heads, total_q, max_seqlen_k]
        torch.cuda.nvtx.range_push("stage 1")
        score = self._compute_indexer_block_scores(
            q_future, k_compressed, attention_mask,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k
        )
        torch.cuda.nvtx.range_pop()  # stage 1
        
        # Get max_seqlen_k from cu_seqlens_k (matches score's last dimension)
        k_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_k = k_seqlens.max().item()
        
        # Keep score at num_kv_heads (NOT expanded to num_heads) for memory efficiency
        # This reduces memory by num_key_value_groups times and avoids int32 overflow
        # in max_pooling_1d_varlen for 64k+ sequences
        topk_idx = compressed_attention(
            q=None,  # Not used when precomputed_score is provided
            k=None,
            k2=None,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
            block_size=self.block_size,
            topk=self.topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_k2=None,  # Not used for indexer inference
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens,
            precomputed_score=score,
        )
        
        # topk_idx: [num_kv_heads, total_q, topk]
        # Keep as-is - both sparse attention kernel and KL loss work with num_kv_heads
        # (sparse kernel handles GQA expansion internally, KL loss expects num_kv_heads)
        return topk_idx

    def sparse_forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_in_batch_q: int,
        max_seqlen_in_batch_k: int,
        topk_idx: torch.Tensor,
        no_rope_param: dict = None,
    ) -> torch.Tensor:
        """Execute sparse attention with pre-computed top-k block indices."""
        topk_attn_output = infllmv2_attn_varlen_func(
            query_layer,
            key_layer,
            value_layer,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            dropout_p=0.0,
            deterministic=False,
            softmax_scale=None,
            causal=max_seqlen_in_batch_q != 1,
            return_attn_probs=False,
            topk_idx=topk_idx
        )
        
        return topk_attn_output

    def _flash_attention_forward_dense(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        dropout: float = 0.0,
        softmax_scale: float = None,
    ) -> torch.Tensor:
        """Dense flash attention for short sequences."""
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            
            attn_output_unpad = flash_attn_varlen_func(
                query_states, key_states, value_states,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout, softmax_scale=softmax_scale, causal=causal,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, 
                softmax_scale=softmax_scale, causal=causal
            )
        return attn_output

    def _upad_input(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
    ):
        """Unpad inputs for variable length sequences."""
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer, key_layer, value_layer, indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def _unpad_indexer_query(self, q_states: Optional[torch.Tensor], attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        if q_states is None:
            return None
        if q_states.shape[1] == 1:
            return q_states.squeeze(1)
        return _unpad_one_tensor(q_states, attention_mask)[0]

    def _build_aligned_training_probs(
        self,
        q_unpad: torch.Tensor,
        compressed_k_train: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens_train: torch.Tensor,
        compressed_cu_seqlens_default: torch.Tensor,
        max_seqlen_q: int,
        cache_lens: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        """Compute training-grid stage1 probabilities and align them to the default grid."""
        num_kv_heads = self.num_key_value_heads
        total_q = int(cu_seqlens_q[-1].item())
        default_seqlens = compressed_cu_seqlens_default[1:] - compressed_cu_seqlens_default[:-1]
        max_default_len = int(default_seqlens.max().item())
        aligned = torch.zeros(
            (num_kv_heads, total_q, max_default_len),
            device=q_unpad.device,
            dtype=q_unpad.dtype,
        )

        q_seqlens = cu_seqlens_q.detach().to("cpu").tolist()
        train_seqlens = compressed_cu_seqlens_train.detach().to("cpu").tolist()
        default_seqlens_list = compressed_cu_seqlens_default.detach().to("cpu").tolist()
        cu_q_buf = torch.empty((2,), device=q_unpad.device, dtype=torch.int32)
        cu_k_buf = torch.empty((2,), device=q_unpad.device, dtype=torch.int32)

        for b in range(len(q_seqlens) - 1):
            q_start, q_end = q_seqlens[b], q_seqlens[b + 1]
            k_start, k_end = train_seqlens[b], train_seqlens[b + 1]
            default_len = default_seqlens_list[b + 1] - default_seqlens_list[b]
            q_len = q_end - q_start
            k_len = k_end - k_start
            if q_len == 0 or k_len == 0 or default_len == 0:
                continue

            q_batch = q_unpad[q_start:q_end].contiguous()
            k_batch = compressed_k_train[k_start:k_end].contiguous()
            cu_q_buf[0] = 0
            cu_q_buf[1] = q_len
            cu_k_buf[0] = 0
            cu_k_buf[1] = k_len
            query_offset = 0
            if cache_lens is not None and q_len == 1:
                query_offset = int(cache_lens[b].item())
            probs = _call_infllmv2_stage1_kl_teacher_compat(
                q_batch,
                k_batch,
                k_batch,
                cu_seqlens_q=cu_q_buf,
                cu_seqlens_k=cu_k_buf,
                cu_seqlens_v=cu_k_buf,
                max_seqlen_q=q_len,
                max_seqlen_k=k_len,
                causal=causal,
                causal_stride=self.training_kernel_stride,
                query_offset=query_offset,
            )
            probs = probs[:, :q_len, :k_len].contiguous()
            probs = _align_probabilities_to_target_grid_1d(
                probs.reshape(-1, 1, k_len),
                source_stride=self.training_kernel_stride,
                target_stride=self.kernel_stride,
                target_length=default_len,
                mode=self.kl_target_align_pooling,
            ).reshape(num_kv_heads, q_len, default_len)
            aligned[:, q_start:q_end, :default_len] = probs

        return aligned

    def _compute_training_aligned_topk_idx(
        self,
        q_unpad: torch.Tensor,
        compressed_k_train: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens_train: torch.Tensor,
        compressed_cu_seqlens_default: torch.Tensor,
        max_seqlen_q: int,
        cache_lens: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        aligned_probs = self._build_aligned_training_probs(
            q_unpad,
            compressed_k_train,
            cu_seqlens_q,
            compressed_cu_seqlens_train,
            compressed_cu_seqlens_default,
            max_seqlen_q,
            cache_lens,
            causal,
        )
        default_seqlens = compressed_cu_seqlens_default[1:] - compressed_cu_seqlens_default[:-1]
        max_default_len = int(default_seqlens.max().item())
        return compressed_attention(
            q=None,
            k=None,
            k2=None,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
            block_size=self.block_size,
            topk=self.topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=compressed_cu_seqlens_default,
            cu_seqlens_k2=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_default_len,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens,
            precomputed_score=aligned_probs,
            score_stride=self.kernel_stride,
        )

    def _compute_indexer_kl_loss(
        self,
        query_states_unpad: torch.Tensor,
        q_future: torch.Tensor,
        compressed_k_attn: torch.Tensor,
        compressed_k_index: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens_attn: torch.Tensor,
        compressed_cu_seqlens_index: torch.Tensor,
        compressed_cu_seqlens_default: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        causal: bool = True,
        topk_idx: torch.Tensor = None,
        compressed_k2: torch.Tensor = None,
        compressed_cu_seqlens2: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        Compute KL divergence loss for training.
        
        Target: softmax(Q @ K * scale) per head → sum within group
        Pred: softmax(q_future @ k_index * scale) - one head per KV group
        
        Init + local blocks are excluded from topk selection in the KL loss.
        
        Uses infllmv2_attn_stage1 for the native sparse grid, and a separate
        training-only KL-teacher stage1 kernel for mismatched KL-target compression.
        
        Args:
            query_states_unpad: Actual query states [total_q, num_heads, head_dim]
            q_future: Predicted query from indexer [batch, seq, num_kv_heads, head_dim]
            compressed_k_attn: Attention's compressed key [total_k_attn, num_kv_heads, head_dim]
            compressed_k_index: Indexer's compressed key [total_k_index, num_kv_heads, head_dim]
            cu_seqlens_q: Cumulative sequence lengths for queries
            compressed_cu_seqlens_attn: Cumulative lengths for attention's compressed K
            compressed_cu_seqlens_index: Cumulative lengths for indexer's compressed K
            attention_mask: Attention mask for unpadding
            causal: Whether to apply causal masking
            topk_idx: Optional top-k block indices for filtering
            compressed_k2: Second-level compressed key for infllmv2_attn_stage1
            compressed_cu_seqlens2: Cumulative lengths for compressed_k2
            
        Returns:
            kl_loss: Pre-computed KL loss (scalar tensor)
            "precomputed": Signal to use loss directly
        """
        batch_size = len(cu_seqlens_q) - 1
        device = query_states_unpad.device
        num_kv_heads = self.num_key_value_heads
        scale_index = self._get_indexer_scale()

        # Small tensor caches to avoid repeated allocations in the per-batch loop.
        # These are per-call (not global) to keep memory bounded and semantics simple.
        _arange_cache: Dict[Tuple[str, int, torch.dtype], torch.Tensor] = {}

        def _get_arange(length: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            key = (str(device), int(length), dtype)
            t = _arange_cache.get(key)
            if t is None:
                t = torch.arange(length, device=device, dtype=dtype)
                _arange_cache[key] = t
            return t

        def _compute_kl_with_topk_chunk(
            topk_batch: torch.Tensor,
            probs_t: torch.Tensor,
            probs_p: torch.Tensor,
            *,
            q_offset: int,
            k_len_t: int,
            k_len_p: int,
        ) -> torch.Tensor:
            q_len_local = probs_t.shape[1]
            q_block_positions = (
                _get_arange(q_len_local, dtype=torch.int64, device=device) + q_offset
            ) // self.block_size
            q_block_positions = q_block_positions.view(1, q_len_local, 1)
            local_start = (q_block_positions - self.local_blocks).clamp(min=0)
            local_end = q_block_positions

            valid = topk_batch >= 0
            safe_topk = topk_batch.clamp(min=0).to(torch.int64)
            init_mask = safe_topk >= self.init_blocks
            local_mask = (safe_topk < local_start) | (safe_topk > local_end)
            valid_mask = valid & init_mask & local_mask

            topk_k = topk_batch.shape[-1]
            key_offsets = _get_arange(keys_per_block, dtype=torch.int64, device=device)
            expanded_idx = safe_topk.unsqueeze(-1) * keys_per_block + key_offsets
            expanded_idx = expanded_idx.reshape(num_kv_heads, q_len_local, topk_k * keys_per_block)
            idx_t = expanded_idx.clamp(min=0, max=k_len_t - 1)
            idx_p = expanded_idx.clamp(min=0, max=k_len_p - 1)

            valid_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, -1, keys_per_block)
            valid_expanded = valid_expanded.reshape(num_kv_heads, q_len_local, topk_k * keys_per_block)

            with torch.no_grad():
                pt_topk = torch.gather(probs_t, dim=2, index=idx_t.long())
                pt_topk = pt_topk.masked_fill(~valid_expanded, 0.0)
                rest_t = probs_t.sum(dim=-1, keepdim=True) - pt_topk.sum(dim=-1, keepdim=True)
                rest_t = rest_t.clamp_min(0.0)
                pt_final = torch.cat([pt_topk, rest_t], dim=-1)
                target_c = pt_final.clamp_min(1e-8)
                target_c = target_c / target_c.sum(dim=-1, keepdim=True)

            pp_topk = torch.gather(probs_p, dim=2, index=idx_p.long())
            pp_topk = pp_topk.masked_fill(~valid_expanded, 0.0)
            rest_p = probs_p.sum(dim=-1, keepdim=True) - pp_topk.sum(dim=-1, keepdim=True)
            rest_p = rest_p.clamp_min(0.0)
            pp_final = torch.cat([pp_topk, rest_p], dim=-1)
            pred_c = pp_final.clamp_min(1e-8)
            pred_c = pred_c / pred_c.sum(dim=-1, keepdim=True)
            return F.kl_div(pred_c.log(), target_c, reduction='sum', log_target=False)

        # Reusable 2-entry cu_seqlens buffers for the stage1 kernel call.
        cu_q_buf = torch.empty((2,), device=device, dtype=torch.int32)
        cu_k_buf = torch.empty((2,), device=device, dtype=torch.int32)
        cu_k2_buf = torch.empty((2,), device=device, dtype=torch.int32)
        
        use_topk = topk_idx is not None
        keys_per_block = self.block_size // self.kernel_stride if use_topk else None
        
        # Unpad q_future
        seq_len = q_future.shape[1]
        if seq_len == 1:
            q_future_unpad = q_future.squeeze(1)
        else:
            q_future_unpad, _, _, _ = _unpad_one_tensor(q_future, attention_mask)
        
        all_kl_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        all_num_distributions = 0
        
        if compressed_k2 is None or compressed_cu_seqlens2 is None:
            compressed_k2 = compressed_k_attn
            cu_seqlens_k2 = compressed_cu_seqlens_attn
        else:
            cu_seqlens_k2 = compressed_cu_seqlens2

        # Convert small cu_seqlens tensors to python lists once (avoids per-iteration .item() syncs).
        cu_seqlens_q_list = cu_seqlens_q.detach().to("cpu").tolist()
        cu_seqlens_attn_list = compressed_cu_seqlens_attn.detach().to("cpu").tolist()
        cu_seqlens_index_list = compressed_cu_seqlens_index.detach().to("cpu").tolist()
        cu_seqlens_k2_list = cu_seqlens_k2.detach().to("cpu").tolist()
        cu_seqlens_default_list = None
        if self._kl_target_compress_differs:
            if compressed_cu_seqlens_default is None:
                raise ValueError("compressed_cu_seqlens_default is required when KL training grid differs from default.")
            cu_seqlens_default_list = compressed_cu_seqlens_default.detach().to("cpu").tolist()
        
        for b in range(batch_size):
            q_start, q_end = cu_seqlens_q_list[b], cu_seqlens_q_list[b + 1]
            q_len = q_end - q_start
            k_attn_start, k_attn_end = cu_seqlens_attn_list[b], cu_seqlens_attn_list[b + 1]
            k_attn_len = k_attn_end - k_attn_start
            k_index_start, k_index_end = cu_seqlens_index_list[b], cu_seqlens_index_list[b + 1]
            k_index_len = k_index_end - k_index_start
            k2_start, k2_end = cu_seqlens_k2_list[b], cu_seqlens_k2_list[b + 1]
            
            if q_len == 0 or k_attn_len == 0 or k_index_len == 0:
                continue
            
            q_pred = q_future_unpad[q_start:q_end]
            k_index = compressed_k_index[k_index_start:k_index_end]
            k_index_t = k_index.permute(1, 2, 0).contiguous()
            target_stride = self.training_kernel_stride if self._kl_target_compress_differs else self.kernel_stride
            pred_stride = self.kernel_stride
            
            # Build causal mask for PRED
            causal_mask_index = None
            if causal:
                q_positions = _get_arange(q_len, dtype=torch.int64, device=device).unsqueeze(1)
                k_positions_index = _get_arange(k_index_len, dtype=torch.int64, device=device) * pred_stride
                causal_mask_index = k_positions_index.unsqueeze(0) > q_positions

            if self._kl_target_compress_differs:
                k_default_len = cu_seqlens_default_list[b + 1] - cu_seqlens_default_list[b]
                if k_default_len == 0:
                    continue
                with torch.no_grad():
                    q_batch = query_states_unpad[q_start:q_end].contiguous()
                    k_batch = compressed_k_attn[k_attn_start:k_attn_end].contiguous()
                q_pred_t = q_pred.permute(1, 0, 2).contiguous()

                with torch.no_grad():
                    cu_q_buf[0] = 0
                    cu_q_buf[1] = q_len
                    cu_k_buf[0] = 0
                    cu_k_buf[1] = k_attn_len
                    probs_target = _call_infllmv2_stage1_kl_teacher_compat(
                        q_batch,
                        k_batch,
                        k_batch,
                        cu_seqlens_q=cu_q_buf,
                        cu_seqlens_k=cu_k_buf,
                        cu_seqlens_v=cu_k_buf,
                        max_seqlen_q=q_len,
                        max_seqlen_k=k_attn_len,
                        causal=causal,
                        causal_stride=target_stride,
                        query_offset=0,
                    )
                    probs_target = probs_target[:, :q_len, :k_attn_len].contiguous()
                    probs_target = _align_probabilities_to_target_grid_1d(
                        probs_target.reshape(-1, 1, k_attn_len),
                        source_stride=self.training_kernel_stride,
                        target_stride=self.kernel_stride,
                        target_length=k_default_len,
                        mode=self.kl_target_align_pooling,
                    ).reshape(num_kv_heads, q_len, k_default_len)

                scores_pred = torch.matmul(q_pred_t, k_index_t) * scale_index
                if causal_mask_index is not None:
                    scores_pred = scores_pred.masked_fill(causal_mask_index.unsqueeze(0), float("-inf"))
                probs_pred = torch.softmax(scores_pred.float(), dim=-1).to(q_pred.dtype)
                probs_pred = _align_probabilities_to_target_grid_1d(
                    probs_pred.reshape(-1, 1, k_index_len),
                    source_stride=pred_stride,
                    target_stride=self.kernel_stride,
                    target_length=k_default_len,
                    mode=self.kl_target_align_pooling,
                ).reshape(num_kv_heads, q_len, k_default_len)
                probs_target_norm = probs_target / (probs_target.sum(dim=-1, keepdim=True) + 1e-8)
                probs_pred_norm = probs_pred / (probs_pred.sum(dim=-1, keepdim=True) + 1e-8)

                if not use_topk:
                    with torch.no_grad():
                        target_c = probs_target_norm.clamp_min(1e-8)
                    batch_kl = F.kl_div(
                        probs_pred_norm.clamp_min(1e-8).log(),
                        target_c,
                        reduction='sum',
                        log_target=False,
                    )
                else:
                    target_topk_batch = topk_idx[:, q_start:q_end, :]
                    batch_kl = _compute_kl_with_topk_chunk(
                        target_topk_batch,
                        probs_target_norm,
                        probs_pred_norm,
                        q_offset=0,
                        k_len_t=k_default_len,
                        k_len_p=k_default_len,
                    )

                all_kl_sum = all_kl_sum + batch_kl
                all_num_distributions += num_kv_heads * q_len
                continue
            
            # TARGET teacher on the native sparse grid: keep the stage1 fast path.
            with torch.no_grad():
                q_batch = query_states_unpad[q_start:q_end]
                k_batch = compressed_k_attn[k_attn_start:k_attn_end]
                k2_batch = compressed_k2[k2_start:k2_end]

                # Create per-batch cu_seqlens
                cu_q_buf[0] = 0
                cu_q_buf[1] = q_len
                cu_k_buf[0] = 0
                cu_k_buf[1] = k_attn_len
                cu_k2_buf[0] = 0
                cu_k2_buf[1] = (k2_end - k2_start)

                probs_target = _call_infllmv2_stage1_compat(
                    q_batch.contiguous(),
                    k_batch.contiguous(),
                    k2_batch.contiguous(),
                    cu_seqlens_q=cu_q_buf,
                    cu_seqlens_k=cu_k_buf,
                    cu_seqlens_v=cu_k2_buf,
                    max_seqlen_q=q_len,
                    max_seqlen_k=k_attn_len,
                    causal=causal,
                )
                # Trim to actual k_attn_len (kernel may pad to max)
                probs_target = probs_target[:, :q_len, :k_attn_len].contiguous()
            
            # PRED: softmax(q_future @ k_index)
            q_pred_t = q_pred.permute(1, 0, 2).contiguous()
            scores_pred = torch.matmul(q_pred_t, k_index_t) * scale_index
            if causal_mask_index is not None:
                scores_pred = scores_pred.masked_fill(causal_mask_index.unsqueeze(0), float("-inf"))
            probs_pred = torch.softmax(scores_pred.float(), dim=-1).to(q_pred.dtype)

            # Renormalize
            with torch.no_grad():
                probs_target_norm = probs_target / (probs_target.sum(dim=-1, keepdim=True) + 1e-8)
            probs_pred_norm = probs_pred / (probs_pred.sum(dim=-1, keepdim=True) + 1e-8)
            
            if not use_topk:
                with torch.no_grad():
                    target_c = probs_target_norm.clamp_min(1e-8)
                pred_c = probs_pred_norm.clamp_min(1e-8)
                batch_kl = F.kl_div(pred_c.log(), target_c, reduction='sum', log_target=False)
                all_kl_sum = all_kl_sum + batch_kl
                all_num_distributions += num_kv_heads * q_len
            else:
                target_topk_batch = topk_idx[:, q_start:q_end, :]
                batch_kl = _compute_kl_with_topk_chunk(
                    target_topk_batch,
                    probs_target_norm,
                    probs_pred_norm,
                    q_offset=0,
                    k_len_t=k_attn_len,
                    k_len_p=k_index_len,
                )
                
                all_kl_sum = all_kl_sum + batch_kl
                all_num_distributions += num_kv_heads * q_len
        
        if all_num_distributions == 0:
            return None, None
        
        kl_loss = all_kl_sum / all_num_distributions
        return kl_loss, "precomputed"


# =============================================================================
# ATTENTION CLASS REGISTRY
# =============================================================================

MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMFlashAttention2,
    "sdpa": MiniCPMSdpaAttention,
}


def _get_attention_class(config, layer_idx: int):
    """Get appropriate attention class based on config."""
    # Use InfLLMv2 attention if sparse_config is set and CUDA is available
    if getattr(config, 'sparse_config', None) is not None and torch.cuda.is_available():
        return MiniCPMInfLLMv2Attention
    return MINICPM_ATTENTION_CLASSES[config._attn_implementation]


# =============================================================================
# DECODER LAYER WITH Q_FUTURE SUPPORT
# =============================================================================

class MiniCPMDecoderLayer(nn.Module):
    """Decoder layer with InfLLM V2 indexer for sparse attention."""
    
    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Use sparse attention if sparse_config is set
        attention_class = _get_attention_class(config, layer_idx)
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)
        self.use_sparse_attn = (attention_class == MiniCPMInfLLMv2Attention)

        # Cache indexer parameter references for fast access in the model loop.
        # This avoids scanning named_parameters() every layer/step in immediate-grad mode.
        self.indexer_future_params = []
        self.indexer_curr_params = []
        if self.use_sparse_attn:
            attn = self.self_attn
            q_future_proj = getattr(attn, "q_future_proj", None)
            q_curr_proj = getattr(attn, "q_curr_proj", None)
            if q_future_proj is not None:
                self.indexer_future_params = list(q_future_proj.parameters())
            if q_curr_proj is not None:
                self.indexer_curr_params = list(q_curr_proj.parameters())

        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        
        # Note: Indexer (q_future_proj, q_curr_proj, compress_k_index) is now integrated
        # into MiniCPMInfLLMv2Attention. No separate indexer module needed.

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        q_future_from_prev: Optional[torch.Tensor] = None,
        training_stage: int = 0,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass for decoder layer with indexer.
        
        Args:
            hidden_states: Input tensor [batch, seq, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: KV cache
            output_attentions: Whether to return attention weights
            use_cache: Whether to use KV cache
            q_future_from_prev: Predicted query from previous layer's indexer
            training_stage: 0=Inference, 1=Training
            
        Returns:
            Tuple containing:
            - hidden_states: Output tensor
            - (optional) self_attn_weights: Attention weights if output_attentions
            - (optional) present_key_value: KV cache if use_cache
            - (if sparse) q_future_curr: Predicted query for next layer
            - (if sparse) aux_scores: Auxiliary scores for training loss
        """
        
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated.")

        # Apply input layernorm FIRST (matching reference design)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Detect prefill vs decode
        seq_len = hidden_states.shape[1]
        is_prefill = past_key_value is None or (
            isinstance(past_key_value, InfLLMv2Cache) and 
            len(past_key_value.layers[self.layer_idx].compress_k_cache) == 0
        )
        is_decode = seq_len == 1 and not is_prefill
        
        is_first_layer = (self.layer_idx == 0)
        q_future, q_curr = None, None
        aux_scores = None

        # Self Attention with integrated indexer
        # Attention computes q_future, q_curr, averaged K, and compressed K internally
        if self.use_sparse_attn:
            # Layer 0: use q_curr from this layer; Layer 1+: use q_future from previous layer
            q_for_scoring = q_future_from_prev if not is_first_layer else None
            
            attn_result = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                q_future_from_prev=q_for_scoring,
                training_stage=training_stage,
                **kwargs,
            )
            # Returns: (output, weights, past_kv, aux_scores, q_future, q_curr, key_avg)
            hidden_states, self_attn_weights, present_key_value, aux_scores, \
                q_future, q_curr, key_states_for_indexer = attn_result
            
            # Note: No separate K caching needed for indexer
            # Indexer scoring uses avg(compressed_k) computed at query time from attention's cached compressed_k
        else:
            attn_result = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            # Standard returns: (output, weights, past_kv)
            hidden_states, self_attn_weights, present_key_value = attn_result
            aux_scores = None
        
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        # MLP
        torch.cuda.nvtx.range_push("ffn")
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        torch.cuda.nvtx.range_pop()  # ffn

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        # Pass q_future to next layer (None for last layer)
        if self.use_sparse_attn:
            outputs += (q_future, aux_scores)

        return outputs


# =============================================================================
# MODEL CLASSES
# =============================================================================

MINICPM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MiniCPMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMPreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = MiniCPMConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable or disable gradient checkpointing for decoder layers."""
        if isinstance(module, MiniCPMModel):
            module.gradient_checkpointing = value

    def _init_weights(self, module):
        std = self.config.initializer_range
        
        # Check if this is an indexer module and if we should use Kaiming init
        sparse_config = getattr(self.config, 'sparse_config', None) or {}
        use_kaiming_for_indexer = sparse_config.get('indexer_kaiming_init', True)  # Default True for better training
        
        is_indexer = getattr(module, '_is_indexer', False)
        
        if isinstance(module, nn.Linear):
            # Use Kaiming uniform for indexer projections (better for softmax-based training)
            if use_kaiming_for_indexer and is_indexer:
                fan_in = module.weight.shape[1]
                bound = 1.0 / math.sqrt(fan_in)
                module.weight.data.uniform_(-bound, bound)
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MINICPM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states for faster decoding.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can directly pass an embedded representation.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        training_stage (`int`, *optional*, defaults to 0):
            InfLLM stage selector: `0` for inference, `1` for training.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMModel(MiniCPMPreTrainedModel):
    """MiniCPM decoder model with InfLLM V2 support."""

    def __init__(self, config: MiniCPMConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sparse_attn = getattr(config, 'sparse_config', None) is not None and torch.cuda.is_available()

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training_stage: int = 0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        use_legacy_cache = False
        if use_cache:
            if past_key_values is not None:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            
            # Get past key values length
            if isinstance(past_key_values, InfLLMv2Cache):
                past_key_values_length = past_key_values.get_seq_length()
            elif hasattr(past_key_values, 'get_usable_length'):
                past_key_values_length = past_key_values.get_usable_length(seq_length)
            elif hasattr(past_key_values, 'get_seq_length'):
                past_key_values_length = past_key_values.get_seq_length()
            else:
                past_key_values_length = 0
            
            # Initialize InfLLMv2Cache for sparse attention when cache is empty (matches original MiniCPM)
            if self._use_sparse_attn and past_key_values_length == 0:
                past_key_values = InfLLMv2Cache(
                    config=self.config, 
                    num_hidden_layers=self.config.num_hidden_layers
                )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        # Prepare attention mask based on attention implementation
        if self._use_flash_attention_2 or self._use_sparse_attn:
            # Flash attention / sparse attention need 2D mask
            if attention_mask is None:
                raise ValueError("attention_mask is required for flash attention or sparse attention")
            # Keep 2D attention mask for variable length handling
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # Indexer state management (only for sparse attention)
        # q_future is passed from layer N to layer N+1
        q_future_prev = None
        self._prev_layer_indexer_params = None  # For immediate gradient mode
        
        # Accumulated KL loss across layers (for training stages 1, 2)
        accumulated_kl_loss = None
        kl_loss_count = 0
        
        # Disable cache when using gradient checkpointing during training
        use_cache_effective = use_cache and not (self.gradient_checkpointing and self.training)
        _infinigen_active = any(
            getattr(decoder_layer.self_attn, 'infinigen_enabled', False)
            for decoder_layer in self.layers
        )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing: recompute activations during backward to save memory
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,  # past_key_values must be None for checkpointing
                    output_attentions,
                    False,  # use_cache must be False for checkpointing
                    q_future_prev,
                    training_stage,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache_effective,
                    q_future_from_prev=q_future_prev,
                    training_stage=training_stage,
                )

            hidden_states = layer_outputs[0]

            if _infinigen_active and idx + 1 < len(self.layers):
                next_attn = self.layers[idx + 1].self_attn
                attn = decoder_layer.self_attn
                if (getattr(next_attn, 'infinigen_enabled', False)
                        and hasattr(attn, '_ig_attn_input') and attn._ig_attn_input is not None):
                    next_attn._ig_prev_hidden_states = attn._ig_attn_input

            # Update q_future for next layer (only for sparse attention)
            # Layer outputs for sparse: (hidden, [attn_weights], [cache], q_future, aux_scores)
            if decoder_layer.use_sparse_attn:
                q_future_prev = layer_outputs[-2]  # Second from last is q_future
                # Handle KL loss for training
                aux_scores = layer_outputs[-1]
                if aux_scores is not None and training_stage == 1:
                    scores_pred, scores_target = aux_scores
                    if scores_target == "precomputed" and scores_pred is not None:
                        # Immediate gradient mode: compute indexer gradients NOW and free KL graph
                        # This is compatible with gradient checkpointing because:
                        # 1. With use_reentrant=False, forward runs WITH gradient tracking
                        # 2. scores_pred has a gradient graph
                        # 3. We compute indexer gradients immediately and free the KL graph
                        # 4. Checkpointing only recomputes main forward during LM backward
                        # Immediate gradient mode: autograd.grad() per layer with CORRECT params
                        # Layer 0: KL depends on this layer's q_curr_proj
                        # Layer N>0: KL depends on PREVIOUS layer's q_future_proj
                        if self.training and hasattr(self, '_accumulate_kl_grads') and self._accumulate_kl_grads:
                            # Get the CORRECT indexer params for this layer's KL loss
                            if hasattr(self, '_prev_layer_indexer_params') and self._prev_layer_indexer_params is not None:
                                # Layer N>0: use previous layer's q_future_proj
                                target_params = self._prev_layer_indexer_params
                            else:
                                # Layer 0: use this layer's q_curr_proj
                                target_params = [p for p in decoder_layer.indexer_curr_params if p.requires_grad]
                            
                            if target_params:
                                scaled_loss = scores_pred / self._kl_grad_scale
                                grads = torch.autograd.grad(
                                    scaled_loss,
                                    target_params,
                                    retain_graph=False,
                                    allow_unused=True,
                                )
                                for param, grad in zip(target_params, grads):
                                    if grad is None:
                                        continue
                                    if param.grad is None:
                                        param.grad = grad
                                    else:
                                        param.grad.add_(grad)
                            
                            # Record detached loss for logging
                            if accumulated_kl_loss is None:
                                accumulated_kl_loss = scores_pred.detach()
                                kl_loss_count = 1
                            else:
                                accumulated_kl_loss = accumulated_kl_loss + scores_pred.detach()
                                kl_loss_count += 1
                            self._kl_grads_need_allreduce = True
                        else:
                            # Standard mode: accumulate loss for single backward
                            if accumulated_kl_loss is None:
                                accumulated_kl_loss = scores_pred
                                kl_loss_count = 1
                            else:
                                accumulated_kl_loss = accumulated_kl_loss + scores_pred
                                kl_loss_count += 1
                
                # ALWAYS store this layer's q_future_proj for NEXT layer's KL
                # (even if this layer had no KL loss, e.g., layer 0 with no q_future_from_prev)
                if self.training and hasattr(self, '_accumulate_kl_grads') and self._accumulate_kl_grads:
                    self._prev_layer_indexer_params = [p for p in decoder_layer.indexer_future_params if p.requires_grad]
                
                # Explicit cleanup to help garbage collection.
                # IMPORTANT: scores_pred/scores_target are only defined in training_stage == 1.
                del aux_scores
                if training_stage == 1:
                    try:
                        del scores_pred, scores_target
                    except NameError:
                        pass

            if use_cache_effective:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache_effective and next_decoder_cache is not None:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        
        # Compute average KL loss if accumulated
        avg_kl_loss = None
        if accumulated_kl_loss is not None and kl_loss_count > 0:
            avg_kl_loss = accumulated_kl_loss / kl_loss_count
        
        if not return_dict:
            result = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            if avg_kl_loss is not None:
                result = result + (avg_kl_loss,)
            return result
        
        # Store KL loss in attentions field if training (hacky but avoids new output class)
        # CausalLMOutputWithPast will extract it
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        # Attach KL loss as attribute for single-pass training
        output.kl_loss = avg_kl_loss
        return output


class MiniCPMForCausalLM(MiniCPMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training_stage: int = 0,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            training_stage (`int`, *optional*, defaults to 0):
                InfLLM stage selector: `0` for inference, `1` for training.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                If an int, compute logits for the last `logits_to_keep` tokens. If 0, compute all.
                If a `torch.Tensor`, compute logits for the specified positions.

        Returns:
            CausalLMOutputWithPast or tuple
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Backward-compat: some call sites pass `num_logits_to_keep` (older naming).
        if "num_logits_to_keep" in kwargs:
            if isinstance(logits_to_keep, int) and logits_to_keep == 0:
                logits_to_keep = kwargs.pop("num_logits_to_keep")
            else:
                kwargs.pop("num_logits_to_keep")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training_stage=training_stage,
        )

        hidden_states = outputs[0]

        # Keep a copy for loss computation (we may compute logits only for a slice to save memory)
        hidden_states_full = hidden_states
        chunk_size_cfg = int(getattr(self.config, "chunk_size", 0) or 0)

        # Only compute necessary logits for efficiency.
        # When chunked loss is enabled (chunk_size_cfg > 0) and labels are provided, we typically do NOT
        # need full logits and materializing [B, 64K, vocab] can OOM. In that case, default to last-token logits.
        if (
            self.training
            and labels is not None
            and chunk_size_cfg > 0
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 0
        ):
            slice_indices = slice(-1, None)
        else:
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int) and logits_to_keep > 0
                else logits_to_keep
            )

        hidden_states_for_logits = hidden_states_full
        if isinstance(slice_indices, slice) or isinstance(slice_indices, torch.Tensor):
            if isinstance(slice_indices, slice) and slice_indices != slice(None):
                hidden_states_for_logits = hidden_states_for_logits[:, slice_indices, :].contiguous()
            elif isinstance(slice_indices, torch.Tensor):
                hidden_states_for_logits = hidden_states_for_logits[:, slice_indices, :].contiguous()

        def _lm_head(h: torch.Tensor) -> torch.Tensor:
            """Project hidden states to vocab logits (supports pretraining_tp)."""
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits_parts = [F.linear(h, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                return torch.cat(logits_parts, dim=-1)
            scale = (self.config.hidden_size / self.config.dim_model_base)
            return self.lm_head(h / scale)

        # Compute loss without materializing full [seq_len, vocab] logits for long sequences.
        loss = None
        if labels is not None:
            bsz, seq_len, _ = hidden_states_full.shape
            if seq_len <= 1:
                loss = None
            else:
                if self.training and chunk_size_cfg > 0:
                    # Chunked CE (enabled): avoids allocating [seq_len, vocab] logits.
                    hs = hidden_states_full[:, :-1, :]
                    tgt = labels[:, 1:].to(hs.device)
                    chunk_size = chunk_size_cfg
                    total_loss = hs.new_zeros((), dtype=torch.float32)
                    total_tokens = 0
                    loss_fct = CrossEntropyLoss(reduction="sum")

                    for start in range(0, hs.shape[1], chunk_size):
                        end = min(start + chunk_size, hs.shape[1])
                        hs_chunk = hs[:, start:end, :]
                        tgt_chunk = tgt[:, start:end]

                        logits_chunk = _lm_head(hs_chunk)
                        loss_chunk = loss_fct(
                            logits_chunk.reshape(-1, self.config.vocab_size),
                            tgt_chunk.reshape(-1),
                        )
                        total_loss = total_loss + loss_chunk
                        total_tokens += tgt_chunk.numel()

                    loss = total_loss / max(total_tokens, 1)
                else:
                    # Non-chunked CE (disabled): compute full logits for loss (may OOM at long seq).
                    logits_full = _lm_head(hidden_states_full)
                    if not self.training:
                        logits_full = logits_full.float()
                    shift_logits = logits_full[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # Compute logits for output (can still be large; by default training callers shouldn't rely on it)
        logits = _lm_head(hidden_states_for_logits)
        if not self.training:
            logits = logits.float()

        # Extract KL loss from model outputs (set during training stages 1, 2)
        kl_loss = getattr(outputs, 'kl_loss', None)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            result = (loss,) + output if loss is not None else output
            if kl_loss is not None:
                result = result + (kl_loss,)
            return result

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # Attach KL loss for single-pass training
        result.kl_loss = kl_loss
        return result

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Check if using sparse attention
        use_sparse_attn = getattr(self.config, 'sparse_config', None) is not None and torch.cuda.is_available()
        
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                
                # Initialize InfLLMv2Cache for sparse attention when cache is empty (matches original MiniCPM)
                if use_sparse_attn and cache_length == 0:
                    past_key_values = InfLLMv2Cache(
                        config=self.config,
                        num_hidden_layers=self.config.num_hidden_layers
                    )
                
                # Handle InfLLMv2Cache which may not have seen_tokens
                if hasattr(past_key_values, 'seen_tokens'):
                    past_length = past_key_values.seen_tokens
                else:
                    past_length = cache_length
                max_cache_length = past_key_values.get_max_length() if hasattr(past_key_values, 'get_max_length') else None
            else:
                raise ValueError(
                    'You must use the new past_key_values format, such as the Cache class, instead of the old tuple format.'
                )

            # Keep only unprocessed tokens
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        else:
            # No past cache - will be created in model.forward if needed
            pass

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        
        # Forward additional kwargs
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
    @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 4096, num_beams=1, do_sample=True, top_p=0.8, temperature=0.3, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs
        }
        
        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(history_str, return_tensors='pt').to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(r".*?(?=<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) > 0:
            response = matches[0]
        history.append({"role": "assistant", "content": response})
        return response, history
