# This file is copied/adapted from infllmv2_cuda_impl (https://github.com/OpenBMB/infllmv2_cuda_impl).
# Copyright (c) OpenBMB.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Tuple, Optional
from . import C

# torch.compile() support
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


@_torch_custom_op_wrapper("infllmv2_attn::_topk_to_uint64", mutates_args=(), device_types="cuda")
def _topk_to_uint64_inner(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
) -> torch.Tensor:
    """Inner custom op for topk_to_uint64 - torch.compile compatible."""
    assert topk_idx.dtype == torch.int32
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    
    original_shape = topk_idx.shape
    has_batch = len(original_shape) == 4
    
    if has_batch:
        batch_size, num_heads, total_seqlen, k = original_shape
        flat_dims = batch_size * num_heads * total_seqlen
        output_shape = (batch_size, num_heads, total_seqlen, (k_blocks + 63) // 64)
    else:
        num_heads, total_seqlen, k = original_shape
        flat_dims = num_heads * total_seqlen
        output_shape = (num_heads, total_seqlen, (k_blocks + 63) // 64)
    
    n_uint64_per_row = (k_blocks + 63) // 64
    
    with torch.cuda.device(topk_idx.device):
        stream = torch.cuda.current_stream().cuda_stream
        result = torch.zeros(output_shape, dtype=torch.int64, device=topk_idx.device)
        
        C.topk_to_uint64(
            stream,
            topk_idx.data_ptr(),
            result.data_ptr(),
            flat_dims,
            original_shape[-1],  # k
            k_blocks,
            n_uint64_per_row
        )
    
    return result


@_torch_register_fake_wrapper("infllmv2_attn::_topk_to_uint64")
def _topk_to_uint64_fake(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    n_uint64_per_row = (k_blocks + 63) // 64
    
    original_shape = topk_idx.shape
    has_batch = len(original_shape) == 4
    
    if has_batch:
        output_shape = (original_shape[0], original_shape[1], original_shape[2], n_uint64_per_row)
    else:
        output_shape = (original_shape[0], original_shape[1], n_uint64_per_row)
    
    return torch.empty(output_shape, dtype=torch.int64, device=topk_idx.device)


# Wrap for torch.compile compatibility
if torch.__version__ >= "2.4.0":
    _wrapped_topk_to_uint64 = torch.ops.infllmv2_attn._topk_to_uint64
else:
    _wrapped_topk_to_uint64 = _topk_to_uint64_inner


def topk_to_uint64(topk_idx: torch.Tensor, max_seqlen_k: int, block_size: int, 
                   memory_buffer: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
    """
    Convert topk indices directly to uint64 representation without intermediate bool mask
    
    Args:
        topk_idx: Tensor of shape [batch, num_heads, total_seqlen, k] or [num_heads, total_seqlen, k]
                 containing block indices
        max_seqlen_k: Maximum sequence length for keys
        block_size: Size of each block
        memory_buffer: Optional pre-allocated buffer to reuse (ignored, kept for API compatibility)
        
    Returns:
        Tuple of:
            uint64_arrays: Tensor with the same batch dimensions but last dim replaced with uint64 values
            k_blocks: Number of key blocks
    """
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    result = _wrapped_topk_to_uint64(topk_idx, max_seqlen_k, block_size)
    return result, k_blocks


    """
    A class that manages memory buffer for topk_to_uint64 conversions.
    This can improve performance by reusing memory across multiple calls.
    """
    
    def __init__(self):
        self.memory_buffer = None
    
    def convert(self, topk_idx: torch.Tensor, max_seqlen_k: int, block_size: int) -> Tuple[torch.Tensor, int]:
        """
        Convert topk indices to uint64 representation, reusing memory buffer when possible.
        
        Args:
            topk_idx: Tensor of shape [batch, num_heads, total_seqlen, k] or [num_heads, total_seqlen, k]
                     containing block indices
            max_seqlen_k: Maximum sequence length for keys
            block_size: Size of each block
            
        Returns:
            Tuple of:
                uint64_arrays: Tensor with the same batch dimensions but last dim replaced with uint64 values
                k_blocks: Number of key blocks
        """
        result, k_blocks = topk_to_uint64(topk_idx, max_seqlen_k, block_size, self.memory_buffer)
        # Update our memory buffer reference for next time
        self.memory_buffer = result
        return result, k_blocks
    
    def clear_memory(self):
        """Clear the internal memory buffer"""
        self.memory_buffer = None 