# This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
# Copyright (c) 2026 THUNLP.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Fused InfiniGen partial-dimension scoring with GQA.

Two-phase design:
  Phase 1 (PyTorch): q_combined = sum_g(Q_g @ A_partial[h])
    Small matmul handled by cuBLAS — fast, no register pressure.
  Phase 2 (Triton):  scores = PK_block @ q_combined
    Memory-bandwidth-bound dot product.  Lean kernel with minimal
    registers → high occupancy → good latency hiding on HBM reads.

Requires precomputed A_partial = gather(A, partial_indices) at warmup time.
"""

import math
import torch
import triton
import triton.language as tl


def precompute_a_partial(skewing_matrix, partial_indices):
    """Precompute A_partial by gathering columns from skewing matrix.

    Called once at warmup. Stores the result on the layer for reuse.

    Args:
        skewing_matrix: (H_kv, D, D)
        partial_indices: (H_kv, NC) int64

    Returns:
        A_partial: (H_kv, D, NC) -- columns of A selected by partial_indices
    """
    H_kv, D, _ = skewing_matrix.shape
    NC = partial_indices.shape[1]
    idx = partial_indices.unsqueeze(1).expand(-1, D, -1)  # (H_kv, D, NC)
    A_partial = torch.gather(skewing_matrix, 2, idx)  # (H_kv, D, NC)
    return A_partial.contiguous()


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2 Triton kernel: PK @ q_combined
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _pk_dot_kernel(
    # q_combined: (B, H_KV, NC) contiguous
    QC_ptr,
    # Partial key cache: (B, S, H_KV, NC) contiguous
    PK_ptr,
    # Output: (H_KV, B, S)
    OUT_ptr,
    # PK strides (elements)
    pk_stride_b,
    pk_stride_s,
    pk_stride_h,
    # Dynamic dims
    B, S,
    # Compile-time dims
    H_KV: tl.constexpr,
    NC: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Grid: (B, H_KV, cdiv(S, BLOCK_S))

    Each program computes scores for one (batch, kv_head, seq_block):
        scores[s] = PK[b, s, h, :] . q_combined[b, h, :] * SCALE
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    # Load q_combined vector: (NC,)  — tiny, stays in registers / L1
    qc_base = (pid_b * H_KV + pid_h) * NC
    nc_range = tl.arange(0, NC)
    q_vec = tl.load(QC_ptr + qc_base + nc_range).to(tl.float32)  # (NC,)

    # Sequence offsets for this block
    s_start = pid_s * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)  # (BLOCK_S,)
    s_mask = s_offsets < S

    # Load PK block: (BLOCK_S, NC) and compute dot product
    pk_base = pid_b * pk_stride_b + pid_h * pk_stride_h
    pk_ptrs = (PK_ptr + pk_base
               + s_offsets[:, None] * pk_stride_s
               + nc_range[None, :])                          # (BLOCK_S, NC)
    pk_block = tl.load(pk_ptrs, mask=s_mask[:, None],
                       other=0.0).to(tl.float32)             # (BLOCK_S, NC)

    # (BLOCK_S, NC) * (1, NC) -> sum axis=1 -> (BLOCK_S,)
    acc = tl.sum(pk_block * q_vec[None, :], axis=1) * SCALE

    # Store: output layout (H_KV, B, S)
    out_base = (pid_h * B + pid_b) * S
    tl.store(OUT_ptr + out_base + s_offsets,
             acc.to(OUT_ptr.dtype.element_ty), mask=s_mask)


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def infinigen_score_triton(
    q_after_rope,       # (B, H_Q, D)
    a_partial,          # (H_KV, D, NC) -- precomputed at warmup
    partial_key_cache,  # (B, S, H_KV, NC) or (S, H_KV, NC)
    num_channels,       # NC
):
    """Fused InfiniGen scoring: q @ A_partial -> dot(PK) -> GQA sum -> scale.

    Returns:
        scores: (num_kv_heads, B, seq_len)
    """
    if partial_key_cache.dim() == 3:
        partial_key_cache = partial_key_cache.unsqueeze(0).expand(
            q_after_rope.shape[0], -1, -1, -1).contiguous()

    B, S, H_KV, NC = partial_key_cache.shape
    H_Q = q_after_rope.shape[1]
    D = q_after_rope.shape[2]
    GQA = H_Q // H_KV

    ap = a_partial.to(q_after_rope.dtype).contiguous()
    pk = partial_key_cache.contiguous()

    # ── Phase 1 (PyTorch): q_combined = Σ_g Q_g @ A_partial ──────────
    # Sum Q-heads within each GQA group first (reduces D matmul to 1 per KV-head)
    q_per_kv = q_after_rope.reshape(B, H_KV, GQA, D).sum(2)  # (B, H_KV, D)
    # Batched matmul: (B, H_KV, D) × (H_KV, D, NC) → (B, H_KV, NC)
    q_combined = torch.einsum('bhd,hdc->bhc', q_per_kv, ap).contiguous()

    # ── Phase 2 (Triton): scores = PK @ q_combined ───────────────────
    out = torch.empty(H_KV, B, S, dtype=q_after_rope.dtype, device=q_after_rope.device)
    scale = 1.0 / math.sqrt(num_channels)

    BLOCK_S = 128
    grid = (B, H_KV, triton.cdiv(S, BLOCK_S))

    _pk_dot_kernel[grid](
        q_combined, pk, out,
        pk_stride_b=pk.stride(0),
        pk_stride_s=pk.stride(1),
        pk_stride_h=pk.stride(2),
        B=B, S=S,
        H_KV=H_KV, NC=NC,
        SCALE=scale,
        BLOCK_S=BLOCK_S,
    )

    return out
