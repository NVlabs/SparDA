# This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
# Copyright (c) 2026 THUNLP.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""InfiniGen controllers adapted for GQA models (MiniCPM4.1-8B).

Ported from:
  - accuracy/setup/gen_llama_skewing_matrix.py (SVD skewing)
  - accuracy/src/modeling_llama_ours_setup.py (partial weight extraction)
  - accuracy/src/modeling_llama_ours.py (kv_cache_mask)
  - speedup/infinigen/infinigen/kv_selection_controller.py (speculate_attention)
  - speedup/infinigen/infinigen/partial_weight_generation_controller.py
  - speedup/infinigen/infinigen/skewing_controller.py

Adapted for GQA: MiniCPM has 32 Q-heads and 2 KV-heads (group_size=16).
Skewing is per-Q-head. Partial weight masks are per-Q-head.
Speculation aggregates pre-softmax scores across Q-heads within a KV group.
"""

import math
import queue
import time
import threading
import torch
import torch.nn.functional as F

# Triton fused scoring kernel (lazy import — only on GPU with triton available)
_triton_score_fn = None
_triton_precompute_fn = None
_triton_checked = False


def _ensure_triton_score():
    global _triton_score_fn, _triton_precompute_fn, _triton_checked
    if _triton_checked:
        return
    _triton_checked = True
    try:
        from .triton_infinigen_score import infinigen_score_triton, precompute_a_partial
        _triton_score_fn = infinigen_score_triton
        _triton_precompute_fn = precompute_a_partial
    except ImportError:
        pass


def get_precompute_a_partial():
    """Return the precompute_a_partial function for warmup-time precomputation."""
    _ensure_triton_score()
    if _triton_precompute_fn is None:
        # Fallback: pure PyTorch implementation
        def _precompute(skewing_matrix, partial_indices):
            H_kv, D, _ = skewing_matrix.shape
            NC = partial_indices.shape[1]
            idx = partial_indices.unsqueeze(1).expand(-1, D, -1)
            return torch.gather(skewing_matrix, 2, idx).contiguous()
        return _precompute
    return _triton_precompute_fn

# Lazy-loaded C++ multithreaded gather (falls back to F.embedding if unavailable)
_cpu_gather_mod = None
_cpu_gather_checked = False


def _ensure_cpu_gather():
    global _cpu_gather_mod, _cpu_gather_checked
    if not _cpu_gather_checked:
        _cpu_gather_checked = True
        from .flash_cache_engine.cpu_gather_loader import get_cpu_gather
        _cpu_gather_mod = get_cpu_gather()
        if _cpu_gather_mod is None:
            raise RuntimeError(
                "Failed to build C++ cpu_gather extension. "
                "Ensure a C++ compiler with OpenMP support is available."
            )


def fast_cpu_gather(src_flat, indices):
    """Gather rows from a flat CPU tensor using C++ multithreaded gather with
    software prefetching.

    Args:
        src_flat: (N, D) contiguous CPU tensor
        indices:  (M,)   int64 CPU tensor of row indices

    Returns:
        (M, D) CPU tensor with gathered rows
    """
    _ensure_cpu_gather()
    return _cpu_gather_mod.cpu_gather(src_flat.contiguous(), indices.contiguous())


# =============================================================================
# SVD Skewing Matrix
# =============================================================================

def compute_skewing_matrix(q_act, k_act, num_heads, num_kv_heads, head_dim):
    """Compute per-KV-head skewing matrix A from post-RoPE Q/K activations.

    For GQA: Q-heads in each group are averaged before SVD (matching CLO impl).
    Returns one skewing matrix per KV-head.

    Args:
        q_act: (bsz, num_heads, seq_len, head_dim) — post-RoPE Q activations
        k_act: (bsz, num_kv_heads, seq_len, head_dim) — post-RoPE K activations
        num_heads: total number of Q heads (32 for MiniCPM)
        num_kv_heads: number of KV heads (2 for MiniCPM)
        head_dim: dimension per head (128 for MiniCPM)

    Returns:
        A: (num_kv_heads, head_dim, head_dim) — skewing matrix per KV-head
    """
    group_size = num_heads // num_kv_heads
    device = q_act.device
    dtype = q_act.dtype
    A = torch.zeros(num_kv_heads, head_dim, head_dim, device=device, dtype=dtype)

    for kv_h in range(num_kv_heads):
        # Average Q-heads in this KV group (matching CLO's gen_skewing.py)
        q_group = q_act[0, kv_h * group_size:(kv_h + 1) * group_size]  # (group_size, seq, head_dim)
        q_avg = q_group.mean(dim=0)  # (seq, head_dim)

        _, sq, vq = torch.svd(q_avg.float())
        _, sk, _ = torch.svd(k_act[0, kv_h].float())

        s = sq * sk
        _, ind = s.sort()  # ascending order

        vq = vq.to(dtype)
        a = torch.zeros(head_dim, head_dim, device=device, dtype=dtype)
        A[kv_h] = a.scatter(-1, ind.unsqueeze(0).expand(head_dim, -1), vq)

    return A


# =============================================================================
# Partial Weight Index Extraction
# =============================================================================

def compute_partial_weight_indices(q_act, k_act, A, num_heads, num_kv_heads, head_dim,
                                   num_channels=32):
    """Compute per-KV-head partial dimension indices (matching CLO implementation).

    For each KV-head: skew Q (averaged across GQA group) and K, take abs-sum
    over sequence, sum across Q-heads in group, select top num_channels dims.

    Args:
        q_act: (bsz, num_heads, seq_len, head_dim) — post-RoPE Q
        k_act: (bsz, num_kv_heads, seq_len, head_dim) — post-RoPE K
        A: (num_kv_heads, head_dim, head_dim) — skewing matrices (per KV-head)
        num_heads: total Q heads
        num_kv_heads: KV heads
        head_dim: head dimension
        num_channels: number of dimensions to select (default 32)

    Returns:
        indices: (num_kv_heads, num_channels) — selected dimension indices per KV-head
    """
    group_size = num_heads // num_kv_heads
    device = q_act.device
    A = A.to(device=device, dtype=q_act.dtype)

    indices = torch.zeros(num_kv_heads, num_channels, dtype=torch.long, device=device)

    for kv_h in range(num_kv_heads):
        # Skew all Q-heads in this group, take abs-sum over seq, then sum across group
        q_group = q_act[0, kv_h * group_size:(kv_h + 1) * group_size]  # (group_size, seq, head_dim)
        q_skewed = torch.matmul(q_group, A[kv_h])  # (group_size, seq, head_dim)
        q_importance = q_skewed.abs().sum(dim=1).sum(dim=0)  # (head_dim,) — sum over seq then group

        _, topk = torch.topk(q_importance, num_channels)
        indices[kv_h] = topk

    return indices

# =============================================================================
# Block-Level InfiniGen Token Scoring (HF Accuracy Path)
# =============================================================================

def infinigen_token_scores_hf(
    prev_hidden,
    q_proj_weight,
    skewing_matrix,
    partial_indices,
    partial_key_cache,
    cos,
    sin,
    position_ids,
    num_heads,
    num_kv_heads,
    head_dim,
    num_channels,
    apply_rope_fn,
    a_partial=None,
):
    """Compute InfiniGen token-level scores for HF accuracy paths.

    Runtime einsum skewing (CLO style). Speculates Q from previous layer's
    attention input (already post-layernorm), applies RoPE, skews, extracts
    partial dims, scores against partial key cache. Pre-softmax SUM across
    GQA group.

    Args:
        prev_hidden: (B, 1, D) — previous layer's attention input (post-layernorm)
        q_proj_weight: (num_heads * head_dim, D) — Q projection weight
        skewing_matrix: (num_kv_heads, head_dim, head_dim) — per-KV-head skewing
        partial_indices: (num_kv_heads, num_channels) — selected dim indices
        partial_key_cache: (B, seq_len, H_kv, nc) or (seq_len, H_kv, nc) — partial K on GPU
        cos: RoPE cosine tensor
        sin: RoPE sine tensor
        position_ids: (B, 1) — current position
        num_heads: total Q heads
        num_kv_heads: KV heads
        head_dim: head dimension
        num_channels: number of partial dimensions
        apply_rope_fn: function(q, k, cos, sin, pos) -> (q_rotated, _)

    Returns:
        token_scores: (num_kv_heads, B, seq_len) — pre-softmax aggregated scores
    """
    bsz = prev_hidden.shape[0]
    group_size = num_heads // num_kv_heads

    # 1. Q projection (no layernorm — prev_hidden is already post-layernorm)
    spec_q = F.linear(prev_hidden, q_proj_weight)  # (B, 1, num_heads * head_dim)
    spec_q = spec_q.view(bsz, 1, num_heads, head_dim).transpose(1, 2)  # (B, H, 1, D)

    # 3. Apply RoPE
    spec_q, _ = apply_rope_fn(spec_q, spec_q[:, :num_kv_heads], cos, sin, position_ids)

    # 4-9: Fused scoring — Triton kernel if a_partial is provided, else PyTorch fallback
    _ensure_triton_score()
    if a_partial is not None and _triton_score_fn is not None and spec_q.is_cuda:
        # Triton fast path: fused skew → extract → dot → GQA sum → scale
        q_for_triton = spec_q.squeeze(2)  # (B, H, 1, D) → (B, H, D)
        return _triton_score_fn(q_for_triton, a_partial, partial_key_cache, num_channels)

    # PyTorch fallback
    # 4. Reshape to (B, H_kv, group_size, 1, head_dim)
    spec_q_grouped = spec_q.view(bsz, num_kv_heads, group_size, 1, head_dim)

    # 5. Skew: einsum('bhgsd,hde->bhgse', q_grouped, A)
    skew = skewing_matrix.to(spec_q.dtype).to(spec_q.device)
    spec_q_skewed = torch.einsum('bhgsd,hde->bhgse', spec_q_grouped, skew)

    # 6. Extract partial dims via gather
    p_idx = partial_indices.to(spec_q.device)
    idx_q = p_idx.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(
        bsz, -1, group_size, 1, -1)
    spec_q_partial = torch.gather(spec_q_skewed, -1, idx_q)

    # 7. Score: partial_q @ partial_k^T
    if partial_key_cache.dim() == 3:
        p_k = partial_key_cache.unsqueeze(0).expand(bsz, -1, -1, -1).permute(0, 2, 1, 3)
    else:
        p_k = partial_key_cache.permute(0, 2, 1, 3)

    scores = torch.matmul(spec_q_partial, p_k.unsqueeze(2).transpose(-1, -2))
    scores = scores / math.sqrt(num_channels)

    # 8-9. GQA sum → permute
    score_agg = scores.sum(dim=2).squeeze(2)
    return score_agg.permute(1, 0, 2).contiguous()


def infinigen_token_scores_nosi(
    prev_hidden,
    q_proj_weight,
    skewing_matrix,
    partial_indices,
    partial_key_cache,
    cos_sin_cache,
    position_ids,
    num_heads,
    num_kv_heads,
    head_dim,
    num_channels,
    apply_rope_inplace_fn=None,
    a_partial=None,
):
    """Compute InfiniGen token-level scores for NOSI efficiency paths.

    Uses runtime skewing (same as HF) for correctness.

    Args:
        prev_hidden: (B, 1, D) — previous layer's attention input (post-layernorm)
        q_proj_weight: (num_heads * head_dim, D) — Q projection weight (NOT baked)
        skewing_matrix: (num_kv_heads, head_dim, head_dim) — per-KV-head skewing
        partial_indices: (num_kv_heads, num_channels) — selected dim indices
        partial_key_cache: (B, seq_len, H_kv, nc) or (seq_len, H_kv, nc) — partial K on GPU
        cos_sin_cache: (max_seq, head_dim) — combined cos/sin for flashinfer RoPE
        position_ids: (B, 1) — current position
        num_heads: total Q heads
        num_kv_heads: KV heads
        head_dim: head dimension
        num_channels: number of partial dimensions
        apply_rope_inplace_fn: flashinfer apply_rope_with_cos_sin_cache_inplace

    Returns:
        token_scores: (num_kv_heads, B, seq_len) — pre-softmax aggregated scores
    """
    bsz = prev_hidden.shape[0]
    group_size = num_heads // num_kv_heads

    # 1. Q projection (no layernorm — prev_hidden is already post-layernorm)
    spec_q = F.linear(prev_hidden, q_proj_weight)  # (B, 1, num_heads * head_dim)
    spec_q = spec_q.view(bsz, 1, num_heads, head_dim)  # (B, 1, H, D)

    # 3. Apply RoPE in-place
    if apply_rope_inplace_fn is not None:
        apply_rope_inplace_fn(
            spec_q.view(-1, num_heads, head_dim),
            cos_sin_cache,
            position_ids.view(-1).int(),
        )

    # 4-9: Fused scoring — Triton kernel if a_partial is provided
    _ensure_triton_score()
    if a_partial is not None and _triton_score_fn is not None and spec_q.is_cuda:
        q_for_triton = spec_q.squeeze(1)  # (B, 1, H, D) → (B, H, D)
        return _triton_score_fn(q_for_triton, a_partial, partial_key_cache, num_channels)

    # PyTorch fallback
    spec_q_grouped = spec_q.view(bsz, num_kv_heads, group_size, 1, head_dim)

    skew = skewing_matrix.to(spec_q.dtype).to(spec_q.device)
    spec_q_skewed = torch.einsum('bhgsd,hde->bhgse', spec_q_grouped, skew)

    p_idx = partial_indices.to(spec_q.device)
    idx_q = p_idx.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(
        bsz, -1, group_size, 1, -1)
    spec_q_partial = torch.gather(spec_q_skewed, -1, idx_q)

    if partial_key_cache.dim() == 3:
        p_k = partial_key_cache.unsqueeze(0).expand(bsz, -1, -1, -1).permute(0, 2, 1, 3)
    else:
        p_k = partial_key_cache.permute(0, 2, 1, 3)
    scores = torch.matmul(spec_q_partial, p_k.unsqueeze(2).transpose(-1, -2))
    scores = scores / math.sqrt(num_channels)

    score_agg = scores.sum(dim=2).squeeze(2)
    return score_agg.permute(1, 0, 2).contiguous()


# =============================================================================
# Partial Key Cache Build/Update for Block-Level InfiniGen
# =============================================================================

def infinigen_build_partial_cache(full_k, skewing_matrix, partial_indices):
    """Build partial key cache from full post-RoPE keys during prefill.

    Applies skewing matrix per KV-head, then extracts partial dimensions.
    Supports both unbatched and batched inputs.

    Args:
        full_k: (seq_len, H_kv, D) or (B, seq_len, H_kv, D) — full post-RoPE keys
        skewing_matrix: (H_kv, D, D) — per-KV-head skewing
        partial_indices: (H_kv, nc) — selected dim indices

    Returns:
        partial_k: (seq_len, H_kv, nc) or (B, seq_len, H_kv, nc) — matching input batch dims
    """
    batched = full_k.dim() == 4
    if batched:
        B, seq_len, num_kv_heads, head_dim = full_k.shape
    else:
        seq_len, num_kv_heads, head_dim = full_k.shape
        B = 1
        full_k = full_k.unsqueeze(0)  # (1, seq, H_kv, D)

    skew = skewing_matrix.to(full_k.dtype).to(full_k.device)
    p_idx = partial_indices.to(full_k.device)
    nc = p_idx.shape[1]

    # (B, seq, H_kv, D) → (B*H_kv, seq, D)
    k_t = full_k.permute(0, 2, 1, 3).reshape(B * num_kv_heads, seq_len, head_dim)
    # Expand skew: (H_kv, D, D) → (B*H_kv, D, D)
    skew_exp = skew.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_kv_heads, head_dim, head_dim)
    k_skewed = torch.bmm(k_t, skew_exp)  # (B*H_kv, seq, D)

    # Extract partial dims
    idx_exp = p_idx.unsqueeze(1).expand(-1, seq_len, -1)  # (H_kv, seq, nc)
    idx_exp = idx_exp.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_kv_heads, seq_len, nc)
    partial_k = torch.gather(k_skewed, -1, idx_exp)  # (B*H_kv, seq, nc)

    partial_k = partial_k.view(B, num_kv_heads, seq_len, nc).permute(0, 2, 1, 3)  # (B, seq, H_kv, nc)

    if not batched:
        return partial_k.squeeze(0).contiguous()  # (seq, H_kv, nc)
    return partial_k.contiguous()  # (B, seq, H_kv, nc)


def infinigen_update_partial_cache(partial_key_cache, new_k, position,
                                   skewing_matrix, partial_indices):
    """Append single decode token to partial key cache.

    Supports both unbatched and batched caches.

    Args:
        partial_key_cache: (max_seq, H_kv, nc) or (B, max_seq, H_kv, nc)
        new_k: (H_kv, D) or (B, H_kv, D) — new key(s)
        position: int or (B,) tensor — position(s) to write at
        skewing_matrix: (H_kv, D, D) — per-KV-head skewing
        partial_indices: (H_kv, nc) — selected dim indices
    """
    skew = skewing_matrix.to(new_k.dtype).to(new_k.device)
    p_idx = partial_indices.to(new_k.device)

    if new_k.dim() == 2:
        # Unbatched: (H_kv, D)
        new_k_skewed = torch.einsum('hd,hde->he', new_k, skew)
        new_k_partial = torch.gather(new_k_skewed, -1, p_idx)
        if isinstance(position, int) and position < partial_key_cache.shape[-3]:
            if partial_key_cache.dim() == 3:
                partial_key_cache[position] = new_k_partial
            else:
                # Batched cache but unbatched key — write to all batches
                partial_key_cache[:, position] = new_k_partial
    else:
        # Batched: (B, H_kv, D)
        new_k_skewed = torch.einsum('bhd,hde->bhe', new_k, skew)  # (B, H_kv, D)
        new_k_partial = torch.gather(
            new_k_skewed, -1, p_idx.unsqueeze(0).expand(new_k.shape[0], -1, -1))  # (B, H_kv, nc)
        if isinstance(position, int):
            pos = position
            if pos < partial_key_cache.shape[-3]:
                if partial_key_cache.dim() == 4:
                    partial_key_cache[:, pos] = new_k_partial
                else:
                    partial_key_cache[pos] = new_k_partial[0]
        else:
            # Per-batch positions: position is (B,) tensor
            for b in range(new_k.shape[0]):
                pos = int(position[b].item())
                if pos < partial_key_cache.shape[-3]:
                    if partial_key_cache.dim() == 4:
                        partial_key_cache[b, pos] = new_k_partial[b]
                    else:
                        partial_key_cache[pos] = new_k_partial[b]


# =============================================================================
# InfiniGen background gather worker
# =============================================================================

class InfiniGenGatherWorker:
    """Background thread that runs CPU gather + non_blocking H2D for InfiniGen.

    Uses a bounded queue (depth *max_depth*) so the main thread can submit
    the next gather without blocking -- the GPU keeps running kernels instead
    of idling while the previous gather finishes.  The main thread only
    blocks when the queue is full (back-pressure).

    The main CUDA stream later does ``wait_event(prefetch_done_event)`` at
    step 9 in the per-layer forward -- a purely GPU-side stall that never
    blocks the Python thread.

    Supports optional kv_bias gathering for NOSA models: pass
    ``bias_src`` (the CPU bias tensor, e.g. total_cis reshaped to (-1, 1))
    and ``bias_dst`` (the GPU bias tensor, e.g. _kv_bias_gpu reshaped to
    (-1, 1)) in the submit() call.
    """

    _SENTINEL = None  # poison pill for shutdown

    def __init__(self, prefetch_stream, max_depth=0):
        self._stream = prefetch_stream
        self._queue = queue.Queue(maxsize=max_depth)
        # H2D phase breakdown (CPU-side timing, gated by enable_breakdown)
        self._bd_enabled = False
        self._bd_mask_index_us = 0.0
        self._bd_cpu_gather_us = 0.0
        self._bd_h2d_us = 0.0
        self._bd_scatter_us = 0.0
        self._bd_calls = 0
        # Pre-allocated pinned staging buffer for H2D. cpu_gather returns
        # pageable memory; copying through a pinned buffer enables true async
        # DMA and avoids the synchronous pageable->staging path that halves
        # PCIe throughput. Combined K+V buffer keeps the downstream GPU scatter
        # path unchanged whether the source cache is packed or split K/V.
        self._pinned_kv = None  # lazily allocated on first use
        self._worker_error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- public API (called from main thread) --------------------------------

    def submit(self, *, ce_next, prefetch_ready_event, prefetch_done_event,
               record_stats, bias_src=None, bias_dst=None):
        """Enqueue one gather task.  Returns immediately if the queue has
        room; blocks only when *max_depth* tasks are already queued.

        The BG thread gathers into pinned staging on CPU and then enqueues
        H2D + scatter on the prefetch stream.

        Args:
            ce_next: CacheEngine for the next layer
            prefetch_ready_event: CUDA event recorded after PREP finishes
            prefetch_done_event: CUDA event to record when H2D + scatter done
            record_stats: whether to record H2D stats
            bias_src: optional bias source tensor. Can be flattened `(N, 1)` or
                structured `(B, S, H)` (e.g. `total_cis`).
            bias_dst: optional bias destination tensor. Can be flattened `(M, 1)`
                or structured `(B, S_gpu, H)` (e.g. `_kv_bias_gpu`).
        """
        self._queue.put((ce_next, prefetch_ready_event,
                         prefetch_done_event, record_stats,
                         bias_src, bias_dst))

    def drain(self):
        """Block until every queued task has been processed."""
        self._queue.join()
        if self._worker_error is not None:
            raise RuntimeError("InfiniGen gather worker failed") from self._worker_error

    def shutdown(self):
        self.drain()
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=5)

    def enable_breakdown(self, enabled=True):
        self._bd_enabled = enabled

    def reset_breakdown(self):
        self._bd_mask_index_us = 0.0
        self._bd_cpu_gather_us = 0.0
        self._bd_h2d_us = 0.0
        self._bd_scatter_us = 0.0
        self._bd_calls = 0

    def get_breakdown(self):
        return {
            "mask_index_us": self._bd_mask_index_us,
            "cpu_gather_us": self._bd_cpu_gather_us,
            "h2d_us": self._bd_h2d_us,
            "scatter_us": self._bd_scatter_us,
            "calls": self._bd_calls,
        }

    # -- worker loop (runs on background thread) -----------------------------

    def _loop(self):
        while True:
            task = self._queue.get()
            if task is self._SENTINEL:
                self._queue.task_done()
                return
            ce_next, ready_evt, done_evt, record_stats, bias_src, bias_dst = task
            try:
                self._do_gather(ce_next, ready_evt, done_evt, record_stats,
                                bias_src, bias_dst)
            except Exception as exc:
                self._worker_error = exc
                self._queue.task_done()
                while True:
                    try:
                        pending = self._queue.get_nowait()
                    except queue.Empty:
                        return
                    self._queue.task_done()
                    if pending is self._SENTINEL:
                        return
            else:
                self._queue.task_done()

    @torch.inference_mode()
    def _do_gather(self, ce, ready_evt, done_evt, record_stats,
                   bias_src, bias_dst):
        # Pipeline: compute indices (CPU-blocks on load_mask.cpu()) ->
        # CPU gather into pinned staging -> enqueue H2D + scatter on the
        # prefetch stream.

        bd = self._bd_enabled
        if bd:
            _t0 = time.perf_counter()

        # -- Phase 1: index computation (CPU-blocks on load_mask.cpu()) --
        # load_mask.cpu() must run on the prefetch stream so it also
        # synchronises with the previous task's H2D (protecting pinned bufs).
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(ready_evt)
            if record_stats:
                ce.record_prefetch_stats()
            lm_cpu = ce._prefetch_load_mask.cpu()

        bs = ce.block_size
        _Hkv, _Bsz, _Topk = ce._prefetch_load_mask.shape
        _D = ce.head_dim
        _S = ce._k_cpu.shape[1]
        _gpu_seq = ce._k_gpu.shape[1]
        valid = lm_cpu >= 0
        has_miss = valid.any()

        src_linear = dst_linear = None
        bias_src_linear = None
        bias_src_gather = None
        bias_dst_gather = None
        M_rows = 0
        if has_miss:
            miss_h, miss_b, miss_slot = valid.nonzero(as_tuple=True)
            miss_blk = lm_cpu[miss_h, miss_b, miss_slot]
            token_offsets = torch.arange(bs, dtype=torch.long)
            dst_linear = (miss_b.unsqueeze(1) * _gpu_seq * _Hkv
                          + (miss_slot.unsqueeze(1) * bs
                             + token_offsets.unsqueeze(0)) * _Hkv
                          + miss_h.unsqueeze(1)).reshape(-1)
            src_token_pos = (miss_blk.unsqueeze(1) * bs
                             + token_offsets.unsqueeze(0)).clamp(
                                 max=ce.seq_length - 1)
            src_linear = (miss_b.unsqueeze(1) * _S * _Hkv
                          + src_token_pos * _Hkv
                          + miss_h.unsqueeze(1)).reshape(-1)
            M_rows = src_linear.shape[0]
            if bias_src is not None and bias_dst is not None:
                if bias_src.dim() == 3:
                    bias_src_batch_stride = bias_src.shape[1] * bias_src.shape[2]
                    bias_src_heads = bias_src.shape[2]
                    bias_src_linear = (
                        miss_b.unsqueeze(1) * bias_src_batch_stride
                        + src_token_pos * bias_src_heads
                        + miss_h.unsqueeze(1)
                    ).reshape(-1)
                    bias_src_gather = bias_src.reshape(-1, 1)
                else:
                    bias_src_linear = src_linear
                    bias_src_gather = bias_src
                if bias_dst.dim() == 3:
                    bias_dst_gather = bias_dst.reshape(-1, 1)
                else:
                    bias_dst_gather = bias_dst

        if bd:
            _t1 = time.perf_counter()

        # -- Phase 2: enqueue GPU pipeline on prefetch stream --
        _2D = 2 * _D
        if has_miss:
            pkv = self._pinned_kv
            if pkv is None or pkv.shape[0] < M_rows or pkv.shape[1] != _2D:
                self._pinned_kv = torch.empty(M_rows, _2D,
                                              dtype=ce._k_cpu.dtype,
                                              pin_memory=True)
                pkv = self._pinned_kv

        gathered_bias = None
        gather_bias_on_gpu = False
        if has_miss:
            packed_src = getattr(ce, "_kv_cpu_flat_packed", None)
            if packed_src is not None:
                gathered_kv = fast_cpu_gather(packed_src, src_linear)

                if bd:
                    _t2 = time.perf_counter()

                pkv[:M_rows].copy_(gathered_kv)
            else:
                k_flat = ce._k_cpu.reshape(-1, _D)
                v_flat = ce._v_cpu.reshape(-1, _D)
                gathered_k = fast_cpu_gather(k_flat, src_linear)
                gathered_v = fast_cpu_gather(v_flat, src_linear)

                if bd:
                    _t2 = time.perf_counter()

                pkv[:M_rows, :_D].copy_(gathered_k)
                pkv[:M_rows, _D:].copy_(gathered_v)
            if bias_src is not None and bias_dst is not None:
                if bias_src_gather.device.type == "cpu":
                    gathered_bias = fast_cpu_gather(bias_src_gather, bias_src_linear)
                else:
                    gather_bias_on_gpu = True

            if bd:
                _t3 = time.perf_counter()
        elif bd:
            _t2 = _t3 = time.perf_counter()

        with torch.cuda.stream(self._stream):
            self._stream.wait_event(ready_evt)
            if has_miss:
                gpu_kv = pkv[:M_rows].to(ce._k_gpu.device, non_blocking=True)
                gpu_k_flat = ce._k_gpu.reshape(-1, _D)
                gpu_v_flat = ce._v_gpu.reshape(-1, _D)
                dst_linear_gpu = dst_linear.to(ce._k_gpu.device,
                                               non_blocking=True)
                gpu_k_flat[dst_linear_gpu] = gpu_kv[:, :_D]
                gpu_v_flat[dst_linear_gpu] = gpu_kv[:, _D:]
                if gathered_bias is not None:
                    gpu_bias_flat = bias_dst_gather
                    gpu_bias_flat[dst_linear_gpu] = gathered_bias.to(
                        ce._k_gpu.device, non_blocking=True)
                elif gather_bias_on_gpu:
                    bias_src_linear_gpu = bias_src_linear.to(
                        bias_src_gather.device,
                        non_blocking=True,
                    )
                    gathered_bias_gpu = bias_src_gather.index_select(0, bias_src_linear_gpu)
                    if gathered_bias_gpu.device != bias_dst_gather.device:
                        gathered_bias_gpu = gathered_bias_gpu.to(
                            bias_dst_gather.device,
                            non_blocking=True,
                        )
                    bias_dst_gather[dst_linear_gpu] = gathered_bias_gpu
            done_evt.record(self._stream)

        if bd:
            _t4 = time.perf_counter()
            self._bd_mask_index_us += (_t1 - _t0) * 1e6
            self._bd_cpu_gather_us += (_t2 - _t1) * 1e6
            self._bd_h2d_us += (_t3 - _t2) * 1e6
            self._bd_scatter_us += (_t4 - _t3) * 1e6
            self._bd_calls += 1
