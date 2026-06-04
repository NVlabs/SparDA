#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
All-in-one profiling + wall-time breakdown script.

Default mode uses in-process CUDA event timing (negligible overhead).
Optional --nsys mode wraps execution with Nsight Systems (heavier, but gives
kernel-level detail).

Usage:
    # Fast CUDA-event profiling (recommended):
    python profile_breakdown.py --model-path openbmb/NOSA-8B -B 16 -L 128K
    python profile_breakdown.py --model-path openbmb/NOSA-8B --no-offload -B 4 -L 16K
    python profile_breakdown.py --model-path openbmb/MiniCPM4.1-8B

    # With Nsight Systems (slower, for kernel-level detail):
    python profile_breakdown.py --nsys --model-path openbmb/NOSA-8B -B 16 -L 128K

    # Reuse existing .nsys-rep (skip profiling):
    python profile_breakdown.py --nsys --model-path openbmb/NOSA-8B -B 16 -L 128K --skip-profile

    # Parse an existing .nsys-rep directly:
    python profile_breakdown.py --parse my_report.nsys-rep

    # Custom output prefix:
    python profile_breakdown.py --model-path openbmb/NOSA-8B -B 16 -L 128K --output-prefix my_run

Output:
    <prefix>_nvtx_breakdown.csv    per-component timing CSV
    <prefix>_breakdown.png         horizontal stacked bar chart
    (--nsys only) <prefix>.nsys-rep  Nsight Systems trace
"""

import argparse
import builtins
import csv
import gc
import os
import sqlite3
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Ensure local nosi package is importable (prioritize repo over site-packages).
# nosi package lives at NOSA/nosi/nosi/, so NOSA/nosi/ must be on sys.path.
_NOSI_ROOT = str(Path(__file__).resolve().parents[2] / "nosi")
if _NOSI_ROOT not in sys.path:
    sys.path.insert(0, _NOSI_ROOT)

_BENCHMARKS_ROOT = str(Path(__file__).resolve().parents[1])
if _BENCHMARKS_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARKS_ROOT)

# ─── Suppress noisy warnings before any imports ────────────────────────────
warnings.filterwarnings("ignore", message=".*rope_scaling.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*new version of the following files.*")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*malicious code.*")

_original_print = builtins.print

def _quiet_print(*args, **kwargs):
    if args and "Use InfLLMv2" in str(args[0]):
        return
    _original_print(*args, **kwargs)

builtins.print = _quiet_print

import torch
from transformers import AutoTokenizer

from bench_utils import parse_seq_len, prepare_input_batches
from torch_load_compat import torch_load_compat

NATIVE_MAX = {"nosa": 32768, "minicpm": 65536}


def detect_method(model_path: str) -> str:
    combined = str(model_path).lower()
    if "nosa" in combined:
        return "nosa"
    if "minicpm" in combined:
        return "minicpm"
    return "other"

# ════════════════════════════════════════════════════════════════════════════
#  NVTX label → Figure-9 component mapping
# ════════════════════════════════════════════════════════════════════════════

COMPONENT_RULES = [
    # ── Primary decode labels (one canonical label per component) ────
    # Order matters: more-specific substrings must precede less-specific.
    ("o linear",       "O Linear"),        # must precede "linear"
    ("offloading_l0",  "Offloading (L0)"),
    ("offloading_l1p", "Offloading (L1+)"),
    ("kv_prep",        "Offloading (L1+)"),
    ("offloading",     "Offloading"),
    ("linear",         "QKV Linear"),
    ("rope",           "RoPE"),
    ("compress",       "Compression"),     # matches compress, compress k, compress cis, compression
    ("attention",      "Attention"),
    ("stage 1",        "Stage-1"),         # matches stage 1, real stage 1
    ("pooling",        "Pooling"),         # matches pooling, kernel: max pooling
    ("h2d_prefetch",   "H2D Total"),
    ("h2d_wait",       "H2D Exposed"),
    ("stage 2",        "Stage-2"),
    ("ffn",            "FFN"),             # matches ffn, ffn forward norm/actual
    # ── Warmup / legacy fallback labels ──────────────────────────────
    ("qkv_proj",       "QKV Linear"),
    ("split and cis",  "QKV Linear"),
]

COMPONENT_ORDER = [
    "QKV Linear", "RoPE", "Compression", "Attention", "Stage-1", "Pooling",
    "Offloading (L0)", "Offloading (L1+)", "Offloading",
    "Stage-2", "O Linear", "FFN", "Others",
]

COMPONENT_COLORS = {
    "QKV Linear":       "#4C72B0",
    "RoPE":             "#C44E52",
    "Compression":      "#8172B2",
    "Attention":        "#B279A2",
    "Stage-1":          "#CCB974",
    "Pooling":          "#E5AE38",
    "Offloading (L0)":  "#DA8BC3",
    "Offloading (L1+)": "#C87BAF",
    "Offloading":       "#DA8BC3",
    "Stage-2":          "#D5BB67",
    "O Linear":         "#7A9E9F",
    "FFN":              "#8C8C8C",
    "Others":           "#E8E8E8",
    # Internal-only keys (not shown in main table/plot order):
    "H2D Total":        "#FF6347",
    "H2D Exposed":      "#FF4500",
}

DETAIL_LABEL_PREFIX = "detail:"

# NVTX labels to skip during aggregation to avoid double-counting.
# Two categories:
#   1. Parent ranges whose time is the sum of their children (e.g.
#      "decoupled_decode" wraps linear + rope + stage1 + stage2 + ffn).
#   2. Child ranges whose time is already included in a parent we DO
#      count (e.g. "kernel: stage 1 *" are internals of "stage 1").
SKIP_LABELS = {
    # top-level wrappers (parents)
    "decoupled_decode", "decoupled_no_offload",
    "decoupled_decode_infllmv2", "decoupled_no_offload_infllmv2",
    # gate wrappers
    "__prefill_timed__", "__decode_timed__",
    # SparDA prep wrapper (children are explicitly labeled)
    "prep_next_layer",
    # kernel-level children of "stage 1" (from infllmv2_attn_stage1_fast)
    "kernel: stage 1 prepare",
    "kernel: stage 1 reshape",
    "kernel: stage 1",
    # kernel-level children from the non-fast variant (warmup; usually
    # already excluded by the gate, but belt-and-suspenders)
    "kernel: stage 1 contiguous",
    "kernel: stage 1 after",
}

# NVTX labels unique to the warmup / prefill path — never emitted during
# steady-state decode.  Used for temporal filtering: the end of the last
# warmup-only range marks the prefill+warmup → decode transition.
WARMUP_ONLY_LABELS = frozenset(x.lower() for x in [
    "qkv_proj", "split and cis", "compress cis",
    "real stage 1", "kernel: max pooling", "_offloading update",
    "ffn forward norm", "ffn forward actual",
])


def classify_label(label: str) -> str:
    low = label.strip().lower()
    for substring, component in COMPONENT_RULES:
        if substring.lower() in low:
            return component
    return "Others"


# ════════════════════════════════════════════════════════════════════════════
#  CUDA Event Profiler — negligible overhead
# ════════════════════════════════════════════════════════════════════════════

PREFILL_GATE = "__prefill_timed__"
DECODE_GATE = "__decode_timed__"
PROFILE_GATES = (PREFILL_GATE, DECODE_GATE)


class CudaEventProfiler:
    """Monkey-patches torch.cuda.nvtx to record CUDA events at push/pop
    boundaries.  Uses a pre-allocated event pool to avoid per-call
    cudaEventCreate overhead (~5-10us → ~0.5us per record).

    Only events inside the ``__prefill_timed__`` and ``__decode_timed__`` NVTX
    markers are collected, so warmup-decode events are automatically excluded.
    """

    _POOL_CHUNK = 4096   # events allocated per grow step

    def __init__(self):
        self._stack = []       # [(label, pool_idx | -1, gate_label | None, prev_gate), ...]
        self._ranges = []      # [(label, start_idx, end_idx, gate_label), ...]
        self._collecting = False
        self._active_gate = None
        self._saw_gates = set()
        self._orig_push = torch.cuda.nvtx.range_push
        self._orig_pop = torch.cuda.nvtx.range_pop
        # Labels to skip event recording for (saves 2 cudaEventRecord per skip).
        # Gate labels are excluded because we need their timing for "Others" computation.
        self._skip_labels = frozenset(SKIP_LABELS - set(PROFILE_GATES))
        # Pre-allocated CUDA event pool
        self._pool = []        # list of torch.cuda.Event
        self._pool_next = 0    # next free index
        self._grow_pool()

    # ── event pool ─────────────────────────────────────────────────────────

    def _grow_pool(self):
        """Extend the pool by _POOL_CHUNK pre-allocated CUDA events."""
        new_events = [torch.cuda.Event(enable_timing=True)
                      for _ in range(self._POOL_CHUNK)]
        self._pool.extend(new_events)

    def _alloc_event(self):
        """Return the index of a free event and record it on the current stream."""
        idx = self._pool_next
        if idx >= len(self._pool):
            self._grow_pool()
        self._pool_next += 1
        self._pool[idx].record()
        return idx

    # ── lifecycle ─────────────────────────────────────────────────────────

    def install(self):
        """Replace nvtx.range_push/pop with instrumented versions."""
        profiler = self
        orig_push = self._orig_push
        orig_pop = self._orig_pop

        skip_labels = profiler._skip_labels

        def patched_push(label):
            orig_push(label)
            if not profiler._collecting:
                return
            if label in PROFILE_GATES:
                prev_gate = profiler._active_gate
                profiler._active_gate = label
                profiler._saw_gates.add(label)
                start_idx = profiler._alloc_event()
                profiler._stack.append((label, start_idx, label, prev_gate))
                return
            gate_label = profiler._active_gate
            if gate_label is None:
                # Outside the profiled gates — track nesting but
                # skip cudaEventRecord to avoid inflating overhead.
                profiler._stack.append((label, -1, None, None))
                return
            # Skip event recording for labels discarded in aggregate()
            # (parent wrappers, kernel sub-components). Saves ~10-20us per range.
            if label in skip_labels:
                profiler._stack.append((label, -1, gate_label, None))
                return
            start_idx = profiler._alloc_event()
            profiler._stack.append((label, start_idx, gate_label, None))

        def patched_pop():
            orig_pop()
            if not profiler._collecting:
                return
            if not profiler._stack:
                return
            label, start_idx, gate_label, prev_gate = profiler._stack.pop()
            if label in PROFILE_GATES:
                profiler._active_gate = prev_gate
                if start_idx >= 0:
                    end_idx = profiler._alloc_event()
                    profiler._ranges.append((label, start_idx, end_idx, gate_label))
                return
            if start_idx >= 0:
                end_idx = profiler._alloc_event()
                profiler._ranges.append((label, start_idx, end_idx, gate_label))

        torch.cuda.nvtx.range_push = patched_push
        torch.cuda.nvtx.range_pop = patched_pop

    def uninstall(self):
        torch.cuda.nvtx.range_push = self._orig_push
        torch.cuda.nvtx.range_pop = self._orig_pop

    def start(self):
        self._collecting = True

    def stop(self):
        self._collecting = False

    def reset(self):
        self._stack.clear()
        self._ranges.clear()
        self._active_gate = None
        self._saw_gates.clear()
        self._pool_next = 0     # reuse pool slots

    # ── results ───────────────────────────────────────────────────────────

    def get_ranges(self, gate_label=None):
        """Synchronize GPU then return [(label, duration_us), ...]."""
        torch.cuda.synchronize()
        results = []
        if gate_label is not None and gate_label not in self._saw_gates:
            print(f"  WARNING: {gate_label} NVTX gate not seen; no gated ranges collected.")
        for label, s_idx, e_idx, recorded_gate in self._ranges:
            if gate_label is not None and recorded_gate != gate_label:
                continue
            dur_ms = self._pool[s_idx].elapsed_time(self._pool[e_idx])
            results.append((label, dur_ms * 1000.0))   # → microseconds
        return results


# ════════════════════════════════════════════════════════════════════════════
#  Run model with CUDA event profiling  (default mode)
# ════════════════════════════════════════════════════════════════════════════

# Alias kept for any internal references; canonical version is in bench_utils.
_parse_seq_len_int = parse_seq_len


class _BlockStatsTracker:
    """Track block hit-rate across decode steps for HF models.

    Compares consecutive decode steps' topk_idx to compute hit-rate (overlap).
    This is exactly the same metric as the nosi cache engine's diff_offload:
    block_map[t] = topk_idx[t], so miss[t+1] = entries in topk_idx[t+1] not
    found in topk_idx[t].

    Counting convention matches nosi:
      total = topk_idx.numel()      (all slots, including -1 padding)
      miss  = valid entries NOT found in prev
    This means -1 entries are implicitly counted as hits (no load needed),
    which is what diff_offload does (load_mask=-1 → not counted as miss).
    """

    def __init__(self, num_layers, block_size=64, topk=64,
                 head_dim=128, num_kv_heads=8, elem_size=2):
        self.num_layers = num_layers
        self.block_size = block_size
        self.topk = topk
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.elem_size = elem_size  # 2 for bf16
        self._prev = [None] * num_layers
        # per-layer accumulators (match nosi: total=numel, miss=valid not in prev)
        self._miss = [0] * num_layers
        self._total = [0] * num_layers
        self._steps = [0] * num_layers
        # Cross-backend comparison dump
        self._dump_path = os.environ.get("INFLLMV2_DUMP_TOPK", "")
        self._topk_log = {}
        self._current_step = 0

    def step(self, layer_idx, topk_idx):
        """Record one decode step's topk_idx for a layer.

        Args:
            layer_idx: Layer index.
            topk_idx: [num_kv_heads, batch_size, topk] block indices (int32).
                      Padding entries should be -1.
        """
        if topk_idx is None:
            return
        curr = topk_idx.detach()  # [H, B, K]
        prev = self._prev[layer_idx]
        if prev is not None and prev.shape == curr.shape:
            # For each (h, b, k): is curr[h,b,k] found anywhere in prev[h,b,:]?
            # curr[:,:,:,None] == prev[:,:,None,:]  -> [H, B, K, K]
            in_prev = (curr.unsqueeze(-1) == prev.unsqueeze(-2)).any(-1)  # [H,B,K]
            valid = curr != -1
            # miss = valid entries NOT in prev (same as diff_offload load_mask>=0)
            miss = int((valid & ~in_prev).sum().item())
            # total = numel (same as nosi _load_mask.numel())
            total = curr.numel()
            self._miss[layer_idx] += miss
            self._total[layer_idx] += total
            self._steps[layer_idx] += 1
        # First step (prev is None): only seed _prev, don't count in stats.
        self._prev[layer_idx] = curr.clone()
        if self._dump_path:
            self._topk_log[(self._current_step, layer_idx)] = {
                "topk_idx": curr.cpu().clone(),
            }

    def end_step(self):
        """Advance the step counter after all layers have been processed."""
        self._current_step += 1

    def save_dump(self):
        """Save collected topk entries to disk."""
        if self._dump_path and self._topk_log:
            import torch
            torch.save(self._topk_log, self._dump_path)
            print(f"[DUMP] Saved {len(self._topk_log)} HF topk entries "
                  f"to {self._dump_path}")

    def get_stats(self):
        """Return stats dict compatible with ``_print_h2d_stats``.

        Produces the same structure as ``InfLLMv2CacheEngine.get_h2d_stats()``
        so the existing print routine works unchanged.
        """
        bytes_per_miss = self.block_size * self.head_dim * self.elem_size * 2  # K+V

        per_layer = []
        agg_miss = 0
        agg_total = 0
        for i in range(self.num_layers):
            total = self._total[i]
            miss = self._miss[i]
            hr = 1.0 - miss / total if total > 0 else 0.0
            per_layer.append({
                "pf_calls": 0, "pf_total": 0, "pf_miss": 0,
                "corr_calls": self._steps[i],
                "corr_total": total, "corr_miss": miss,
                "total_blocks": total, "miss_blocks": miss,
                "hit_rate": hr,
                "h2d_bytes": miss * bytes_per_miss,
            })
            agg_miss += miss
            agg_total += total

        agg_hr = 1.0 - agg_miss / agg_total if agg_total > 0 else 0.0
        return {
            "miss_blocks": agg_miss,
            "total_blocks": agg_total,
            "hit_rate": agg_hr,
            "h2d_bytes": agg_miss * bytes_per_miss,
            "true_miss": agg_miss,
            "wasted_reload": 0,
            "block_size": self.block_size,
            "head_dim": self.head_dim,
            "elem_size": self.elem_size,
            "num_layers": self.num_layers,
            "per_layer": per_layer,
            "hf_backend": True,  # HF mode: no actual H2D transfer
        }


class _HFModelWrapper:
    """Wrapper around a HuggingFace model providing the same
    batch_generate_benchmark() interface as the nosi Llama class,
    so the CUDA-event profiling loop can work with both backends."""

    def __init__(self, model, max_new_tokens_default=4):
        self.model = model
        self._max_new_tokens_default = max_new_tokens_default
        self._collect_h2d_stats = False
        self._last_cache_engine = None  # HF has no cache_engine
        self._block_stats = None  # populated when _collect_h2d_stats is True
        self._logits_keep_kw = self._detect_logits_keep_kwarg(model)

    # ------------------------------------------------------------------
    #  Block-stats helpers
    # ------------------------------------------------------------------
    def _get_attn_layers(self):
        """Return list of attention sub-modules that expose _last_topk_idx."""
        inner = getattr(self.model, 'model', None)
        if inner is None or not hasattr(inner, 'layers'):
            return []
        layers = []
        for layer in inner.layers:
            attn = getattr(layer, 'self_attn', None)
            if attn is not None:
                layers.append(attn)
        return layers

    def _enable_block_stats(self, enable=True):
        """Toggle _collect_block_stats on every attention layer."""
        for attn in self._get_attn_layers():
            attn._collect_block_stats = enable
            # Release any cached topk tensor when disabling to avoid
            # retaining large prefill tensors on GPU.
            if not enable and hasattr(attn, '_last_topk_idx'):
                attn._last_topk_idx = None

    def _make_tracker(self):
        """Create a _BlockStatsTracker from the model config."""
        cfg = self.model.config
        sparse_cfg = getattr(cfg, 'sparse_config', None) or {}
        num_layers = cfg.num_hidden_layers
        block_size = int(sparse_cfg.get('block_size', 64))
        topk = int(sparse_cfg.get('topk', 64))
        head_dim = getattr(cfg, 'head_dim',
                           cfg.hidden_size // cfg.num_attention_heads)
        num_kv_heads = getattr(cfg, 'num_key_value_heads',
                               cfg.num_attention_heads)
        return _BlockStatsTracker(
            num_layers=num_layers, block_size=block_size, topk=topk,
            head_dim=head_dim, num_kv_heads=num_kv_heads, elem_size=2)

    def _collect_topk_step(self, tracker):
        """Read _last_topk_idx from each attention layer and feed the tracker."""
        for idx, attn in enumerate(self._get_attn_layers()):
            topk_idx = getattr(attn, '_last_topk_idx', None)
            if topk_idx is not None:
                tracker.step(idx, topk_idx)
            if tracker._dump_path:
                entry = tracker._topk_log.get((tracker._current_step, idx))
                if entry is not None:
                    extras = getattr(attn, '_last_dump_extras', None)
                    if extras:
                        entry["q_vec"] = extras.get("q_vec")
                        entry["ck"] = extras.get("ck")
                    raw = getattr(attn, '_last_raw_scores', None)
                    if raw is not None:
                        entry["raw_scores"] = raw
        tracker.end_step()

    @staticmethod
    def _detect_logits_keep_kwarg(model):
        """Inspect model.forward() signature once to find the right kwarg name."""
        import inspect
        sig = inspect.signature(model.forward)
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in sig.parameters:
                return name
        return None

    def _model_forward(self, **kwargs):
        """Forward with last-token-only logits to avoid full [B, S, V] allocation."""
        if self._logits_keep_kw is not None:
            kwargs[self._logits_keep_kw] = 1
        return self.model(**kwargs)

    def _prepare_infinigen_runtime(self, max_new_tokens):
        hooks = []
        if hasattr(self.model, "_prepare_infinigen_request_state"):
            self.model._prepare_infinigen_request_state(max_new_tokens)

        if hasattr(self.model, "_setup_infinigen_hooks") and hasattr(self.model, "_get_infinigen_attns"):
            if hasattr(self.model, "_ensure_infinigen_upgrade"):
                self.model._ensure_infinigen_upgrade()
            attns = self.model._get_infinigen_attns()
            hooks = self.model._setup_infinigen_hooks()
            return attns, hooks

        inner = getattr(self.model, "model", None)
        if inner is None or not hasattr(inner, "layers"):
            return [], hooks

        attns = []
        for layer in inner.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or not getattr(attn, "infinigen_enabled", False):
                continue
            attns.append(attn)
            if hasattr(attn, "_ig_max_new_tokens"):
                attn._ig_max_new_tokens = int(max_new_tokens)
            if hasattr(attn, "_ig_partial_key_cache"):
                attn._ig_partial_key_cache = None
            if hasattr(attn, "_ig_prev_hidden_states"):
                attn._ig_prev_hidden_states = None
            if hasattr(attn, "_ig_attn_input"):
                attn._ig_attn_input = None
        return attns, hooks

    def _clear_infinigen_runtime(self, attns, hooks):
        for h in hooks:
            h.remove()
        if hasattr(self.model, "_cleanup_infinigen_request_state"):
            self.model._cleanup_infinigen_request_state()
            return
        for attn in attns:
            if hasattr(attn, "previous_hidden_states"):
                attn.previous_hidden_states = None
            if hasattr(attn, "partial_key_cache"):
                attn.partial_key_cache = None
            if hasattr(attn, "_ig_prev_hidden_states"):
                attn._ig_prev_hidden_states = None
            if hasattr(attn, "_ig_partial_key_cache"):
                attn._ig_partial_key_cache = None
            if hasattr(attn, "_ig_attn_input"):
                attn._ig_attn_input = None
            if hasattr(attn, "_path_infinigen"):
                attn._path_infinigen = 0
            if hasattr(attn, "_path_fallback"):
                attn._path_fallback = 0

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def batch_generate_benchmark(self, input_ids, max_new_tokens=None):
        import time as _time
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens_default
        B = input_ids.shape[0]
        profile_verbose = bool(getattr(self, "_profile_verbose", False))
        collect_split_timing = bool(getattr(self, "_collect_prefill_decode_timings", False))
        self._last_prefill_time_s = None
        self._last_decode_time_s = None

        collect = self._collect_h2d_stats
        tracker = self._make_tracker() if collect else None
        ig_attns, ig_hooks = self._prepare_infinigen_runtime(max_new_tokens)
        if collect:
            self._block_stats = None
            # Decode-only stats: keep disabled during prefill to avoid
            # caching very large prefill topk tensors.
            self._enable_block_stats(False)

        try:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
            # Prefill
            if collect_split_timing:
                torch.cuda.synchronize()
                prefill_beg = _time.time()
            torch.cuda.nvtx.range_push(PREFILL_GATE)
            output = self._model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            if profile_verbose:
                torch.cuda.nvtx.range_push(f"{DETAIL_LABEL_PREFIX} token select")
            next_ids = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if profile_verbose:
                torch.cuda.nvtx.range_pop()
            past_key_values = output.past_key_values
            gen_ids = [next_ids]
            torch.cuda.nvtx.range_pop()
            if collect_split_timing:
                torch.cuda.synchronize()
                self._last_prefill_time_s = max(0.0, _time.time() - prefill_beg)

            if collect:
                self._enable_block_stats(True)

            # First decode (warmup, excluded from timing)
            ones_col = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, ones_col], dim=1)
            output = self._model_forward(
                input_ids=next_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_ids = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values
            gen_ids.append(next_ids)
            # Seed tracker with warmup step's topk (so first timed step has a prev)
            if collect:
                self._collect_topk_step(tracker)

            torch.cuda.synchronize()
            beg = _time.time()
            torch.cuda.nvtx.range_push("__decode_timed__")

            for _ in range(max_new_tokens - 2):
                ones_col = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, ones_col], dim=1)
                output = self._model_forward(
                    input_ids=next_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_ids = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = output.past_key_values
                gen_ids.append(next_ids)
                if collect:
                    self._collect_topk_step(tracker)

            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            end = _time.time()
            if collect_split_timing:
                self._last_decode_time_s = max(0.0, end - beg)

            if collect:
                self._block_stats = tracker.get_stats()
                tracker.save_dump()

            gen_ids = torch.cat(gen_ids, dim=-1)
            thru = B * (max_new_tokens - 2) / (end - beg) if (end - beg) > 0 else 0
            return gen_ids, thru
        finally:
            if collect:
                self._enable_block_stats(False)
            if ig_attns or ig_hooks:
                self._clear_infinigen_runtime(ig_attns, ig_hooks)


def _load_hf_model(args, model_path):
    """Load a HuggingFace model (mirrors bench.py run_hf logic)."""
    import logging
    from transformers import AutoModelForCausalLM, AutoConfig

    method = detect_method(model_path)
    probed_cfg = None
    is_minicpm_model = False
    if method == "minicpm":
        try:
            probed_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            is_minicpm_model = (getattr(probed_cfg, "model_type", "") == "minicpm")
        except Exception:
            is_minicpm_model = ("minicpm" in str(model_path).lower())
        if getattr(args, "indexer_path", None) and not getattr(args, "sparda", False):
            print("[SparDA] --indexer-path provided; enabling SparDA mode for HF MiniCPM.")
            args.sparda = True

    use_minicpm_infinigen = (
        method == "minicpm" and
        getattr(args, "infinigen", False) and
        not getattr(args, "dense", False)
    )

    use_minicpm_loader = (
        method == "minicpm" and
        not getattr(args, "dense", False) and
        not use_minicpm_infinigen and
        (getattr(args, "sparda", False) or is_minicpm_model)
    )

    if use_minicpm_infinigen:
        warmup_budget = int(args.max_new_tokens) + 2
        print("[HF Loader] using MiniCPM InfiniGen loader")
        logging.info("Loading MiniCPM InfiniGen from modeling_minicpm_infinigen.py")
        minicpm_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models", "minicpm"))
        if minicpm_dir not in sys.path:
            sys.path.append(minicpm_dir)
        from modeling_minicpm_infinigen import MiniCPMForCausalLM_InfiniGen
        from modeling_minicpm import apply_minicpm41_128k_longrope

        config = probed_cfg if probed_cfg is not None else AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(args, "long_context", False):
            apply_minicpm41_128k_longrope(config)
            print("[LongContext] InfiniGen: applied official MiniCPM4.1 128K LongRoPE factors.")
        default_sparse_config = {
            "kernel_size": 32, "kernel_stride": 16, "init_blocks": 1,
            "block_size": 64, "window_size": 2048, "topk": 96,
            "use_nope": False, "dense_len": -1,
        }
        sparse_cfg = default_sparse_config.copy()
        sparse_cfg.update({k: v for k, v in (getattr(config, "sparse_config", None) or {}).items() if v is not None})
        block_size = int(sparse_cfg.get("block_size", 64))
        window_size = int(sparse_cfg.get("window_size", 2048))
        local_blocks = (window_size // block_size) if block_size > 0 else 0
        nosi_topk = int(sparse_cfg.get("topk", 64))
        sparse_cfg["topk"] = max(1, nosi_topk - local_blocks)
        sparse_cfg["use_q_future_for_topk"] = False
        sparse_cfg["create_indexer"] = False
        config.sparse_config = sparse_cfg
        config._attn_implementation = "flash_attention_2"

        model = MiniCPMForCausalLM_InfiniGen.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model.config.sparse_config = config.sparse_config
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "dense_len"):
                layer.self_attn.dense_len = -1

        warmup_device = next(model.parameters()).device
        warmup_ids = torch.randint(0, config.vocab_size, (1, 2048), device=warmup_device)
        model.infinigen_warmup(input_ids=warmup_ids, max_new_tokens=warmup_budget)
        print("[HF Loader] MiniCPM InfiniGen warmup complete")
    elif use_minicpm_loader:
        mode_note = "sparda mode" if getattr(args, "sparda", False) else "auto MiniCPM mode"
        print(f"[HF Loader] using MiniCPM loader ({mode_note})")
        logging.info(f"Loading MiniCPM infllmv2 from modeling_minicpm.py ({mode_note})")
        minicpm_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models", "minicpm"))
        if minicpm_dir not in sys.path:
            sys.path.append(minicpm_dir)
        from modeling_minicpm import (
            MiniCPMForCausalLM as MiniCPMForCausalLM_Future,
            apply_minicpm41_128k_longrope,
        )
        from transformers import GenerationConfig

        config = probed_cfg if probed_cfg is not None else AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(args, "long_context", False):
            apply_minicpm41_128k_longrope(config)
            print("[LongContext] infllmv2: applied official MiniCPM4.1 128K LongRoPE factors.")
        default_sparse_config = {
            "kernel_size": 32, "kernel_stride": 16, "init_blocks": 1,
            "block_size": 64, "window_size": 2048, "topk": 96,
            "use_nope": False, "dense_len": -1,
        }
        config.sparse_config = default_sparse_config.copy()
        # Align MiniCPM-future topk semantics with nosi:
        #   MiniCPM-future effective_topk = sparse_config["topk"] + local_blocks
        #   nosi effective_topk           = sparse_config["topk"]
        # So set HF base topk = max(1, nosi_topk - local_blocks).
        block_size = int(config.sparse_config.get("block_size", 64))
        window_size = int(config.sparse_config.get("window_size", 2048))
        local_blocks = (window_size // block_size) if block_size > 0 else 0
        nosi_topk = int(config.sparse_config.get("topk", 64))
        hf_base_topk = max(1, nosi_topk - local_blocks)
        if hf_base_topk != nosi_topk:
            print(
                f"[HF Loader] topk semantics aligned: "
                f"nosi_topk={nosi_topk}, local_blocks={local_blocks}, "
                f"hf_base_topk={hf_base_topk}, hf_effective_topk={hf_base_topk + local_blocks}"
            )
        config.sparse_config["topk"] = hf_base_topk
        config.sparse_config['use_q_future_for_topk'] = bool(getattr(args, "sparda", False))
        config._attn_implementation = "flash_attention_2"

        model = MiniCPMForCausalLM_Future.from_pretrained(
            model_path, config=config, trust_remote_code=True,
            device_map="auto", torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model.config.sparse_config = config.sparse_config
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'use_q_future_for_topk'):
                layer.self_attn.use_q_future_for_topk = bool(getattr(args, "sparda", False))
            if hasattr(layer.self_attn, 'dense_len'):
                layer.self_attn.dense_len = -1

        # Load indexer weights if provided
        if getattr(args, "indexer_path", None):
            checkpoint = torch_load_compat(args.indexer_path, map_location='cpu')
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model_state_dict'), dict):
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and isinstance(checkpoint.get('state_dict'), dict):
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                # Fallback: root dict itself may be a state_dict (+ optional metadata keys)
                tensor_items = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
                state_dict = tensor_items
            else:
                state_dict = {}
            is_full = bool(checkpoint.get('full_model', False)) if isinstance(checkpoint, dict) else False

            def _map_state_dict_to_model_keys(src_state_dict, model_state_keys):
                """Map checkpoint keys to target model keys by stripping common wrappers."""
                prefixes = (
                    "module.", "model.", "model.model.",
                    "module.model.", "module.model.model.",
                    "_orig_mod.", "base_model.model.",
                )
                mapped = {}
                for src_k, v in src_state_dict.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    variants = [src_k]
                    # Repeatedly strip known prefixes (e.g., module.model.)
                    changed = True
                    while changed:
                        changed = False
                        for p in prefixes:
                            cur = variants[-1]
                            if cur.startswith(p):
                                nxt = cur[len(p):]
                                variants.append(nxt)
                                changed = True
                                break
                    candidates = []
                    for k in variants:
                        candidates.append(k)
                        if not k.startswith("model."):
                            candidates.append("model." + k)
                        if k.startswith("model."):
                            candidates.append(k[len("model."):])
                    tgt_k = next((c for c in candidates if c in model_state_keys), None)
                    if tgt_k is not None:
                        mapped[tgt_k] = v
                return mapped

            if is_full:
                mapped_full = _map_state_dict_to_model_keys(state_dict, set(model.state_dict().keys()))
                model.load_state_dict(mapped_full, strict=False)
                logging.info(f"Loaded FULL model weights ({len(state_dict)} tensors)")
            else:
                indexer_weights = {k: v for k, v in state_dict.items()
                                   if 'q_future_proj' in k or 'q_curr_proj' in k}
                if indexer_weights:
                    model_state_keys = set(model.state_dict().keys())
                    mapped_indexer = _map_state_dict_to_model_keys(indexer_weights, model_state_keys)
                    expected_indexer = sorted(
                        k for k in model_state_keys
                        if ('q_future_proj' in k or 'q_curr_proj' in k)
                    )
                    missing_indexer = [k for k in expected_indexer if k not in mapped_indexer]
                    if missing_indexer:
                        preview = ", ".join(missing_indexer[:5])
                        raise RuntimeError(
                            f"Indexer checkpoint missing {len(missing_indexer)} expected tensors. "
                            f"Examples: {preview}"
                        )
                    model.load_state_dict(mapped_indexer, strict=False)
                    print(
                        f"[Indexer] loaded {len(mapped_indexer)}/{len(expected_indexer)} tensors "
                        f"from {args.indexer_path}"
                    )
                    logging.info(
                        f"Loaded {len(mapped_indexer)}/{len(expected_indexer)} indexer tensors "
                        f"(source tensors: {len(indexer_weights)})"
                    )
                else:
                    raise RuntimeError(
                        f"No q_future_proj/q_curr_proj tensors found in indexer checkpoint: {args.indexer_path}"
                    )

        if model.generation_config is None:
            model.generation_config = GenerationConfig.from_model_config(config)
    elif getattr(args, "dense", False):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="cuda",
        )
    else:
        if method == "nosa":
            nosa_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "nosa"))
            if nosa_dir not in sys.path:
                sys.path.insert(0, nosa_dir)
            import modeling_llama_nosa as _hf_mod
        else:
            raise ValueError(f"HF backend: cannot detect method from model path {model_path}")

        if method == "nosa" and args.long_context:
            if getattr(args, "yarn", False):
                _hf_mod.LONG_CONTEXT_ENABLED = True
                _hf_mod.LONG_CONTEXT_ROPE_THETA = None
            else:
                _hf_mod.LONG_CONTEXT_ENABLED = False
                _hf_mod.LONG_CONTEXT_ROPE_THETA = 40000
        else:
            _hf_mod.LONG_CONTEXT_ENABLED = False
            _hf_mod.LONG_CONTEXT_ROPE_THETA = None
        SparseLlamaForCausalLM = _hf_mod.SparseLlamaForCausalLM

        from transformers import AutoConfig as _AutoConfig
        hf_config = _AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_sparse_cfg = dict(getattr(hf_config, 'sparse_config', None) or {})
        checkpoint_state = None
        checkpoint_sparse_cfg = {}
        if getattr(args, "indexer_path", None):
            checkpoint_state = torch_load_compat(args.indexer_path, map_location='cpu')
            if isinstance(checkpoint_state, dict):
                checkpoint_sparse_cfg = dict(checkpoint_state.get("sparse_config") or {})

        default_sparse = {
            "kernel_size": 32,
            "kernel_stride": 16,
            "init_blocks": 1,
            "block_size": 64,
            "window_size": 1024,  # NOSA local window = 16 blocks
            "topk": 80,           # Effective topk = 80 + 16 = 96
            "use_nope": False,
            "dense_len": 8192,
        }
        sparse_cfg = default_sparse.copy()
        sparse_cfg.update({k: v for k, v in model_sparse_cfg.items() if v is not None})
        if checkpoint_sparse_cfg:
            sparse_cfg.update(checkpoint_sparse_cfg)

        if getattr(args, 'infinigen', False) and method == "nosa":
            sparse_cfg['infinigen_enabled'] = True
            sparse_cfg['infinigen_num_channels'] = 32
            sparse_cfg['use_q_future_for_topk'] = False
            sparse_cfg['create_indexer'] = False
            hf_config.sparse_config = sparse_cfg
            print(f"[HF Loader] NOSA InfiniGen sparse_config: {hf_config.sparse_config}")
        elif getattr(args, 'sparda', False) and method == "nosa":
            sparse_cfg['use_q_future_for_topk'] = True
            sparse_cfg['create_indexer'] = True
            hf_config.sparse_config = sparse_cfg
            print(f"[HF Loader] NOSA SparDA sparse_config: {hf_config.sparse_config}")
        else:
            sparse_cfg['use_q_future_for_topk'] = False
            sparse_cfg['create_indexer'] = False
            hf_config.sparse_config = sparse_cfg

        model = SparseLlamaForCausalLM.from_pretrained(
            model_path, config=hf_config, device_map="cuda", torch_dtype=torch.bfloat16,
        )

        if getattr(args, 'infinigen', False) and method == "nosa":
            warmup_budget = int(args.max_new_tokens) + 2
            warmup_device = next(model.parameters()).device
            warmup_ids = torch.randint(0, hf_config.vocab_size, (1, 2048), device=warmup_device)
            model.infinigen_warmup(input_ids=warmup_ids, max_new_tokens=warmup_budget)
            print("[HF Loader] NOSA InfiniGen warmup complete")
        elif getattr(args, 'sparda', False) and getattr(args, 'indexer_path', None) and method == "nosa":
            sd = checkpoint_state.get('model_state_dict', checkpoint_state) if isinstance(checkpoint_state, dict) else {}
            iw = {k: v for k, v in sd.items() if 'q_future_proj' in k or 'q_curr_proj' in k}
            if iw:
                model.load_state_dict(iw, strict=False)
                print(f"[HF Loader] NOSA: loaded {len(iw)} indexer weights")

        # Ensure sparse_config is set for block stats tracker
        if not getattr(model.config, 'sparse_config', None):
            is_nosa_model = (method == "nosa")
            model.config.sparse_config = {
                "kernel_size": 32, "kernel_stride": 16, "init_blocks": 1,
                "block_size": 64,
                "window_size": 1024 if is_nosa_model else 2048,
                "topk": 80 if is_nosa_model else 96,
            }

    model.eval()
    return _HFModelWrapper(model, max_new_tokens_default=args.max_new_tokens)


def _collect_h2d_from_model(model):
    """Extract h2d_stats dict from the model after one batch_generate_benchmark call."""
    h2d = None
    if getattr(model, '_last_cache_engine', None) is not None:
        ce = model._last_cache_engine
        if hasattr(ce, 'get_h2d_stats'):
            candidate = ce.get_h2d_stats()
            if candidate and candidate.get("total_blocks", 0) > 0:
                h2d = candidate
    if h2d is None and getattr(model, '_block_stats', None) is not None:
        h2d = model._block_stats
    # Fallback: no-offload topk-overlap tracker from the nosi SparDA path
    if h2d is None and getattr(model, '_nooff_miss', None) is not None:
        _miss = model._nooff_miss
        _total = model._nooff_total
        num_layers = len(_miss)
        _sparse_cfg = getattr(getattr(model, 'config', None), 'sparse_config', None) or {}
        block_size = int(_sparse_cfg.get('block_size', 64))
        head_dim = getattr(model, 'head_dim', 128)
        elem_size = 2
        bytes_per_miss = block_size * head_dim * elem_size * 2
        per_layer = []
        agg_miss = agg_total = 0
        for i in range(num_layers):
            t = _total[i]; m = _miss[i]
            hr = 1.0 - m / t if t > 0 else 0.0
            per_layer.append({
                "pf_calls": 0, "pf_total": 0, "pf_miss": 0,
                "corr_calls": model._nooff_steps[i],
                "corr_total": t, "corr_miss": m,
                "total_blocks": t, "miss_blocks": m,
                "hit_rate": hr, "h2d_bytes": m * bytes_per_miss,
            })
            agg_miss += m; agg_total += t
        agg_hr = 1.0 - agg_miss / agg_total if agg_total > 0 else 0.0
        h2d = {
            "miss_blocks": agg_miss, "total_blocks": agg_total,
            "hit_rate": agg_hr, "h2d_bytes": agg_miss * bytes_per_miss,
            "true_miss": agg_miss, "wasted_reload": 0,
            "block_size": block_size, "head_dim": head_dim,
            "elem_size": elem_size, "num_layers": num_layers,
            "per_layer": per_layer, "nooff_topk_overlap": True,
        }
    return h2d


def _format_iter_h2d_line(iter_h2d):
    """Format one-line per-iteration H2D hit-rate summary."""
    if not iter_h2d:
        return None
    miss = int(iter_h2d.get("miss_blocks", 0))
    total = int(iter_h2d.get("total_blocks", 0))
    if total <= 0:
        return None
    hit_rate = iter_h2d.get("hit_rate")
    if hit_rate is None:
        hit_rate = 1.0 - miss / total
    return f"    h2d hit-rate: {hit_rate:.1%} ({total - miss:,}/{total:,})"


def _get_prefill_decode_metrics(model, batch_size, input_len, decode_tokens, decode_thru):
    """Return per-call prefill/decode timing metrics from the model wrapper."""
    decode_time_s = getattr(model, "_last_decode_time_s", None)
    if (decode_time_s is None or decode_time_s <= 0) and decode_thru > 0:
        decode_time_s = decode_tokens / decode_thru

    prefill_time_s = getattr(model, "_last_prefill_time_s", None)
    if prefill_time_s is None or prefill_time_s <= 0:
        prefill_time_s = 0.0

    prefill_tokens = batch_size * input_len
    prefill_thru = prefill_tokens / prefill_time_s if prefill_time_s > 0 else 0.0
    return {
        "prefill_tokens": prefill_tokens,
        "prefill_time_s": prefill_time_s,
        "prefill_thru": prefill_thru,
        "decode_tokens": decode_tokens,
        "decode_time_s": decode_time_s or 0.0,
        "decode_thru": decode_thru,
    }


def _print_profile_iter(prefix, idx, metrics):
    print(
        f"  {prefix} {idx}: prefill = {metrics['prefill_thru']:.2f} tok/s, "
        f"decode = {metrics['decode_thru']:.2f} tok/s"
    )


def _merge_h2d_stats(accumulated, new_stats):
    """Merge one iteration's h2d_stats into a running accumulator (in-place)."""
    if new_stats is None:
        return accumulated
    if accumulated is None:
        import copy
        return copy.deepcopy(new_stats)

    _SUM_KEYS = ('miss_blocks', 'total_blocks', 'h2d_bytes',
                 'true_miss', 'wasted_reload')
    for k in _SUM_KEYS:
        accumulated[k] = accumulated.get(k, 0) + new_stats.get(k, 0)
    total = accumulated['total_blocks']
    miss = accumulated['miss_blocks']
    accumulated['hit_rate'] = 1.0 - miss / total if total > 0 else 0.0

    acc_pl = accumulated.get('per_layer', [])
    new_pl = new_stats.get('per_layer', [])
    _PL_SUM_KEYS = ('pf_calls', 'pf_total', 'pf_miss',
                    'corr_calls', 'corr_total', 'corr_miss',
                    'total_blocks', 'miss_blocks', 'h2d_bytes')
    if len(acc_pl) == len(new_pl):
        for i in range(len(acc_pl)):
            for k in _PL_SUM_KEYS:
                acc_pl[i][k] = acc_pl[i].get(k, 0) + new_pl[i].get(k, 0)
            ti = acc_pl[i]['total_blocks']
            mi = acc_pl[i]['miss_blocks']
            acc_pl[i]['hit_rate'] = 1.0 - mi / ti if ti > 0 else 0.0
    return accumulated


def run_cuda_event_profile(args):
    """Load model in-process, run warmup + timed iterations, return gated ranges."""
    model_path = args.model_path
    method = detect_method(model_path)

    use_hf = (args.backend == "hf")

    # ── Load model ────────────────────────────────────────────────────
    if use_hf:
        model = _load_hf_model(args, model_path)
    else:
        if getattr(args, "infinigen", False):
            if method == "nosa":
                from nosi import NOSALlama as Llama
            else:
                from nosi import InfLLMv2Llama as Llama
            model = Llama(
                model_name=model_path, device="cuda",
                offload=not args.no_offload,
                offload_layer0=getattr(args, "offload_layer0", False),
                long_context=getattr(args, 'long_context', False),
                yarn=getattr(args, "yarn", False),
                infinigen=True,
            )
        else:
            if method == "nosa":
                from nosi import NOSALlama as Llama
            elif method == "minicpm":
                from nosi import InfLLMv2Llama as Llama
            else:
                raise ValueError(f"NOSI backend: cannot detect method from model path {model_path}")

            offload = not args.no_offload
            model_kwargs = dict(
                model_name=model_path, device="cuda", offload=offload,
                offload_layer0=getattr(args, "offload_layer0", False),
                long_context=getattr(args, 'long_context', False),
            )
            if method == "nosa":
                model_kwargs["yarn"] = getattr(args, "yarn", False)
            if getattr(args, "sparda", False):
                model_kwargs["decoupled"] = True
                model_kwargs["decoupled_no_prefetch"] = getattr(args, "sparda_no_prefetch", False)
            if getattr(args, "indexer_path", None):
                model_kwargs["indexer_path"] = args.indexer_path
            model = Llama(**model_kwargs)
            if getattr(args, "dense", False):
                model.force_dense_inference = True
                sparse_cfg = dict(getattr(model.config, "sparse_config", None) or {})
                sparse_cfg["force_dense_inference"] = True
                model.config.sparse_config = sparse_cfg

    verbose_mode = bool(getattr(args, "verbose", False))
    setattr(model, "_profile_verbose", verbose_mode)
    setattr(model, "_collect_prefill_decode_timings", True)
    setattr(
        model,
        "_collect_nooff_topk_stats",
        bool(verbose_mode and getattr(args, "no_offload", False) and not getattr(args, "dense", False)),
    )

    offload = not args.no_offload

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    B = args.batch_size
    L = parse_seq_len(args.seq_len)
    max_new_tokens = args.max_new_tokens
    test_n = args.test_n

    lc_str = "128K" if getattr(args, 'long_context', False) else "off"
    mode = "dense" if getattr(args, "dense", False) else ("infinigen" if getattr(args, "infinigen", False) else method)
    print(
        f"\n[Config] method={mode}, backend={args.backend}, offload={offload}, "
        f"sparda={getattr(args, 'sparda', False)}, "
        f"sparda_no_prefetch={getattr(args, 'sparda_no_prefetch', False)}, "
        f"layer0_cpu_offload={getattr(args, 'offload_layer0', False)}, "
        f"long_context={lc_str}"
    )
    print(f"[Setup]  batch={B}, input_len={L}, max_new_tokens={max_new_tokens}, "
          f"test_n={test_n}, model={model_path}")

    # ── Install profiler ──────────────────────────────────────────────
    profiler = CudaEventProfiler()
    profiler.install()

    # ── Collect qualifying samples (shared with bench.py) ─────────────
    n_batches = 1 + test_n   # 1 warmup + test_n paired baseline/profiled
    input_batches = prepare_input_batches(
        args.dataset_name, args.dataset_split, tokenizer,
        B, L, n_batches,
        model_path=model_path,
    )
    print(f"  {n_batches} batches (1 warmup + {test_n} paired)")
    decode_tokens = max_new_tokens * B
    prefill_tokens = B * L

    # ── Warmup (not profiled) ─────────────────────────────────────────
    print("\n--- Warmup (excluded from timing) ---")
    gen_ids, thru = model.batch_generate_benchmark(
        input_batches[0], max_new_tokens=max_new_tokens + 2)
    warmup_metrics = _get_prefill_decode_metrics(model, B, L, decode_tokens, thru)
    print(
        f"  warmup: prefill = {warmup_metrics['prefill_thru']:.2f} tok/s, "
        f"decode = {warmup_metrics['decode_thru']:.2f} tok/s"
    )
    gc.collect()
    torch.cuda.empty_cache()

    # ── Baseline iterations (profiler installed but NOT collecting) ────
    print(f"\n--- Baseline iterations ({test_n}, profiler installed, not collecting) ---")
    baseline_prefill_list = []
    baseline_decode_list = []
    for r in range(test_n):
        gen_ids, bl_thru = model.batch_generate_benchmark(
            input_batches[1 + r], max_new_tokens=max_new_tokens + 2)
        metrics = _get_prefill_decode_metrics(model, B, L, decode_tokens, bl_thru)
        baseline_prefill_list.append(metrics["prefill_thru"])
        baseline_decode_list.append(metrics["decode_thru"])
        _print_profile_iter("baseline", r + 1, metrics)
        gc.collect()
        torch.cuda.empty_cache()
    baseline_prefill_thru = sum(baseline_prefill_list) / len(baseline_prefill_list)
    baseline_decode_thru = sum(baseline_decode_list) / len(baseline_decode_list)
    print(
        f"  baseline avg: prefill = {baseline_prefill_thru:.2f} tok/s, "
        f"decode = {baseline_decode_thru:.2f} tok/s"
    )

    # ── Profiled iterations (same inputs as baseline) ─────────────────
    print(f"\n--- Profiled iterations ({test_n}) ---")
    model._collect_h2d_stats = True          # Enable for ALL iterations
    profiled_prefill_list = []
    profiled_decode_list = []
    h2d_stats = None                         # accumulator across iterations
    for r in range(test_n):
        profiler.start()
        gen_ids, thru = model.batch_generate_benchmark(
            input_batches[1 + r], max_new_tokens=max_new_tokens + 2)
        profiler.stop()

        # Collect and merge H2D stats from this iteration
        iter_h2d = _collect_h2d_from_model(model)
        h2d_stats = _merge_h2d_stats(h2d_stats, iter_h2d)

        metrics = _get_prefill_decode_metrics(model, B, L, decode_tokens, thru)
        profiled_prefill_list.append(metrics["prefill_thru"])
        profiled_decode_list.append(metrics["decode_thru"])
        _print_profile_iter("iter", r + 1, metrics)
        iter_h2d_line = _format_iter_h2d_line(iter_h2d)
        if iter_h2d_line:
            print(iter_h2d_line)
        gc.collect()
        torch.cuda.empty_cache()
    model._collect_h2d_stats = False

    # ── Collect results ───────────────────────────────────────────────
    prefill_ranges = profiler.get_ranges(PREFILL_GATE)
    decode_ranges = profiler.get_ranges(DECODE_GATE)
    profiler.uninstall()

    profiled_prefill_list.sort()
    profiled_decode_list.sort()
    median_prefill_thru = profiled_prefill_list[len(profiled_prefill_list) // 2]
    median_decode_thru = profiled_decode_list[len(profiled_decode_list) // 2]
    prefill_time = prefill_tokens / median_prefill_thru if median_prefill_thru > 0 else 0.0
    decode_time = decode_tokens / median_decode_thru if median_decode_thru > 0 else 0.0
    prefill_overhead_pct = (
        (baseline_prefill_thru / median_prefill_thru - 1) * 100
        if median_prefill_thru > 0 else 0.0
    )
    decode_overhead_pct = (
        (baseline_decode_thru / median_decode_thru - 1) * 100
        if median_decode_thru > 0 else 0.0
    )

    print(f"\n{'=' * 50}")
    print(f"  Prefill {prefill_tokens} tokens in {prefill_time:.3f} s")
    print(f"  Baseline:  {baseline_prefill_thru:.2f} tokens/s (no event collection)")
    print(f"  Profiled:  {median_prefill_thru:.2f} tokens/s (median over {len(profiled_prefill_list)} runs)")
    print(f"  Overhead:  {prefill_overhead_pct:+.1f}%")
    print()
    print(f"  Decode {decode_tokens} tokens in {decode_time:.3f} s")
    print(f"  Baseline:  {baseline_decode_thru:.2f} tokens/s (no event collection)")
    print(f"  Profiled:  {median_decode_thru:.2f} tokens/s (median over {len(profiled_decode_list)} runs)")
    print(f"  Overhead:  {decode_overhead_pct:+.1f}%")
    print(f"{'=' * 50}")

    ig_h2d_breakdown = model.get_ig_h2d_breakdown() if hasattr(model, 'get_ig_h2d_breakdown') else None
    return (
        prefill_ranges,
        decode_ranges,
        baseline_prefill_thru,
        median_prefill_thru,
        baseline_decode_thru,
        median_decode_thru,
        h2d_stats,
        ig_h2d_breakdown,
    )


# ════════════════════════════════════════════════════════════════════════════
#  Nsight Systems path (--nsys mode)
# ════════════════════════════════════════════════════════════════════════════

def build_output_prefix(args) -> str:
    if args.output_prefix:
        return args.output_prefix
    method = detect_method(args.model_path)
    parts = ["dense" if getattr(args, "dense", False) else method]
    if getattr(args, "infinigen", False):
        parts.append("infinigen")
    if args.no_offload:
        parts.append("gpu")
    if getattr(args, "sparda", False):
        parts.append("sparda")
    if getattr(args, "sparda_no_prefetch", False):
        parts.append("noprefetch")
    if args.backend != "nosi":
        parts.append(args.backend)
    parts.append(f"B{args.batch_size}")
    parts.append(f"L{args.seq_len}")
    return "_".join(parts)


def build_bench_cmd(args) -> list:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_py = os.path.join(script_dir, "bench.py")

    cmd = ["python", bench_py, "--model-path", args.model_path]
    cmd += ["--backend", args.backend]
    cmd += ["-B", str(args.batch_size)]
    cmd += ["-L", str(args.seq_len)]
    cmd += ["--max-new-tokens", str(args.max_new_tokens)]
    cmd += ["--test-n", str(args.test_n)]
    if getattr(args, "dense", False):
        cmd += ["--dense"]
    if getattr(args, "infinigen", False):
        cmd += ["--infinigen"]
    if args.no_offload:
        cmd += ["--no-offload"]
    if getattr(args, "offload_layer0", False):
        cmd += ["--offload-layer0"]
    if getattr(args, "sparda", False):
        cmd += ["--sparda"]
    if getattr(args, "sparda_no_prefetch", False):
        cmd += ["--sparda-no-prefetch"]
    if getattr(args, "indexer_path", None):
        cmd += ["--indexer-path", str(args.indexer_path)]
    if args.dataset_name != "emozilla/pg19":
        cmd += ["--dataset", args.dataset_name]
    if args.dataset_split != "test":
        cmd += ["--dataset-split", args.dataset_split]
    return cmd


def run_nsys_profiling(bench_cmd: list, prefix: str) -> str:
    rep_file = f"{prefix}.nsys-rep"

    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "16",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "false",
        "OMP_SCHEDULE": "static",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_AFFINITY": "granularity=fine,compact,1,0",
    })

    cmd = [
        "nsys", "profile",
        "-o", prefix,
        "--trace=cuda,nvtx",
        "--force-overwrite=true",
        "--capture-range=none",
    ] + bench_cmd

    print(f"\n>>> Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: nsys profile exited with code {result.returncode}")
        sys.exit(1)

    if not os.path.exists(rep_file):
        candidates = list(Path(".").glob(f"{prefix}*.nsys-rep"))
        if candidates:
            rep_file = str(candidates[0])
        else:
            print(f"ERROR: {rep_file} was not created.")
            sys.exit(1)

    return rep_file


# ════════════════════════════════════════════════════════════════════════════
#  Nsys extraction helpers
# ════════════════════════════════════════════════════════════════════════════

def _extract_sqlite(rep_path: str):
    """Return list of (label, start_ns, end_ns) from the NVTX table."""
    conn = sqlite3.connect(rep_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    nvtx_table = None
    for candidate in [
        "NVTX_EVENTS", "NVTX_GPU_PROJ_TRACE", "NVTX_PUSHPOP_TRACE",
        "nvtx_events", "NVTX_CPU_TRACE",
    ]:
        if candidate in tables:
            nvtx_table = candidate
            break
    if nvtx_table is None:
        for t in tables:
            if "nvtx" in t.lower():
                nvtx_table = t
                break
    if nvtx_table is None:
        raise RuntimeError(f"No NVTX table in {rep_path}. Tables: {tables}")

    cur.execute(f"PRAGMA table_info({nvtx_table})")
    cols = {r[1] for r in cur.fetchall()}

    if {"text", "start", "end"} <= cols:
        q = (f"SELECT text, start, end FROM {nvtx_table} "
             f"WHERE text IS NOT NULL AND end > start ORDER BY start")
    elif {"textId", "start", "end"} <= cols:
        if "StringIds" in tables:
            q = (f"SELECT s.value, n.start, n.end FROM {nvtx_table} n "
                 f"JOIN StringIds s ON n.textId = s.id "
                 f"WHERE s.value IS NOT NULL AND n.end > n.start "
                 f"ORDER BY n.start")
        else:
            q = (f"SELECT textId, start, end FROM {nvtx_table} "
                 f"WHERE end > start ORDER BY start")
    elif {"Name", "Start", "End"} <= cols:
        q = (f'SELECT Name, "Start", "End" FROM {nvtx_table} '
             f'WHERE Name IS NOT NULL AND "End" > "Start" '
             f'ORDER BY "Start"')
    else:
        raise RuntimeError(f"Unknown column layout in {nvtx_table}: {cols}")

    cur.execute(q)
    rows = cur.fetchall()
    conn.close()
    return [(str(lbl), int(start), int(end)) for lbl, start, end in rows]


def filter_gate_ranges(raw_ranges, gate_label, verbose=True):
    """Keep only NVTX ranges temporally enclosed by a specific gate."""
    gate_low = gate_label.lower()
    intervals = [
        (start, end)
        for label, start, end in raw_ranges
        if label.strip().lower() == gate_low
    ]

    if not intervals:
        if verbose:
            print(f"  WARNING: {gate_label} not found in trace; returning no gated ranges.")
        return []

    gated_ranges = []
    for label, start, end in raw_ranges:
        for gate_start, gate_end in intervals:
            if start >= gate_start and end <= gate_end:
                gated_ranges.append((label, (end - start) / 1000.0))
                break

    if verbose:
        total = len(raw_ranges)
        print(f"  Gate filter {gate_label}: kept {len(gated_ranges)} ranges "
              f"inside {len(intervals)} gate interval(s) from {total} total")

    return gated_ranges


def _extract_cli(rep_path: str, prefix: str):
    csv_base = f"{prefix}_nvtx_raw"
    subprocess.run(
        ["nsys", "stats", rep_path,
         "--report", "nvtx_pushpop_trace",
         "--format", "csv", "-o", csv_base],
        check=True, capture_output=True, text=True,
    )
    actual = f"{csv_base}_nvtx_pushpop_trace.csv"
    if not os.path.exists(actual):
        candidates = list(Path(".").glob(f"{csv_base}*nvtx*.csv"))
        actual = str(candidates[0]) if candidates else actual

    results = []
    with open(actual) as f:
        for row in csv.DictReader(f):
            lbl = row.get("Name") or row.get("Range") or row.get("name") or ""
            dur_str = (row.get("Duration (ns)") or row.get("Duration(ns)")
                       or row.get("duration") or "0")
            try:
                dur_ns = float(dur_str.replace(",", ""))
            except ValueError:
                continue
            results.append((lbl, dur_ns / 1000.0))
    return results


def extract_nvtx(rep_path: str, prefix: str, gate_label: str):
    """Extract NVTX ranges for a specific timed gate."""
    try:
        raw = _extract_sqlite(rep_path)
        if raw:
            return filter_gate_ranges(raw, gate_label)
    except Exception as e:
        print(f"  SQLite extraction failed ({e}), trying nsys CLI ...")
    # CLI fallback: no timestamps — cannot reliably attribute nested ranges to
    # separate gates.  Keep legacy decode-only fallback and skip prefill.
    if gate_label != DECODE_GATE:
        print(f"  CLI fallback cannot split {gate_label}; returning no gated ranges.")
        return []
    ranges = _extract_cli(rep_path, prefix)
    before = len(ranges)
    ranges = [(lbl, dur) for lbl, dur in ranges
              if lbl.strip().lower() not in WARMUP_ONLY_LABELS]
    print(f"  CLI fallback: excluded {before - len(ranges)} warmup-only "
          f"ranges (label-based, no temporal filter)")
    return ranges


# ════════════════════════════════════════════════════════════════════════════
#  Aggregate
# ════════════════════════════════════════════════════════════════════════════

def aggregate(ranges, gate_label=DECODE_GATE):
    """Return {component: total_us} and {component: call_count}.

    "Others" is computed as the gap between the timed gate total and the sum of
    attributed components (excluding off-stream H2D Total).
    This captures inter-range Python overhead, CUDA event recording cost,
    and any other unattributed work.
    """
    comp_us = defaultdict(float)
    comp_count = defaultdict(int)
    gate_total_us = 0.0
    gate_count = 0
    # Kernel-level nested labels should contribute time, but not inflate call counts.
    no_count_substrings = (
        "kernel: max pooling",
        # prep diff_offload contributes Offloading (L1+) time but should not add
        # a second per-layer call on top of offloading_l1p.
        "kv_prep",
    )
    gate_low = gate_label.lower()
    for label, dur_us in ranges:
        low = label.strip().lower()
        # Accumulate gate (decode wrapper) total for computing Others
        if low == gate_low:
            gate_total_us += dur_us
            gate_count += 1
            continue
        if low.startswith(DETAIL_LABEL_PREFIX):
            continue
        if low in SKIP_LABELS:
            continue
        comp = classify_label(label)
        comp_us[comp] += dur_us
        if not any(s in low for s in no_count_substrings):
            comp_count[comp] += 1
    # Derive "Others" from the gap between gate total and attributed components.
    # H2D Total is on the prefetch stream (not the main-stream critical path),
    # so exclude it from the attributed sum.
    if gate_total_us > 0:
        attributed = sum(v for k, v in comp_us.items() if k != "H2D Total")
        others_us = gate_total_us - attributed
        if others_us > 0:
            comp_us["Others"] = others_us
            # Others is distributed across all decode layer calls, so use the
            # same call count as the majority of components (e.g. 128).
            comp_count["Others"] = max(comp_count.values()) if comp_count else gate_count
    return dict(comp_us), dict(comp_count)


def aggregate_other_details(ranges, total_others_us):
    """Aggregate verbose-only NVTX labels that explain the Others bucket."""
    detail_us = defaultdict(float)
    detail_count = defaultdict(int)
    for label, dur_us in ranges:
        low = label.strip().lower()
        if not low.startswith(DETAIL_LABEL_PREFIX):
            continue
        detail_name = label.strip()[len(DETAIL_LABEL_PREFIX):].strip()
        if not detail_name:
            continue
        detail_us[detail_name] += dur_us
        detail_count[detail_name] += 1

    if total_others_us <= 0 and not detail_us:
        return {}, {}

    covered_us = sum(detail_us.values())
    residual_us = max(0.0, total_others_us - covered_us)
    if residual_us > 0:
        detail_us["Residual unlabeled"] = residual_us
        detail_count["Residual unlabeled"] = max(detail_count.values()) if detail_count else 0
    return dict(detail_us), dict(detail_count)


# ════════════════════════════════════════════════════════════════════════════
#  Print / save summary
# ════════════════════════════════════════════════════════════════════════════

def print_summary(comp_us, comp_count, prefix, scale=1.0, save_csv=False, offload=False,
                  ig_h2d_breakdown=None):
    """Print wall-time breakdown table.

    Args:
        scale: profiled_thru / baseline_thru  (< 1 when profiling adds
               overhead).  Est. column = profiled_ms * scale, applied
               uniformly to all components.
        save_csv: if True, write a CSV file alongside the printed table.

    H2D overlap handling (prefetch-capable paths, e.g. decoupled/InfiniGen):
        "H2D Total" runs on the prefetch stream — it overlaps with main-stream
        work and should NOT be counted in the critical-path TOTAL.
        "H2D Exposed" is the wait stall on the main stream — this IS on the
        critical path.  After the main table we print a separate overlap
        summary: H2D Hidden = Total - Exposed.
    """
    # Separate prefetch-stream component ("H2D Total") from main-stream ones.
    # "H2D Total" is NOT on the critical path so it must be excluded from TOTAL.
    H2D_TOTAL_KEY = "H2D Total"
    H2D_EXPOSED_KEY = "H2D Exposed"

    h2d_total_us = comp_us.get(H2D_TOTAL_KEY, 0)
    h2d_total_n = comp_count.get(H2D_TOTAL_KEY, 0)
    # Keep raw exposed time for the overlap section, even if we merge it into Offloading below.
    h2d_exposed_us = comp_us.get(H2D_EXPOSED_KEY, 0)

    # InfiniGen: "Others" is dominated by GPU-side stalls at wait_event
    # waiting for the prefetch stream (CPU gather + H2D).  With all layers
    # pre-enqueued, the stall bleeds into gaps between layer forwards
    # rather than appearing inside the "h2d_wait" NVTX range.  Estimate
    # the H2D-wait portion as max(0, h2d_total - labeled_compute) and
    # move it from "Others" to "H2D Exposed".
    has_ig_bd = (ig_h2d_breakdown is not None
                 and ig_h2d_breakdown.get("calls", 0) > 0)
    others_us = comp_us.get("Others", 0)
    if has_ig_bd and others_us > 0:
        ig_total_us = sum(ig_h2d_breakdown[k] for k in
                          ("mask_index_us", "cpu_gather_us", "h2d_us", "scatter_us"))
        labeled_compute_us = sum(
            v for k, v in comp_us.items()
            if k not in (H2D_TOTAL_KEY, H2D_EXPOSED_KEY, "Others"))
        h2d_wait_us = min(others_us, max(0, ig_total_us - labeled_compute_us))
        if h2d_wait_us > 0:
            h2d_exposed_us += h2d_wait_us
            comp_us["H2D Exposed"] = h2d_exposed_us
            comp_us["Others"] = others_us - h2d_wait_us

    # Critical-path total excludes H2D Total (it's on a different stream)
    crit_us = {k: v for k, v in comp_us.items() if k != H2D_TOTAL_KEY}
    crit_count = {k: v for k, v in comp_count.items() if k != H2D_TOTAL_KEY}
    # Merge exposed wait into Offloading (L1+ preferred, else Offloading) in the main table.
    if h2d_exposed_us > 0:
        merge_target = "Offloading (L1+)" if crit_us.get("Offloading (L1+)", 0) > 0 else "Offloading"
        crit_us[merge_target] = crit_us.get(merge_target, 0) + h2d_exposed_us
        # Keep call count unchanged for the merge target.
        crit_us[H2D_EXPOSED_KEY] = 0
        crit_count[H2D_EXPOSED_KEY] = 0
    total = sum(crit_us.values())
    if total == 0:
        print("WARNING: no NVTX ranges found.")
        return

    show_est = (scale < 0.99)

    print()
    hdr_width = 108 if show_est else 80
    print(f"{'=' * hdr_width}")
    print(f"  Wall-Time Breakdown: {prefix}")
    if show_est:
        overhead_pct = (1.0 / scale - 1.0) * 100
        print(f"  Profiling overhead ~{overhead_pct:.0f}%  "
              f"(scale = profiled_thru / baseline_thru = {scale:.3f})")
        if overhead_pct > 50:
            print(f"  NOTE: overhead >{50}% — consider --test-n 4+ for a more stable scale factor")
    print(f"{'=' * hdr_width}")
    hdr = f"  {'Component':<20} {'Calls':>7} {'Total (ms)':>12} {'Avg (us)':>10} {'Fraction':>10}"
    if show_est:
        hdr += f" {'Est. (ms)':>11} {'Est.Avg(us)':>12}"
    print(hdr)
    sep = f"  {'-' * 20} {'-' * 7} {'-' * 12} {'-' * 10} {'-' * 10}"
    if show_est:
        sep += f" {'-' * 11} {'-' * 12}"
    print(sep)

    rows = []
    for comp in COMPONENT_ORDER:
        if comp == H2D_TOTAL_KEY:
            continue  # printed separately below
        us = crit_us.get(comp, 0)
        if us == 0:
            continue
        n = crit_count.get(comp, 0)
        ms = us / 1000.0
        avg_us = us / n if n > 0 else 0
        pct = us / total * 100
        est_ms = ms * scale
        est_avg_us = avg_us * scale
        line = f"  {comp:<20} {n:>7} {ms:>12.2f} {avg_us:>10.1f} {pct:>9.1f}%"
        if show_est:
            line += f" {est_ms:>11.2f} {est_avg_us:>12.1f}"
        print(line)
        rows.append((comp, n, ms, avg_us, pct, est_ms, est_avg_us))

    total_ms = total / 1000.0
    total_n = sum(crit_count.values())
    # TOTAL Avg(us): sum of per-component averages over displayed components.
    # Split components (e.g. Offloading L0/L1+) are grouped into one combined
    # average before adding to the total.
    SPLIT_GROUPS = [
        {"Offloading (L0)", "Offloading (L1+)", "Offloading"},
    ]
    sum_avg_us = 0.0
    visited = set()
    for comp in COMPONENT_ORDER:
        if comp in visited:
            continue
        us = crit_us.get(comp, 0)
        if us == 0:
            visited.add(comp)
            continue
        # Check if this comp belongs to a split group
        group = None
        for g in SPLIT_GROUPS:
            if comp in g:
                group = g
                break
        if group:
            group_us = sum(crit_us.get(c, 0) for c in group)
            group_n = sum(crit_count.get(c, 0) for c in group)
            sum_avg_us += (group_us / group_n) if group_n > 0 else 0.0
            visited.update(group)
        else:
            n = crit_count.get(comp, 0)
            sum_avg_us += (us / n) if n > 0 else 0.0
            visited.add(comp)
    est_total_ms = total_ms * scale
    est_sum_avg_us = sum_avg_us * scale
    print(sep)
    line = f"  {'TOTAL':<20} {total_n:>7} {total_ms:>12.2f} {sum_avg_us:>10.1f} {'100.0%':>10}"
    if show_est:
        line += f" {est_total_ms:>11.2f} {est_sum_avg_us:>12.1f}"
    print(line)
    print(f"{'=' * hdr_width}")

    # ── H2D overlap summary ────────────────────────────────────────────
    # If no h2d_prefetch NVTX ranges (BG thread doesn't emit them to avoid
    # stack corruption) but breakdown stats are available, use breakdown as
    # the H2D total.  Must run BEFORE computing has_h2d_prefetch.
    has_ig_bd = (ig_h2d_breakdown is not None
                 and ig_h2d_breakdown.get("calls", 0) > 0)
    if h2d_total_us == 0 and has_ig_bd:
        _bd = ig_h2d_breakdown
        h2d_total_us = sum(_bd[k] for k in ("mask_index_us", "cpu_gather_us",
                                              "h2d_us", "scatter_us"))
        h2d_total_n = _bd["calls"]
    h2d_hidden_us = max(0, h2d_total_us - h2d_exposed_us)
    has_h2d_prefetch = (h2d_total_us > 0 or h2d_exposed_us > 0)

    if has_h2d_prefetch:
        # Prefetch-capable paths: H2D runs on a side stream and can overlap.
        h2d_total_ms_ = h2d_total_us / 1000.0
        h2d_exposed_ms_ = h2d_exposed_us / 1000.0
        h2d_hidden_ms_ = h2d_hidden_us / 1000.0
        overlap_pct = (h2d_hidden_us / h2d_total_us * 100) if h2d_total_us > 0 else 0

        print(f"\n  H2D Prefetch Overlap:")
        if h2d_total_n > 0:
            print(f"    Total H2D (prefetch stream):  {h2d_total_ms_:>8.2f} ms"
                  f"  ({h2d_total_n} calls, avg {h2d_total_us / h2d_total_n:.1f} us)")
        print(f"    Exposed  (main stream wait):  {h2d_exposed_ms_:>8.2f} ms")
        print(f"    Hidden   (overlapped):        {h2d_hidden_ms_:>8.2f} ms"
              f"  ({overlap_pct:.0f}% overlap)")
        if show_est:
            print(f"    Est. Total:  {h2d_total_ms_ * scale:.2f} ms"
                  f"    Est. Exposed:  {h2d_exposed_ms_ * scale:.2f} ms"
                  f"    Est. Hidden:  {h2d_hidden_ms_ * scale:.2f} ms")
        # Phase breakdown from InfiniGen gather worker (CPU-side timing)
        if has_ig_bd:
            _bd = ig_h2d_breakdown
            _n = _bd["calls"]
            _mi = _bd["mask_index_us"] / 1000.0   # ms
            _cg = _bd["cpu_gather_us"] / 1000.0
            _h2 = _bd["h2d_us"] / 1000.0
            _sc = _bd["scatter_us"] / 1000.0
            _tot = _mi + _cg + _h2 + _sc
            print(f"\n    H2D Phase Breakdown ({_n} calls, stream-sync overhead):")
            print(f"      Mask + Index:      {_mi:>10.2f} ms  (avg {_mi/_n*1000:>7.1f} us)")
            print(f"      CPU Gather:        {_cg:>10.2f} ms  (avg {_cg/_n*1000:>7.1f} us)")
            print(f"      H2D Transfer:      {_h2:>10.2f} ms  (avg {_h2/_n*1000:>7.1f} us)")
            print(f"      GPU Scatter:       {_sc:>10.2f} ms  (avg {_sc/_n*1000:>7.1f} us)")
            if _tot > 0:
                print(f"      ────")
                print(f"      Mask+Idx {_mi/_tot*100:>5.1f}%  │  "
                      f"Gather {_cg/_tot*100:>5.1f}%  │  "
                      f"H2D {_h2/_tot*100:>5.1f}%  │  "
                      f"Scatter {_sc/_tot*100:>5.1f}%")
    elif offload:
        # Non-decoupled + offload: H2D is inline on main stream (no overlap)
        offload_us = crit_us.get("Offloading", 0)
        offload_n = crit_count.get("Offloading", 0)
        if offload_us > 0:
            offload_ms = offload_us / 1000.0
            print(f"\n  H2D Inline (no overlap — main stream):")
            print(f"    Total H2D (main stream):      {offload_ms:>8.2f} ms"
                  f"  ({offload_n} calls, avg {offload_us / offload_n:.1f} us)")
            print(f"    Overlap:                       0%  (inline, no prefetch stream)")
            if show_est:
                print(f"    Est. Total:  {offload_ms * scale:.2f} ms")
    print()

    # ── CSV (optional) ────────────────────────────────────────────────
    if save_csv:
        csv_path = f"{prefix}_nvtx_breakdown.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["Component", "Calls", "Total_ms", "Avg_us", "Fraction_pct"]
            if show_est:
                header += ["Est_ms", "Est_Avg_us"]
            w.writerow(header)
            for r in rows:
                w.writerow(r if show_est else r[:5])
            total_row = ["TOTAL", total_n, total_ms, sum_avg_us, 100.0]
            if show_est:
                total_row += [est_total_ms, est_sum_avg_us]
            w.writerow(total_row)
            if has_h2d_prefetch:
                h2d_exposed_n = comp_count.get(H2D_EXPOSED_KEY, 0)
                h2d_rows = [
                    ["H2D Total (prefetch)", h2d_total_n, h2d_total_us / 1000.0,
                     h2d_total_us / h2d_total_n if h2d_total_n > 0 else 0, ""],
                    ["H2D Exposed (wait)", h2d_exposed_n,
                     h2d_exposed_us / 1000.0,
                     h2d_exposed_us / h2d_exposed_n if h2d_exposed_n > 0 else 0, ""],
                    ["H2D Hidden (overlap)", "", h2d_hidden_us / 1000.0, "", ""],
                ]
                for hr in h2d_rows:
                    w.writerow(hr)
        print(f"  CSV  -> {csv_path}")


def print_other_details(detail_us, detail_count, scale=1.0):
    """Print a verbose sub-breakdown for the Others residual."""
    if not detail_us:
        return

    total_us = sum(detail_us.values())
    print("\n  Others Detail (verbose):")
    print("    Label                  Calls   Total (ms)   Avg (us)   Est. (ms)")
    print("    -------------------- ------- ------------ ---------- -----------")
    total_calls = 0
    for label, us in sorted(detail_us.items(), key=lambda kv: kv[1], reverse=True):
        calls = detail_count.get(label, 0)
        total_calls += calls
        avg_us = us / calls if calls > 0 else 0.0
        est_ms = us * scale / 1000.0
        print(f"    {label[:20]:20} {calls:7d} {us/1000.0:12.2f} {avg_us:10.1f} {est_ms:11.2f}")
    print("    -------------------- ------- ------------ ---------- -----------")
    print(f"    {'TOTAL':20} {total_calls:7d} {total_us/1000.0:12.2f} {'':10} {total_us*scale/1000.0:11.2f}")


# ════════════════════════════════════════════════════════════════════════════
#  PCIe bandwidth detection
# ════════════════════════════════════════════════════════════════════════════

# Theoretical unidirectional bandwidth per lane (GB/s), 128b/130b encoding.
_PCIE_GEN_BW_PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def _detect_pcie_peak_gbs(device_index: int = 0) -> tuple:
    """Detect PCIe generation and width, return (peak_gbs, description).

    Tries nvidia-smi first, then falls back to reading sysfs for the GPU's
    PCIe device.  Returns (peak_gbs, desc_string).  On failure, falls back
    to Gen4 x16 = 32 GB/s with a warning printed to stdout.
    """
    gen, width = None, None

    # --- Method 1: nvidia-smi (works on most systems with the driver) ------
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ["nvidia-smi",
             f"--id={device_index}",
             "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=_sp.DEVNULL,
        ).decode().strip()
        parts = out.split(",")
        if len(parts) == 2:
            gen, width = int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass

    # --- Method 2: sysfs via torch (needs GPU to resolve BDF) --------------
    if gen is None or width is None:
        try:
            import torch as _torch
            bdf = None
            props = _torch.cuda.get_device_properties(device_index)
            # PyTorch >= 2.1 exposes pci_bus_id
            if hasattr(props, "pci_bus_id"):
                bdf = props.pci_bus_id
            if bdf is None:
                # Try CUDA runtime query
                import subprocess as _sp
                out = _sp.check_output(
                    ["nvidia-smi",
                     f"--id={device_index}",
                     "--query-gpu=pci.bus_id",
                     "--format=csv,noheader"],
                    timeout=5, stderr=_sp.DEVNULL,
                ).decode().strip()
                if out:
                    bdf = out

            if bdf is not None:
                # Normalise BDF: "00000000:04:00.0" → find matching sysfs dir
                bdf_short = bdf.lower().replace("gpu-", "")
                import glob as _glob
                candidates = _glob.glob(f"/sys/bus/pci/devices/*{bdf_short[-7:]}") or \
                             _glob.glob(f"/sys/bus/pci/devices/*{bdf_short}")
                if candidates:
                    sysfs = candidates[0]
                    with open(f"{sysfs}/current_link_speed") as f:
                        speed_str = f.read().strip()  # e.g. "16.0 GT/s PCIe"
                    with open(f"{sysfs}/current_link_width") as f:
                        width = int(f.read().strip())
                    gt_s = float(speed_str.split()[0])
                    # Map GT/s → generation
                    _gts_to_gen = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
                    gen = _gts_to_gen.get(gt_s)
        except Exception:
            pass

    # --- Fallback -----------------------------------------------------------
    if gen is None or width is None:
        print("  WARNING: could not detect PCIe generation/width; "
              "assuming Gen4 x16 (32 GB/s)")
        return 32.0, "Gen4 x16 (assumed)"

    bw_per_lane = _PCIE_GEN_BW_PER_LANE.get(gen)
    if bw_per_lane is None:
        print(f"  WARNING: unknown PCIe Gen{gen}; assuming Gen4 x16 (32 GB/s)")
        return 32.0, "Gen4 x16 (assumed)"

    peak = bw_per_lane * width
    desc = f"Gen{gen} x{width}"
    return peak, desc


# ════════════════════════════════════════════════════════════════════════════
#  H2D traffic stats
# ════════════════════════════════════════════════════════════════════════════

def _print_h2d_stats(h2d, comp_us, comp_count, scale, prefetch_h2d, test_n,
                     ig_h2d_breakdown=None):
    """Print block hit-rate (aggregate + per-layer), H2D traffic, PCIe throughput.

    Parameters
    ----------
    scale : float
        profiled_thru / baseline_thru — applied to the H2D wall time to
        remove profiling overhead before computing PCIe throughput.
    prefetch_h2d : bool
        Hint that this run uses a prefetch stream. Actual NVTX labels still take
        precedence so the breakdown stays consistent with the measured trace.
    test_n : int
        Number of profiled iterations.  Both comp_us and h2d stats are summed
        across all test_n iterations.  Ratios (hit_rate, bytes_per_step,
        PCIe throughput) are correct as-is because both numerator and
        denominator scale with test_n.  Absolute display values (total bytes,
        decode steps) are divided by test_n to show per-iteration averages.
    """
    pcie_peak_gbs, pcie_desc = _detect_pcie_peak_gbs()
    is_hf = h2d.get("hf_backend", False)  # True for HF backend (no actual H2D)

    miss = h2d["miss_blocks"]
    total = h2d["total_blocks"]
    hit_rate = h2d["hit_rate"]
    h2d_bytes = h2d["h2d_bytes"]
    num_layers = h2d.get("num_layers", 1)
    per_layer = h2d.get("per_layer", [])

    # Derive decode_steps from the busiest per-layer H2D path.
    # Layer 0 can be fully resident, so corr_calls on layer 0 may be zero even
    # when later layers perform one prefetch per decode step.
    decode_steps = max(
        (
            max(int(s.get("corr_calls", 0)), int(s.get("pf_calls", 0)))
            for s in per_layer
        ),
        default=0,
    )
    if decode_steps <= 0:
        decode_steps = 1
    bytes_per_step = h2d_bytes / decode_steps if decode_steps > 0 else h2d_bytes

    # Per-iteration display values (accumulated / test_n)
    h2d_bytes_per_iter = h2d_bytes / max(test_n, 1)
    decode_steps_per_iter = decode_steps / max(test_n, 1)

    has_h2d_prefetch = bool(
        prefetch_h2d
        or comp_us.get("H2D Total", 0) > 0
        or comp_us.get("H2D Exposed", 0) > 0
    )

    # ── PCIe throughput estimation (nosi only — HF has no actual H2D) ──
    pcie_gbs = 0
    h2d_time_label = ""
    if not is_hf:
        # Both comp_us and h2d_bytes are summed across test_n iterations.
        # The test_n factors cancel in the ratio, giving correct PCIe.
        if has_h2d_prefetch:
            h2d_time_us = comp_us.get("H2D Total", 0)
            h2d_time_label = "H2D Total (NVTX)"
            # InfiniGen path: no NVTX h2d_prefetch label (H2D runs on BG
            # thread).  Use the pageable→pinned copy time ("h2d_us") from
            # the gather breakdown as the best CPU-side proxy for transfer
            # bandwidth.  If that's also zero (e.g. block-oriented gather
            # writes directly to pinned), fall back to exposed wait time.
            if h2d_time_us == 0 and ig_h2d_breakdown is not None:
                h2d_time_us = ig_h2d_breakdown.get("h2d_us", 0)
                h2d_time_label = "H2D copy (gather breakdown)"
                if h2d_time_us == 0:
                    # No separate copy timing — use H2D Exposed as proxy
                    h2d_time_us = h2d_exposed_us
                    h2d_time_label = "H2D Exposed (inferred)"
        else:
            h2d_time_us = comp_us.get("Offloading", 0)
            h2d_time_label = "Offloading"
        effective_time_us = h2d_time_us * scale
        if effective_time_us > 0:
            pcie_gbs = h2d_bytes / (effective_time_us / 1e6) / 1e9

    # ── Aggregate summary ───────────────────────────────────────────
    backend_tag = "HF" if is_hf else "nosi"
    iter_label = f"avg over {test_n} iterations" if test_n > 1 else "1 iteration"
    print()
    print(f"{'=' * 80}")
    if is_hf:
        print(f"  Block Hit-Rate ({backend_tag} backend, {iter_label})")
    else:
        print(f"  H2D Traffic & Block Hit-Rate ({backend_tag} backend, {iter_label})")
    print(f"{'=' * 80}")
    print(f"  Block hit-rate:       {hit_rate:.1%}"
          f"  ({total - miss:,} hits / {total:,} total across {num_layers} layers"
          + (f" x {test_n} iters)" if test_n > 1 else ")"))
    if not is_hf:
        true_miss = h2d.get("true_miss", 0)
        wasted = h2d.get("wasted_reload", 0)
        if true_miss > 0:
            print(f"  True miss blocks:     {true_miss:,}  (theoretical min, final_topk not in old_block_map)")
            print(f"  Wasted reloads:       {wasted:,}  (pf_miss + corr_miss - true_miss)")
        print(f"  H2D bytes/iter:       {h2d_bytes_per_iter / 1e6:,.2f} MB"
              f"  ({decode_steps_per_iter:.0f} decode steps x {num_layers} layers)")
        print(f"  H2D bytes/step:       {bytes_per_step / 1e3:,.1f} KB")
        if pcie_gbs > 0:
            scaling_note = f"{h2d_time_label} x scale={scale:.3f}" if scale < 0.99 else h2d_time_label
            print(f"  Est. PCIe throughput: {pcie_gbs:.2f} GB/s  ({scaling_note})")
            print(f"  PCIe link:           {pcie_desc}  (peak {pcie_peak_gbs:.1f} GB/s,"
                  f" utilization {pcie_gbs / pcie_peak_gbs * 100:.0f}%)")
            if pcie_gbs > pcie_peak_gbs:
                print(f"  WARNING: exceeds {pcie_desc} peak"
                      f" — scale factor likely unreliable, try --test-n 4+")

    # ── Per-layer hit-rate table ────────────────────────────────────
    if per_layer:
        print()
        hdr = f"  {'Layer':>5}  {'Hit-Rate':>10}  {'Miss/Step':>10}"
        print(hdr)
        print(f"  {'-' * 5}  {'-' * 10}  {'-' * 10}")
        for i, s in enumerate(per_layer):
            total_i = s.get("total_blocks", 0)
            miss_i = s.get("miss_blocks", s.get("pf_miss", 0) + s.get("corr_miss", 0))
            hr = f"{1.0 - miss_i / total_i:.1%}" if total_i > 0 else "N/A"
            miss_ps = miss_i / decode_steps
            print(f"  {i:>5}  {hr:>10}  {miss_ps:>10.1f}")

        # Summary: min/mean/max hit-rate
        hit_rates = [s["hit_rate"] for s in per_layer if s["total_blocks"] > 0]
        if hit_rates:
            print(f"\n  Per-layer hit-rate: "
                  f"min={min(hit_rates):.1%}  "
                  f"mean={sum(hit_rates)/len(hit_rates):.1%}  "
                  f"max={max(hit_rates):.1%}")

    print(f"{'=' * 80}\n")


# ════════════════════════════════════════════════════════════════════════════
#  Plot
# ════════════════════════════════════════════════════════════════════════════

def plot_breakdown(comp_us, prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping plot.")
        return

    # Keep plot consistent with table: hide internal H2D-total row and
    # merge exposed wait into Offloading (L1+ preferred, else Offloading).
    comp_plot_us = {k: v for k, v in comp_us.items() if k != "H2D Total"}
    h2d_exposed_us = comp_plot_us.get("H2D Exposed", 0)
    if h2d_exposed_us > 0:
        merge_target = "Offloading (L1+)" if comp_plot_us.get("Offloading (L1+)", 0) > 0 else "Offloading"
        comp_plot_us[merge_target] = comp_plot_us.get(merge_target, 0) + h2d_exposed_us
        comp_plot_us["H2D Exposed"] = 0

    total = sum(comp_plot_us.get(c, 0) for c in COMPONENT_ORDER)
    if total == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 2.0))
    left = 0
    handles = []

    for comp in COMPONENT_ORDER:
        us = comp_plot_us.get(comp, 0)
        if us == 0:
            continue
        frac = us / total
        color = COMPONENT_COLORS[comp]
        ax.barh(0, frac, left=left, height=0.6,
                color=color, edgecolor="white", linewidth=0.5)
        if frac > 0.04:
            ms = us / 1000.0
            ax.text(left + frac / 2, 0,
                    f"{comp}\n{ms:.1f}ms\n({frac * 100:.1f}%)",
                    ha="center", va="center", fontsize=7, fontweight="bold")
        left += frac
        handles.append(mpatches.Patch(color=color, label=comp))

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of Decode Time")
    ax.set_title(f"Wall-Time Breakdown: {prefix}", fontsize=11, fontweight="bold")
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.35), ncol=len(handles),
              fontsize=7, frameon=False)

    fig.tight_layout()
    png_path = f"{prefix}_breakdown.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {png_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Env-var summary
# ════════════════════════════════════════════════════════════════════════════

_ENV_KEYS = [
    "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC",
    "OMP_SCHEDULE", "OMP_WAIT_POLICY", "KMP_AFFINITY",
    "TORCH_CUDA_ARCH_LIST", "CUDA_VISIBLE_DEVICES",
]


def _print_env():
    """Print performance-relevant environment variables."""
    print("    Env:")
    for k in _ENV_KEYS:
        v = os.environ.get(k)
        if v is not None:
            print(f"      {k}={v}")
        else:
            print(f"      {k}  (not set)")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Profile bench.py and produce a wall-time breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode selection ────────────────────────────────────────────────
    parser.add_argument(
        "--nsys", action="store_true",
        help="Use Nsight Systems instead of CUDA events (slower, +30%% overhead, "
             "but gives full kernel-level detail).",
    )
    parser.add_argument(
        "--parse", default=None, metavar="FILE.nsys-rep",
        help="Parse an existing .nsys-rep file directly (no profiling).",
    )

    # ── Bench parameters (same as bench.py) ──────────────────────────
    parser.add_argument(
        "--model-path", default=None,
        help="HuggingFace model path (e.g. openbmb/NOSA-8B, openbmb/MiniCPM4.1-8B).",
    )
    parser.add_argument(
        "--backend", default="nosi",
        choices=["nosi", "hf"],
        help="Inference backend (default: nosi). Use 'hf' to profile via HuggingFace.",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Force dense (full) attention on the selected backend.",
    )
    parser.add_argument(
        "--infinigen", action="store_true",
        help="Enable InfiniGen decode for MiniCPM or NOSA models.",
    )
    parser.add_argument(
        "--no-offload", action="store_true",
        help="Disable KV cache offloading.",
    )
    parser.add_argument(
        "--offload-layer0", action="store_true",
        help="With NOSI offload enabled, also store layer-0 KV cache on CPU instead of "
             "keeping it resident on GPU.",
    )
    parser.add_argument(
        "--batch-size", "-B", type=int, default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--seq-len", "-L", type=str, default="128K",
        help="Input sequence length (default: 128K).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=4,
        help="Tokens to generate per sample (default: 4).",
    )
    parser.add_argument(
        "--test-n", type=int, default=4,
        help="Number of timed iterations (default: 4).",
    )
    parser.add_argument(
        "--dataset", dest="dataset_name", type=str, default="emozilla/pg19",
        help="Input source: HuggingFace dataset (default: emozilla/pg19) "
             "or 'ruler' / 'ruler_<task>' for exact RULER prompt text "
             "(default task: niah_single_3).",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="test",
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--sparda", action="store_true",
        help="Enable SparDA sparse attention. Use with --indexer-path.",
    )
    parser.add_argument(
        "--sparda-no-prefetch", action="store_true",
        help="Disable the persistent decode prefetch path for SpARDA on the nosi backend. "
             "Implies --sparda and uses the normal decode_update_kv fetch pipeline instead.",
    )
    parser.add_argument(
        "--indexer-path", type=str, default=None,
        help="Path to SparDA indexer weights checkpoint for --sparda.",
    )
    parser.add_argument(
        "--long-context", action="store_true", default=False,
        help="Enable 128K long-context extension. "
             "For MiniCPM: applies official 128K LongRoPE factors. "
             "For NOSA: increases rope_theta to 40000 (default).",
    )
    parser.add_argument(
        "--yarn", action="store_true", default=False,
        help="Use YaRN (factor=4.0) instead of rope_theta increase for NOSA long-context. "
             "Only effective with --long-context on NOSA models.",
    )

    # ── Profile control ──────────────────────────────────────────────
    parser.add_argument(
        "--skip-profile", action="store_true",
        help="(--nsys only) Skip nsys profiling; reuse existing .nsys-rep.",
    )
    parser.add_argument(
        "--output-prefix", default=None,
        help="Custom prefix for output files (default: auto-generated).",
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Write a per-component CSV file (off by default).",
    )
    parser.add_argument(
        "--save-png", action="store_true",
        help="Write a stacked-bar PNG chart (off by default).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Emit extra NVTX labels for normally-unattributed decode regions and "
             "print an Others-detail table. Default summary stays unchanged.",
    )

    args = parser.parse_args()

    if args.sparda_no_prefetch:
        args.sparda = True

    # Auto-enable long-context when seq_len exceeds native max
    if args.model_path is not None:
        method = detect_method(args.model_path)
        native_max = NATIVE_MAX.get(method, 0)
        seq_len = parse_seq_len(args.seq_len) if isinstance(args.seq_len, str) else args.seq_len
        if not getattr(args, 'long_context', False) and native_max and seq_len > native_max:
            args.long_context = True
            print(
                f"[Auto] --long-context enabled: seq_len {seq_len} > "
                f"native max {native_max} for {method}"
            )
        if args.dense and args.sparda:
            parser.error("--dense and --sparda are mutually exclusive.")
        if args.sparda_no_prefetch and args.backend != "nosi":
            parser.error("--sparda-no-prefetch is only supported with --backend nosi.")
        # InfiniGen supports both MiniCPM and NOSA models
        if args.infinigen and args.dense:
            parser.error("--dense and --infinigen are mutually exclusive.")
        # InfiniGen works with both HF and NOSI backends

    # ── Parse-only mode ──────────────────────────────────────────────
    if args.parse:
        rep_file = args.parse
        if not os.path.exists(rep_file):
            print(f"ERROR: {rep_file} not found.")
            sys.exit(1)
        prefix = Path(rep_file).stem
        print(f"=== Parsing existing report: {rep_file}")
        print(f"\n=== Extracting NVTX ranges (prefill/decode) ===")
        prefill_ranges = extract_nvtx(rep_file, prefix, PREFILL_GATE)
        decode_ranges = extract_nvtx(rep_file, prefix, DECODE_GATE)
        print(f"  Using {len(prefill_ranges)} prefill-path ranges.")
        print(f"  Using {len(decode_ranges)} decode-path ranges.")
        prefill_scale = 1.0
        decode_scale = 1.0
        h2d_stats = None
        ig_h2d_breakdown = None

    elif args.nsys:
        # ── Nsight Systems mode ──────────────────────────────────────
        if args.model_path is None:
            parser.error("--model-path is required")
        prefix = build_output_prefix(args)
        rep_file = f"{prefix}.nsys-rep"

        if args.skip_profile:
            if not os.path.exists(rep_file):
                print(f"ERROR: {rep_file} not found. Run without --skip-profile first.")
                sys.exit(1)
            print(f"=== Reusing existing report: {rep_file}")
        else:
            bench_cmd = build_bench_cmd(args)
            method = detect_method(args.model_path)
            mode = "dense" if args.dense else ("infinigen" if args.infinigen else method)
            print(f"=== Profiling with Nsight Systems (expect ~30% overhead) ===")
            print(f"    Config: method={mode}, backend={args.backend}, "
                  f"B={args.batch_size}, L={args.seq_len}, "
                  f"offload={'no' if args.no_offload else 'yes'}")
            rep_file = run_nsys_profiling(bench_cmd, prefix)
            print(f"=== Profile saved to: {rep_file}")

        print(f"\n=== Extracting NVTX ranges (prefill/decode) ===")
        prefill_ranges = extract_nvtx(rep_file, prefix, PREFILL_GATE)
        decode_ranges = extract_nvtx(rep_file, prefix, DECODE_GATE)
        print(f"  Using {len(prefill_ranges)} prefill-path ranges.")
        print(f"  Using {len(decode_ranges)} decode-path ranges.")
        prefill_scale = 1.0
        decode_scale = 1.0
        h2d_stats = None  # not available in nsys mode
        ig_h2d_breakdown = None

    else:
        # ── Default: CUDA event mode (fast) ──────────────────────────
        if args.model_path is None:
            parser.error("--model-path is required (or use --parse FILE.nsys-rep)")
        prefix = build_output_prefix(args)
        method = detect_method(args.model_path)
        mode = "dense" if args.dense else ("infinigen" if args.infinigen else method)
        print(f"=== CUDA Event Profiling (low overhead) ===")
        print(f"    Config: method={mode}, backend={args.backend}, "
              f"B={args.batch_size}, L={args.seq_len}, "
              f"offload={'no' if args.no_offload else 'yes'}")
        _print_env()
        (
            prefill_ranges,
            decode_ranges,
            baseline_prefill_thru,
            profiled_prefill_thru,
            baseline_decode_thru,
            profiled_decode_thru,
            h2d_stats,
            ig_h2d_breakdown,
        ) = run_cuda_event_profile(args)
        print(f"\n=== Collected {len(prefill_ranges) + len(decode_ranges)} NVTX event ranges ===")
        print(f"  Using {len(prefill_ranges)} prefill-path ranges.")
        print(f"  Using {len(decode_ranges)} decode-path ranges.")
        prefill_scale = profiled_prefill_thru / baseline_prefill_thru if baseline_prefill_thru > 0 else 1.0
        decode_scale = profiled_decode_thru / baseline_decode_thru if baseline_decode_thru > 0 else 1.0

    # ── Aggregate ────────────────────────────────────────────────────
    print(f"\n=== Aggregating per component ===")
    prefill_comp_us, prefill_comp_count = aggregate(prefill_ranges, gate_label=PREFILL_GATE)
    decode_comp_us, decode_comp_count = aggregate(decode_ranges, gate_label=DECODE_GATE)
    prefill_other_detail_us, prefill_other_detail_count = {}, {}
    decode_other_detail_us, decode_other_detail_count = {}, {}
    if getattr(args, "verbose", False):
        prefill_other_detail_us, prefill_other_detail_count = aggregate_other_details(
            prefill_ranges, prefill_comp_us.get("Others", 0.0)
        )
        decode_other_detail_us, decode_other_detail_count = aggregate_other_details(
            decode_ranges, decode_comp_us.get("Others", 0.0)
        )

    # ── Report + Plot ────────────────────────────────────────────────
    offload = not getattr(args, 'no_offload', True)
    if prefill_ranges:
        print_summary(prefill_comp_us, prefill_comp_count, f"{prefix}_prefill", prefill_scale,
                      save_csv=args.save_csv, offload=False,
                      ig_h2d_breakdown=None)
        if prefill_comp_us.get("Offloading", 0) or prefill_comp_us.get("Offloading (L0)", 0) or prefill_comp_us.get("Offloading (L1+)", 0):
            print("  Note: Prefill Offloading is inline exposed KV-cache offload/update latency; "
                  "with offload enabled this is the GPU->CPU KV offload path.")
        if getattr(args, "verbose", False):
            print_other_details(prefill_other_detail_us, prefill_other_detail_count, prefill_scale)
    print_summary(decode_comp_us, decode_comp_count, prefix, decode_scale,
                  save_csv=args.save_csv, offload=offload,
                  ig_h2d_breakdown=ig_h2d_breakdown)
    if getattr(args, "verbose", False):
        print_other_details(decode_other_detail_us, decode_other_detail_count, decode_scale)

    # ── H2D traffic stats ────────────────────────────────────────────
    if h2d_stats and h2d_stats.get("total_blocks", 0) > 0:
        prefetch_h2d = getattr(args, 'sparda', False) or getattr(args, 'infinigen', False)
        test_n = getattr(args, 'test_n', 1)
        _print_h2d_stats(h2d_stats, decode_comp_us, decode_comp_count, decode_scale, prefetch_h2d,
                         test_n, ig_h2d_breakdown=ig_h2d_breakdown)

    if args.save_png:
        plot_breakdown(decode_comp_us, prefix)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
