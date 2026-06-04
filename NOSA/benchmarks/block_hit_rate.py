# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared block hit-rate collection utilities for NOSA benchmarks.

Provides _BlockStatsTracker and _BlockHitRateCollector for measuring block
cache hit-rate during decode steps of sparse attention models.

Usage::

    from block_hit_rate import BlockHitRateCollector

    collector = BlockHitRateCollector(hf_model)
    if collector.available:
        collector.install()
        for sample in samples:
            collector.begin_sample()
            output = model.generate(inputs=sample)
            collector.end_sample()
        stats = collector.get_stats()
        collector.uninstall()
"""


class BlockStatsTracker:
    """Track block hit-rate across decode steps by comparing consecutive topk_idx.

    Compares consecutive decode steps' topk_idx to compute overlap.  A "miss" is
    a valid (non-padding) block index in the current step that was not present in
    the previous step.
    """

    def __init__(self, num_layers, block_size=64, topk=64,
                 head_dim=128, num_kv_heads=8, elem_size=2):
        self.num_layers = num_layers
        self.block_size = block_size
        self.topk = topk
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.elem_size = elem_size
        self._prev = [None] * num_layers
        self._miss = [0] * num_layers
        self._total = [0] * num_layers
        self._steps = [0] * num_layers

    def step(self, layer_idx, topk_idx):
        if topk_idx is None:
            return
        curr = topk_idx.detach()
        prev = self._prev[layer_idx]
        if prev is not None and prev.shape == curr.shape:
            in_prev = (curr.unsqueeze(-1) == prev.unsqueeze(-2)).any(-1)
            valid = curr != -1
            miss = int((valid & ~in_prev).sum().item())
            total = curr.numel()
            self._miss[layer_idx] += miss
            self._total[layer_idx] += total
            self._steps[layer_idx] += 1
        self._prev[layer_idx] = curr.clone()

    def end_step(self):
        pass

    def reset_prev(self):
        self._prev = [None] * self.num_layers

    def reset(self):
        """Reset all accumulators and prev state."""
        self._prev = [None] * self.num_layers
        self._miss = [0] * self.num_layers
        self._total = [0] * self.num_layers
        self._steps = [0] * self.num_layers

    def get_stats(self):
        per_layer = []
        agg_miss = agg_total = 0
        for i in range(self.num_layers):
            total = self._total[i]
            miss = self._miss[i]
            hr = 1.0 - miss / total if total > 0 else 0.0
            per_layer.append({
                "total_blocks": total, "miss_blocks": miss,
                "hit_rate": hr, "steps": self._steps[i],
            })
            agg_miss += miss
            agg_total += total
        agg_hr = 1.0 - agg_miss / agg_total if agg_total > 0 else 0.0
        return {
            "miss_blocks": agg_miss,
            "total_blocks": agg_total,
            "hit_rate": agg_hr,
            "num_layers": self.num_layers,
            "per_layer": per_layer,
        }


class BlockHitRateCollector:
    """Collect block hit-rate stats during HF model inference via a forward hook.

    Registers a post-forward hook on the HuggingFace model.  During ``generate()``,
    the hook fires once for prefill (skipped) and once per decode token.  At each
    decode step it reads ``_last_topk_idx`` from every sparse-attention layer and
    feeds the tracker.

    Usage::

        collector = BlockHitRateCollector(hf_model)
        if collector.available:
            collector.install()
            for sample in samples:
                collector.begin_sample()
                output = model.generate(inputs=sample)
                collector.end_sample()
            stats = collector.get_stats()
            collector.uninstall()
    """

    def __init__(self, hf_model):
        # Unwrap torch.compile: OptimizedModule overrides __call__ and skips
        # nn.Module forward hooks, but hooks on _orig_mod still fire.
        self._model = getattr(hf_model, '_orig_mod', hf_model)
        self._hook_target = self._model
        self._attn_layers = self._find_attn_layers()
        self._tracker = self._create_tracker()
        self._step_count = 0
        self._hook = None
        self._active = False

    def _find_attn_layers(self):
        inner = getattr(self._model, 'model', None)
        if inner is None or not hasattr(inner, 'layers'):
            return []
        return [layer.self_attn for layer in inner.layers
                if hasattr(layer, 'self_attn')]

    def _create_tracker(self):
        if not self._attn_layers:
            return None
        cfg = self._model.config
        sparse_cfg = getattr(cfg, 'sparse_config', None) or {}
        if not sparse_cfg:
            return None
        return BlockStatsTracker(
            num_layers=cfg.num_hidden_layers,
            block_size=int(sparse_cfg.get('block_size', 64)),
            topk=int(sparse_cfg.get('topk', 64)),
            head_dim=getattr(cfg, 'head_dim',
                             cfg.hidden_size // cfg.num_attention_heads),
            num_kv_heads=getattr(cfg, 'num_key_value_heads',
                                 cfg.num_attention_heads),
            elem_size=2,
        )

    @property
    def available(self):
        return self._tracker is not None and len(self._attn_layers) > 0

    def install(self):
        if not self.available:
            return
        self._hook = self._hook_target.register_forward_hook(self._forward_hook)

    def uninstall(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        for attn in self._attn_layers:
            attn._collect_block_stats = False
            if hasattr(attn, '_last_topk_idx'):
                attn._last_topk_idx = None

    def begin_sample(self):
        """Reset per-sample state.  Call before each ``model.generate()``."""
        self._step_count = 0
        self._active = True
        self._tracker.reset_prev()
        # Keep collection DISABLED during prefill to avoid caching the very
        # large prefill topk tensor on GPU.  The hook enables it after step 1
        # (prefill), matching profile_breakdown.py behavior.

    def end_sample(self):
        """Call after each ``model.generate()``."""
        self._active = False
        for attn in self._attn_layers:
            attn._collect_block_stats = False
            if hasattr(attn, '_last_topk_idx'):
                attn._last_topk_idx = None

    def _forward_hook(self, module, input, output):
        if not self._active:
            return
        self._step_count += 1

        if self._step_count == 1:
            # Step 1: prefill -- enable collection for decode steps that
            # follow but don't read anything yet (prefill topk has different
            # shape and would waste GPU memory).
            for attn in self._attn_layers:
                attn._collect_block_stats = True
            return

        if self._step_count == 2:
            # Step 2: first decode token -- seed _prev via tracker.step()
            # (prev is None so no measurement counted, just like
            # profile_breakdown.py's warmup decode step).
            for idx, attn in enumerate(self._attn_layers):
                topk_idx = getattr(attn, '_last_topk_idx', None)
                if topk_idx is not None:
                    self._tracker.step(idx, topk_idx)
            self._tracker.end_step()
            return

        # Step 3+: measured decode steps (decode token 2 vs 1, 3 vs 2, ...).
        for idx, attn in enumerate(self._attn_layers):
            topk_idx = getattr(attn, '_last_topk_idx', None)
            if topk_idx is not None:
                self._tracker.step(idx, topk_idx)
        self._tracker.end_step()

    def get_stats(self):
        if self._tracker is None:
            return None
        return self._tracker.get_stats()


def merge_block_stats(stats_list):
    """Merge block hit-rate stats from multiple shards/runs.

    Sums miss_blocks and total_blocks per layer across entries, then
    recomputes hit_rate.

    Args:
        stats_list: list of stats dicts (as returned by BlockHitRateCollector.get_stats())

    Returns:
        Merged stats dict, or None if no valid stats.
    """
    valid = [s for s in stats_list if s and s.get("total_blocks", 0) > 0]
    if not valid:
        return None

    num_layers = valid[0]["num_layers"]
    agg_miss = 0
    agg_total = 0
    per_layer = []
    for li in range(num_layers):
        layer_miss = sum(s["per_layer"][li]["miss_blocks"] for s in valid
                         if li < len(s.get("per_layer", [])))
        layer_total = sum(s["per_layer"][li]["total_blocks"] for s in valid
                          if li < len(s.get("per_layer", [])))
        layer_steps = sum(s["per_layer"][li]["steps"] for s in valid
                          if li < len(s.get("per_layer", [])))
        layer_hr = 1.0 - layer_miss / layer_total if layer_total > 0 else 0.0
        per_layer.append({
            "total_blocks": layer_total, "miss_blocks": layer_miss,
            "hit_rate": layer_hr, "steps": layer_steps,
        })
        agg_miss += layer_miss
        agg_total += layer_total

    agg_hr = 1.0 - agg_miss / agg_total if agg_total > 0 else 0.0
    return {
        "miss_blocks": agg_miss,
        "total_blocks": agg_total,
        "hit_rate": agg_hr,
        "num_layers": num_layers,
        "per_layer": per_layer,
    }
