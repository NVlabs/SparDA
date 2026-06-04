# This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
# Copyright (c) 2026 THUNLP.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import torch
from queue import Queue
import time
from .flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
from .flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias
from torch.utils.cpp_extension import load
from torch.cuda import nvtx
import argparse
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import CacheLayerMixin, DynamicLayer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import torch.nn.functional as F
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

_build_dir = os.path.join(this_dir, ".build")
os.makedirs(_build_dir, exist_ok=True)
diff = load(
    name="diff_offload",
    sources=[os.path.join(this_dir, "flash_cache_engine/diff_offload.cpp"), os.path.join(this_dir, "flash_cache_engine/diff_offload_kernel.cu")],
    build_directory=_build_dir,
    verbose=False,
)

decode_update_fastpath = None
decode_update_fastpath_err = None
try:
    decode_update_fastpath = load(
        name="decode_update_fastpath",
        sources=[
            os.path.join(this_dir, "flash_cache_engine/decode_update_fastpath.cpp"),
            os.path.join(this_dir, "flash_cache_engine/decode_update_fastpath_kernel.cu"),
        ],
        build_directory=_build_dir,
        verbose=False,
    )
except Exception as e:  # noqa: BLE001
    decode_update_fastpath_err = e

class CacheEngine:
    def __init__(self,
        gpu_mem_usage: int = 60,
        num_layers: int = 28,
        block_size: int = 64,
        head_num: int = 2,
        head_dim: int = 128,
        cpu_gpu_mem_size_fraction: int = 5,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
        max_batch_size: int = 1024,
        max_seq_len: int = 131072,
        topk: int = 64,
        has_kv_bias: bool = False,
        fully_resident: bool = False,
        max_gen_len: int = 8192,
    ):
        single_block_size = block_size * head_num * head_dim * (2 if dtype in [torch.float16, torch.bfloat16] else 4)
        self.block_num = gpu_mem_usage * 1024**3 // (single_block_size * num_layers * 2)
        self.block_num_cpu = self.block_num * cpu_gpu_mem_size_fraction
        self.block_size = block_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.max_batch_size = max_batch_size

        self._k_cpu = None
        self._v_cpu = None
        self._kv_cpu = None
        self._kv_cpu_flat_packed = None
        self._k_gpu = None
        self._v_gpu = None


        self.seq_length = 0

        self.total_load_all = 0
        self.total_offload_all = 0
        self.total_update_time = 0
        self.is_first_decode = True
        self.max_seq_len = max_seq_len
        self.topk = topk
        self.max_gen_len = int(max_gen_len)

        self.has_kv_bias = has_kv_bias
        self.fully_resident = fully_resident
        self.decode_update = self.decode_update_has_kv_bias if has_kv_bias else self.decode_update_no_kv_bias

        # ── H2D traffic stats (GPU accumulators, zero-sync overhead) ─────
        # Disabled by default; enabled by the profiler for the last iteration.
        self._h2d_stats_enabled = False
        # Correction path (main-stream diff_offload inside decode_update_*)
        self._h2d_corr_miss = torch.tensor(0, dtype=torch.long, device=device)
        self._h2d_corr_total = torch.tensor(0, dtype=torch.long, device=device)
        self._h2d_corr_calls = 0
        # Prefetch path (decoupled q_future diff_offload, async stream)
        self._h2d_pf_miss = torch.tensor(0, dtype=torch.long, device=device)
        self._h2d_pf_total = torch.tensor(0, dtype=torch.long, device=device)
        self._h2d_pf_calls = 0
        # True miss: count(final_topk not in old_block_map) -- for waste quantification
        self._h2d_true_miss = torch.tensor(0, dtype=torch.long, device=device)
        self._h2d_true_total = torch.tensor(0, dtype=torch.long, device=device)
        # Shared side stream for prefill D2H KV offload. Assigned by the
        # dynamic cache wrapper so decode behavior stays unchanged.
        self._prefill_offload_stream = None
        self._prefill_prepared = False
        self._prefill_shape = None

    # ── H2D stats helpers ─────────────────────────────────────────────

    def record_prefetch_stats(self, load_ids=None):
        """Count miss blocks from load_ids (or _prefetch_load_mask) for the decoupled prefetch path.
        Called from decode_inference after PREP computes the prefetch mask.
        If load_ids is provided (staging mode), use it; otherwise fall back to _prefetch_load_mask."""
        if not self._h2d_stats_enabled:
            return
        ids = load_ids if load_ids is not None else self._prefetch_load_mask
        self._h2d_pf_miss += (ids >= 0).sum()
        self._h2d_pf_total += ids.numel()
        self._h2d_pf_calls += 1

    def record_true_miss_stats(self, block_map, topk_idx):
        """Count blocks in topk_idx that are NOT in block_map (true misses).
        Call BEFORE diff_offload to measure the theoretical minimum miss count.
        block_map: (H, B, M), topk_idx: (H, B, M) -- both int64."""
        if not self._h2d_stats_enabled:
            return
        # (H, B, M, 1) == (H, B, 1, M) → any match along last dim
        in_cache = (topk_idx.unsqueeze(-1) == block_map.unsqueeze(-2)).any(-1)
        self._h2d_true_miss += (~in_cache).sum()
        self._h2d_true_total += topk_idx.numel()

    def get_h2d_stats(self, _sync=True):
        """Return H2D stats dict for this layer."""
        if _sync:
            torch.cuda.synchronize()
        corr_miss = self._h2d_corr_miss.item()
        corr_total = self._h2d_corr_total.item()
        pf_miss = self._h2d_pf_miss.item()
        pf_total = self._h2d_pf_total.item()
        true_miss = self._h2d_true_miss.item()
        true_total = self._h2d_true_total.item()
        miss = corr_miss + pf_miss
        total = corr_total + pf_total
        wasted = miss - true_miss if true_total > 0 else 0
        elem_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        # Per miss block: K(block_size*D*elem) + V(block_size*D*elem) [+ bias(block_size*elem)]
        bytes_per_miss = self.block_size * self.head_dim * elem_size * 2
        if self.has_kv_bias:
            bytes_per_miss += self.block_size * elem_size
        stats = {
            "corr_miss": corr_miss, "corr_total": corr_total,
            "corr_calls": self._h2d_corr_calls,
            "pf_miss": pf_miss, "pf_total": pf_total,
            "pf_calls": self._h2d_pf_calls,
            "true_miss": true_miss, "true_total": true_total,
            "wasted_reload": wasted,
            "miss_blocks": miss, "total_blocks": total,
            "hit_rate": 1.0 - miss / total if total > 0 else 0.0,
            "h2d_bytes": miss * bytes_per_miss,
            "fully_resident": self.fully_resident,
        }
        return stats

    def reset_h2d_stats(self):
        """Reset H2D stats accumulators."""
        self._h2d_corr_miss.zero_()
        self._h2d_corr_total.zero_()
        self._h2d_corr_calls = 0
        self._h2d_pf_miss.zero_()
        self._h2d_pf_total.zero_()
        self._h2d_pf_calls = 0
        self._h2d_true_miss.zero_()
        self._h2d_true_total.zero_()

    def _run_diff_offload(self, topk_idx):
        """Compute new_map/load_mask with tail-slot protection semantics."""
        if hasattr(diff, "diff_offload_fused"):
            diff.diff_offload_fused(
                self._block_map,
                topk_idx,
                self._new_block_map_buf,
                self._load_mask,
            )
            return

        # Legacy kernel does not initialize untouched slots and lacks tail
        # protection. Initialize outputs first, then restore tail slot.
        self._new_block_map_buf.copy_(self._block_map)
        self._load_mask.fill_(-1)
        diff.diff_offload(self._block_map, topk_idx, self._new_block_map_buf, self._load_mask)
        self._new_block_map_buf[:, :, -1].copy_(self._block_map[:, :, -1])
        self._load_mask[:, :, -1].fill_(-1)

    def set_prefill_offload_stream(self, stream):
        self._prefill_offload_stream = stream

    def wait_prefill_offload(self):
        if self._prefill_offload_stream is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self._prefill_offload_stream)

    def prepare_prefill(self, total_bsz, seq_len):
        total_bsz = int(total_bsz)
        seq_len = int(seq_len)
        if self._prefill_prepared and self._prefill_shape == (total_bsz, seq_len):
            return
        H = self.head_num
        D = self.head_dim
        kv_shape = (total_bsz, seq_len + self.max_gen_len, H, D)
        gpu_kv_shape = (total_bsz, self.topk * self.block_size, H, D)
        gpu_bias_shape = (total_bsz, self.topk * self.block_size, H)
        block_map_shape = (H, total_bsz, self.topk)
        cache_lens_shape = (total_bsz,)

        if self.fully_resident:
            need_k_alloc = (
                self._k_cpu is None
                or self._k_cpu.shape != kv_shape
                or self._k_cpu.dtype != self.dtype
                or self._k_cpu.device != self.device
            )
            need_v_alloc = (
                self._v_cpu is None
                or self._v_cpu.shape != kv_shape
                or self._v_cpu.dtype != self.dtype
                or self._v_cpu.device != self.device
            )
            if need_k_alloc:
                self._k_cpu = torch.empty(kv_shape, dtype=self.dtype, device=self.device)
            if need_v_alloc:
                self._v_cpu = torch.empty(kv_shape, dtype=self.dtype, device=self.device)
        else:
            need_k_alloc = (
                self._k_cpu is None
                or self._k_cpu.shape != kv_shape
                or self._k_cpu.dtype != self.dtype
                or self._k_cpu.device.type != "cpu"
            )
            need_v_alloc = (
                self._v_cpu is None
                or self._v_cpu.shape != kv_shape
                or self._v_cpu.dtype != self.dtype
                or self._v_cpu.device.type != "cpu"
            )
            if need_k_alloc:
                self._k_cpu = torch.empty(kv_shape, dtype=self.dtype, device="cpu").pin_memory()
            if need_v_alloc:
                self._v_cpu = torch.empty(kv_shape, dtype=self.dtype, device="cpu").pin_memory()
        self._kv_cpu = None
        self._kv_cpu_flat_packed = None

        if (
            self._k_gpu is None
            or self._k_gpu.shape != gpu_kv_shape
            or self._k_gpu.dtype != self.dtype
            or self._k_gpu.device != self.device
        ):
            self._k_gpu = torch.empty(gpu_kv_shape, dtype=self.dtype, device=self.device)
            self._v_gpu = torch.empty(gpu_kv_shape, dtype=self.dtype, device=self.device)
            self._kv_bias_gpu = torch.empty(gpu_bias_shape, dtype=self.dtype, device=self.device)

        if (
            not hasattr(self, "_block_map")
            or self._block_map.shape != block_map_shape
            or self._block_map.device != self.device
        ):
            self._block_map = torch.empty(block_map_shape, dtype=torch.int64, device=self.device)
            self._new_block_map_buf = torch.empty_like(self._block_map)
            self._load_mask = torch.empty_like(self._block_map)
            self._prefetch_load_mask = torch.empty_like(self._block_map)
        self._block_map.fill_(-1)

        tail_len = seq_len % self.block_size
        if (
            not hasattr(self, "_cache_lens")
            or self._cache_lens.shape != cache_lens_shape
            or self._cache_lens.device != self.device
        ):
            self._cache_lens = torch.empty(cache_lens_shape, dtype=torch.int32, device=self.device)
        self._cache_lens.fill_((self.topk - 1) * self.block_size + tail_len)

        self._tail_block_len_on_gpu = tail_len
        self._tail_block_idx_on_gpu = self.topk - 1
        self.seq_length = seq_len
        self._prefill_prepared = True
        self._prefill_shape = (total_bsz, seq_len)

    def prefill_update(self, key_states, value_states, kv_bias, current_batch_pos, total_bsz):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # return: None
        B, S, H, D = key_states.shape

        assert H == self.head_num and D == self.head_dim

        num_full_blocks = S // self.block_size
        tail_len = S % self.block_size
        self.seq_length = S


        if current_batch_pos == 0 and not self._prefill_prepared:
            self.prepare_prefill(total_bsz, S)

        if self._tail_block_len_on_gpu > 0: # 如果有尾块，搬到gpu上
            _tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size

            self._k_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :, :].copy_(key_states[:, -self._tail_block_len_on_gpu:, :, :], non_blocking=True)

            self._v_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :, :].copy_(value_states[:, -self._tail_block_len_on_gpu:, :, :], non_blocking=True)

            self._kv_bias_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :].copy_(kv_bias[:, -self._tail_block_len_on_gpu:, :], non_blocking=True)

            self._block_map[:, current_batch_pos:current_batch_pos+B, self._tail_block_idx_on_gpu] = S // self.block_size
        else: # 如果没有尾块，在gpu上开一个块作为尾块，预备下一次decode写入
            self._block_map[:, current_batch_pos:current_batch_pos+B, self._tail_block_idx_on_gpu] = S // self.block_size

        # B. 把kv cache写到cpu
        if self.fully_resident or self._prefill_offload_stream is None:
            self._k_cpu[current_batch_pos:current_batch_pos+B, :S, :, :].copy_(key_states, non_blocking=True)
            self._v_cpu[current_batch_pos:current_batch_pos+B, :S, :, :].copy_(value_states, non_blocking=True)
        else:
            main_stream = torch.cuda.current_stream(device=self.device)
            with torch.cuda.stream(self._prefill_offload_stream):
                self._prefill_offload_stream.wait_stream(main_stream)
                self._k_cpu[current_batch_pos:current_batch_pos+B, :S, :, :].copy_(key_states, non_blocking=True)
                self._v_cpu[current_batch_pos:current_batch_pos+B, :S, :, :].copy_(value_states, non_blocking=True)
            key_states.record_stream(self._prefill_offload_stream)
            value_states.record_stream(self._prefill_offload_stream)

        return
    
    def decode_update_has_kv_bias(self, key_states, value_states, kv_bias, topk_idx, skip_offload=False):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # kv_bias: (head_num, seq_len, 1)
        # topk_idx: list, [H, B, M]
        # skip_offload: if True, skip diff_offload + H2D (InfLLMv2 decoupled layers 1+)
        # return: (self._k_gpu, self._v_gpu, mapped_topk_idx)
        B, S, H, D = key_states.shape
        assert H == self.head_num and D == self.head_dim and S == 1

        self.seq_length += 1
        self._cache_lens[...] += 1

        # A. 首先处理尾块。尾块一定写在GPU上，且self._tail_block_idx_on_gpu指示的位置一定是可写的
        _tail_write_pos = self._tail_block_idx_on_gpu * self.block_size + self._tail_block_len_on_gpu
        self._k_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(key_states, non_blocking=True)
        self._v_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(value_states, non_blocking=True)


        self._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos+1, :].copy_(kv_bias[:, self.seq_length-1:self.seq_length, :], non_blocking=True)


        self._tail_block_len_on_gpu += 1
        # 先判断要不要写回。如果需要写回，在完成该层的计算后再写回 TODO: 最后弄写回的逻辑
        tail_full = self._tail_block_len_on_gpu == self.block_size
        # B. 计算load mask和新的映射 尾块的映射不会动
        if not skip_offload:
            # Record true miss stats BEFORE diff_offload (for waste quantification)
            self.record_true_miss_stats(self._block_map, topk_idx)

            self._run_diff_offload(topk_idx)
            self._block_map.copy_(self._new_block_map_buf, non_blocking=True)

            # Accumulate correction-path H2D stats (gated by profiler)
            if self._h2d_stats_enabled:
                self._h2d_corr_miss += (self._load_mask >= 0).sum()
                self._h2d_corr_total += self._load_mask.numel()
                self._h2d_corr_calls += 1

            # C. 从disk取需要load进来的块
            flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, self.block_size)
            flash_h2d_from_mask_bias(self._v_gpu, self._v_cpu, self._kv_bias_gpu, kv_bias, self._load_mask, self.block_size)

        # D. 处理写回逻辑 TODO: 测试时CUDA Graph没有包进来这里的逻辑
        if tail_full:
            tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size
            cpu_block_base_pos = self.seq_length - self.block_size # 现在self.seq_len应该可以整除self.block_size
            self._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._k_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            self._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._v_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            # 复用self._tail_block_idx_on_gpu的位置，但是让block_map里这里指向下一个块，并且将这个块认为全部可写
            self._block_map[..., self._tail_block_idx_on_gpu] += 1
            self._tail_block_len_on_gpu = 0
            self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu
            

        return self._k_gpu, self._v_gpu, self._kv_bias_gpu, self._cache_lens 
    
    def decode_update_no_kv_bias(self, key_states, value_states, kv_bias, topk_idx, skip_offload=False):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # kv_bias: (head_num, seq_len, 1)
        # topk_idx: list, [H, B, M]
        # skip_offload: if True, skip diff_offload + H2D (InfLLMv2 decoupled layers 1+)
        # return: (self._k_gpu, self._v_gpu, mapped_topk_idx)
        B, S, H, D = key_states.shape
        assert H == self.head_num and D == self.head_dim and S == 1

        self.seq_length += 1
        self._cache_lens[...] += 1

        # A. 首先处理尾块。尾块一定写在GPU上，且self._tail_block_idx_on_gpu指示的位置一定是可写的
        _tail_write_pos = self._tail_block_idx_on_gpu * self.block_size + self._tail_block_len_on_gpu
        self._k_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(key_states, non_blocking=True)
        self._v_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(value_states, non_blocking=True)

        # 注意！用python写分支会非常慢
        if kv_bias != None:
            self._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos+1, :].copy_(kv_bias[:, self.seq_length-1:self.seq_length, :], non_blocking=True)


        self._tail_block_len_on_gpu += 1
        # 先判断要不要写回。如果需要写回，在完成该层的计算后再写回 TODO: 最后弄写回的逻辑
        tail_full = self._tail_block_len_on_gpu == self.block_size
        # B. 计算load mask和新的映射 尾块的映射不会动
        if not skip_offload:
            # Record true miss stats BEFORE diff_offload (for waste quantification)
            self.record_true_miss_stats(self._block_map, topk_idx)

            self._run_diff_offload(topk_idx)
            self._block_map.copy_(self._new_block_map_buf, non_blocking=True)

            # Accumulate correction-path H2D stats (gated by profiler)
            if self._h2d_stats_enabled:
                self._h2d_corr_miss += (self._load_mask >= 0).sum()
                self._h2d_corr_total += self._load_mask.numel()
                self._h2d_corr_calls += 1

            # C. 从disk取需要load进来的块
            flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, self.block_size)
            flash_h2d_from_mask(self._v_gpu, self._v_cpu, self._load_mask, self.block_size)

        # D. 处理写回逻辑 TODO: 测试时CUDA Graph没有包进来这里的逻辑
        if tail_full:
            tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size
            cpu_block_base_pos = self.seq_length - self.block_size # 现在self.seq_len应该可以整除self.block_size
            self._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._k_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            self._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._v_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            # 复用self._tail_block_idx_on_gpu的位置，但是让block_map里这里指向下一个块，并且将这个块认为全部可写
            self._block_map[..., self._tail_block_idx_on_gpu] += 1
            self._tail_block_len_on_gpu = 0
            self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu
            

        return self._k_gpu, self._v_gpu, self._kv_bias_gpu, self._cache_lens 

    def dense_decode_update(self, key_states, value_states):
        B, S, H, D = key_states.shape
        assert H == self.head_num and D == self.head_dim and S == 1

        write_pos = self.seq_length
        self._k_cpu[:, write_pos:write_pos + 1, :, :].copy_(key_states, non_blocking=True)
        self._v_cpu[:, write_pos:write_pos + 1, :, :].copy_(value_states, non_blocking=True)
        self.seq_length += 1

        full_k = self._k_cpu[:, :self.seq_length, :, :]
        full_v = self._v_cpu[:, :self.seq_length, :, :]
        if full_k.device != self.device:
            full_k = full_k.to(self.device, non_blocking=False)
            full_v = full_v.to(self.device, non_blocking=False)
        cache_lens = torch.full((B,), self.seq_length, dtype=torch.int32, device=self.device)
        return full_k.contiguous(), full_v.contiguous(), cache_lens
        


class InfLLMv2CacheLayer(DynamicLayer):
    def __init__(self, config, has_kv_bias, fully_resident=False, max_gen_len=8192):
        super().__init__()
        # Initialize any additional attributes specific to InfLLMv2CacheLayer
        self.no_rope_keys = torch.tensor([], dtype=torch.float32)
        self.compress_k_cache = []
        self.no_compress_k_cache = []
        self.cached_compressed_cu_seqlens = torch.tensor([], dtype=torch.int32)
        self.compress_k_cache_varlen = torch.tensor([], dtype=torch.float32)
        self.tail_cis = None
        self.compressed_cis = None
        self.tail_cis_len = 0
        self.total_cis = None

        # Read block_size and topk from config.sparse_config (MiniCPM-style), with defaults
        _sparse_cfg = getattr(config, 'sparse_config', None) or {}
        _block_size = int(_sparse_cfg.get('block_size', 64))
        _topk = int(_sparse_cfg.get('topk', 64))
        self.cache_engine = CacheEngine(
            gpu_mem_usage=12,
            num_layers=config.num_hidden_layers,
            block_size=_block_size,
            head_num=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            has_kv_bias=has_kv_bias,
            topk=_topk,
            fully_resident=fully_resident,
            max_gen_len=max_gen_len,
        )
        self.seq_length = 0

    def update_no_rope_key(self, key_states):
        if self.no_rope_keys.numel() == 0:
            self.no_rope_keys = key_states
        else:
            self.no_rope_keys = torch.cat([self.no_rope_keys, key_states], dim=1)
        return self.no_rope_keys

    def update_compress_k(self, key_states, cu_seqlens=None):
        if len(self.compress_k_cache) == 0:
            if cu_seqlens is not None:
                self.cached_compressed_cu_seqlens = cu_seqlens.clone()
            self.compress_k_cache_varlen = key_states
            split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            self.compress_k_cache = list(torch.split(key_states, split_sizes))
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    self.compress_k_cache[index] = torch.cat([self.compress_k_cache[index], k], dim=0)
            new_seq_lens = torch.tensor([tensor.shape[0] for tensor in self.compress_k_cache], dtype=torch.int32)
            new_cumsum = torch.cumsum(new_seq_lens, dim=0, dtype=torch.int32)
            
            self.compress_k_cache_varlen = torch.cat(self.compress_k_cache, dim=0)
            self.cached_compressed_cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), new_cumsum]).to(self.compress_k_cache_varlen.device)
        return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens
    

    def update_compress_k_prefill(self, key_states, cu_seqlens, max_seqlen, current_batch_pos, total_bsz):
        key_shape = key_states.size()
        key_states_pad = key_states.view(-1, max_seqlen, key_shape[1], key_shape[2]).contiguous()
        expected_shape = (total_bsz, max_seqlen, key_shape[1], key_shape[2])
        need_alloc = (
            (not torch.is_tensor(self.compress_k_cache_varlen))
            or self.compress_k_cache_varlen.shape != expected_shape
            or self.compress_k_cache_varlen.dtype != key_states.dtype
            or self.compress_k_cache_varlen.device != key_states.device
        )
        if need_alloc:
            self.compress_k_cache_varlen = torch.empty(expected_shape, dtype=key_states.dtype, device=key_states.device)
        self.compress_k_cache_varlen[current_batch_pos:current_batch_pos+key_states_pad.shape[0]].copy_(key_states_pad, non_blocking=True)
        if (
            current_batch_pos == 0
            or (not torch.is_tensor(self.cached_compressed_cu_seqlens))
            or self.cached_compressed_cu_seqlens.numel() != (total_bsz + 1)
            or self.cached_compressed_cu_seqlens.dtype != cu_seqlens.dtype
            or self.cached_compressed_cu_seqlens.device != cu_seqlens.device
        ):
            self.cached_compressed_cu_seqlens = torch.empty((total_bsz + 1,), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            self.cached_compressed_cu_seqlens[:cu_seqlens.shape[-1]].copy_(cu_seqlens)
            self.cached_compressed_max_seqlen = max_seqlen
            self.cached_compressed_cu_seqlens_adder = torch.arange(total_bsz, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        else:
            self.cached_compressed_cu_seqlens[current_batch_pos:current_batch_pos+cu_seqlens.shape[-1]].copy_(cu_seqlens + self.cached_compressed_cu_seqlens[current_batch_pos])


    def update_compress_k_decode(self, key_states, cu_seqlens):
        if key_states is None:
            return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens, self.cached_compressed_max_seqlen
        
        self.compress_k_cache_varlen = torch.cat((self.compress_k_cache_varlen, key_states), dim=1)
        self.cached_compressed_cu_seqlens += self.cached_compressed_cu_seqlens_adder
        self.cached_compressed_max_seqlen += 1
        return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens, self.cached_compressed_max_seqlen

    def update_no_compress_k_prefill(self, key_states, kernel_size=32, kernel_stride=16, current_batch_pos=None, total_bsz=None):
        # key_states: (B, N, H, D)
        B, N, H, D = key_states.shape
        expected_shape = (total_bsz, kernel_size, H, D)
        need_alloc = (
            (not torch.is_tensor(self.no_compress_k_cache))
            or self.no_compress_k_cache.shape != expected_shape
            or self.no_compress_k_cache.dtype != key_states.dtype
            or self.no_compress_k_cache.device != key_states.device
        )
        if need_alloc:
            self.no_compress_k_cache = torch.empty(expected_shape, dtype=key_states.dtype, device=key_states.device)
        self.no_compress_k_cache[current_batch_pos:current_batch_pos+B, :N, :, :].copy_(key_states)
        if current_batch_pos == 0 or not hasattr(self, "no_compress_k_len"):
            self.no_compress_k_len = N
        return

    def update_no_compress_k_decode(self, key_states, kernel_size=32, kernel_stride=16):
        # key_states: (B, N, H, D)
        if self.no_compress_k_len >= kernel_size:
            to_compress = self.no_compress_k_cache.clone()
            self.no_compress_k_cache[:, :kernel_stride].copy_(self.no_compress_k_cache[:, kernel_stride:])
            self.no_compress_k_cache[:, kernel_stride:kernel_stride+1].copy_(key_states)
            self.no_compress_k_len = kernel_stride + 1
            return to_compress
        else:
            self.no_compress_k_cache[:, self.no_compress_k_len:self.no_compress_k_len+1, :, :].copy_(key_states)
            self.no_compress_k_len += 1
            return None
    

    # 从update函数拆出来的
    def prefill_update_kv(self, key_states, value_states, kv_bias, current_batch_pos, total_bsz):
        self.seq_length = key_states.shape[1]
        self.cache_engine.prefill_update(key_states, value_states, kv_bias, current_batch_pos, total_bsz)

    def prepare_prefill(self, total_bsz, seq_len):
        self.seq_length = seq_len
        self.cache_engine.prepare_prefill(total_bsz, seq_len)

    def wait_prefill_offload(self):
        self.cache_engine.wait_prefill_offload()
    
    def decode_update_kv(self, key_states, value_states, kv_bias, topk_idx, skip_offload=False):
        self.seq_length += 1
        return self.cache_engine.decode_update(key_states.unsqueeze(1), value_states.unsqueeze(1), kv_bias, topk_idx, skip_offload=skip_offload)

    def dense_decode_update_kv(self, key_states, value_states):
        self.seq_length += 1
        return self.cache_engine.dense_decode_update(key_states.unsqueeze(1), value_states.unsqueeze(1))


    def update(self, key_states, value_states, cache_kwargs=None):
        is_prefill = cache_kwargs.get("is_prefill", True)
        if is_prefill: # prefill
            self.seq_length = key_states.shape[2]
            self.cache_engine.prefill_update(key_states.transpose(1, 2), value_states.transpose(1, 2))
            return key_states, value_states
        else: # decode
            topk_idx = cache_kwargs.get("topk_idx", None)
            self.seq_length += 1
            output = self.cache_engine.decode_update(key_states, value_states, topk_idx)
            return output
    
    def update_cis(self, cis, current_batch_pos, total_bsz, kernel_size=32, kernel_stride=16):
        # cis: (H, B, N)
        H, B, N = cis.shape
        if N > 1: # prefill
            if self.compressed_cis == None:
                full_block_len = N // kernel_stride
                tail_block_len = N % kernel_stride
                full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
                comp_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)
                M = comp_cis.shape[-1]
                self.comp_cis_len = M
                self.compressed_cis =  torch.empty((H, total_bsz, M), dtype=cis.dtype, device=cis.device)
                self.compressed_cis[:, :B, :].copy_(comp_cis, non_blocking=True)
                self.tail_cis = torch.empty((H, total_bsz, kernel_size), dtype=cis.dtype, device=cis.device)
                self.tail_cis_len = kernel_stride + tail_block_len
                self.tail_cis[:, :B, :self.tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * kernel_stride:], non_blocking=True)
                return comp_cis
            else:
                full_block_len = N // kernel_stride
                tail_block_len = N % kernel_stride
                full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
                comp_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)
                M = comp_cis.shape[-1]
                self.compressed_cis[:, current_batch_pos:current_batch_pos+B, :].copy_(comp_cis, non_blocking=True)
                self.tail_cis[:, current_batch_pos:current_batch_pos+B, :self.tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * kernel_stride:], non_blocking=True)
                return comp_cis
            # full_block_len = N // kernel_stride
            # tail_block_len = N % kernel_stride
            # full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
            # self.compressed_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)

            # self.tail_cis = [cis[:, :, (full_block_len - 1) * kernel_stride:]]
            # self.tail_cis_len = kernel_stride + tail_block_len
        else:
            # self.tail_cis.append(cis)
            self.tail_cis[:, :, self.tail_cis_len:self.tail_cis_len+1].copy_(cis, non_blocking=True)
            self.tail_cis_len += 1
            if self.tail_cis_len == kernel_size:
                # tail = torch.cat(self.tail_cis, dim=-1)
                tail = self.tail_cis
                tail_comp = tail.mean(dim=-1, keepdim=True)
                # self.compressed_cis = torch.cat((self.compressed_cis, tail_comp), dim=-1)
                self.compressed_cis[:, :, self.comp_cis_len:self.comp_cis_len+1].copy_(tail_comp, non_blocking=True)
                self.comp_cis_len += 1
                # self.tail_cis = [tail[:, :, kernel_stride:]]
                self.tail_cis[:, :, :kernel_stride].copy_(self.tail_cis[:, :, kernel_stride:])
                self.tail_cis_len = kernel_stride
            return self.compressed_cis
    
    def update_uncompressed_cis(self, cis, current_batch_pos, total_bsz):
        if self.total_cis is None:
            B, S, H = cis.shape
            self.total_cis = torch.empty((total_bsz, S+1024, H), dtype=cis.dtype, device=cis.device)
            self.total_cis[:B, :S, :].copy_(cis, non_blocking=True)
            self.cis_len = S
            self.cis_prefilled_batch = B
            return cis # prefill用的是fa接口
        elif self.cis_prefilled_batch < total_bsz:
            B, S, H = cis.shape
            assert self.cis_len == S
            self.total_cis[current_batch_pos:current_batch_pos+B, :S, :].copy_(cis, non_blocking=True)
            self.cis_prefilled_batch += B
            return cis
        else:
            self.total_cis[:, self.cis_len:self.cis_len+1, :].copy_(cis, non_blocking=True)
            self.cis_len += 1
            return self.total_cis # 用的是flash_attn_nosa的接口，为了用cuda graph保证地址不变
        
    def get_paged_kv(self):
        return self.cache_engine._k, self.cache_engine._v
    
    def get_seq_length(self):
        return self.seq_length


class InfLLMv2Cache(DynamicCache):
    def __init__(self,
                 config, num_hidden_layers: Optional[int] = None, has_kv_bias=False,
                 l0_resident=False, max_gen_len: int = 8192) -> None:
        super().__init__(config=config)
        self.layers = [
            InfLLMv2CacheLayer(
                config,
                has_kv_bias,
                fully_resident=(i == 0 and l0_resident),
                max_gen_len=max_gen_len,
            )
            for i in range(num_hidden_layers)
        ] if num_hidden_layers else []
        self._seen_tokens = 0
        self._prefill_offload_stream = None
        if self.layers and any(not l.cache_engine.fully_resident for l in self.layers):
            device = self.layers[0].cache_engine.device
            self._prefill_offload_stream = torch.cuda.Stream(device=device, priority=0)
            for layer in self.layers:
                layer.cache_engine.set_prefill_offload_stream(self._prefill_offload_stream)

    def prefill_update_kv(self, key_states, value_states, kv_bias, layer_idx, current_batch_pos, total_bsz):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].prefill_update_kv(key_states, value_states, kv_bias, current_batch_pos, total_bsz)

    def prepare_prefill(self, total_bsz, seq_len):
        for layer in self.layers:
            layer.prepare_prefill(total_bsz, seq_len)
    
    def decode_update_kv(self, key_states, value_states, kv_bias, layer_idx, topk_idx, skip_offload=False):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].decode_update_kv(key_states, value_states, kv_bias, topk_idx, skip_offload=skip_offload)

    def dense_decode_update_kv(self, key_states, value_states, layer_idx):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].dense_decode_update_kv(key_states, value_states)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
    
    def update_cis(self, cis, layer_idx, current_batch_pos, total_bsz):
        return self.layers[layer_idx].update_cis(cis, current_batch_pos, total_bsz)

    def update_uncompressed_cis(self, cis, layer_idx, current_batch_pos, total_bsz):
        return self.layers[layer_idx].update_uncompressed_cis(cis, current_batch_pos, total_bsz)
    
    def get_paged_kv(self, layer_idx):
        return self.layers[layer_idx].get_paged_kv()

    def update_no_rope_key(self, key_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update_no_rope_key(key_states)

    def update_compress_k_prefill(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None, max_seqlen=None, current_batch_pos=None, total_bsz=None):
        return self.layers[layer_idx].update_compress_k_prefill(key_states, cu_seqlens, max_seqlen, current_batch_pos, total_bsz)

    def update_compress_k_decode(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k_decode(key_states, cu_seqlens)

    def update_no_compress_k_prefill(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None, current_batch_pos=None, total_bsz=None):
        return self.layers[layer_idx].update_no_compress_k_prefill(key_states, kernel_size, kernel_stride, current_batch_pos=current_batch_pos, total_bsz=total_bsz)

    def update_no_compress_k_decode(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None):
        return self.layers[layer_idx].update_no_compress_k_decode(key_states, kernel_size, kernel_stride)

    def crop(self, max_length):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)

    def wait_prefill_offload(self):
        if self._prefill_offload_stream is not None:
            torch.cuda.current_stream(device=self.layers[0].cache_engine.device).wait_stream(
                self._prefill_offload_stream
            )
    
    def enable_h2d_stats(self):
        """Enable H2D stats collection and reset accumulators."""
        for l in self.layers:
            l.cache_engine.reset_h2d_stats()
            l.cache_engine._h2d_stats_enabled = True

    def disable_h2d_stats(self):
        """Disable H2D stats collection."""
        for l in self.layers:
            l.cache_engine._h2d_stats_enabled = False

    def get_h2d_stats(self):
        """Return aggregate + per-layer H2D stats."""
        torch.cuda.synchronize()
        per_layer = [l.cache_engine.get_h2d_stats(_sync=False) for l in self.layers]
        miss = sum(s["miss_blocks"] for s in per_layer)
        total = sum(s["total_blocks"] for s in per_layer)
        h2d_bytes = sum(s["h2d_bytes"] for s in per_layer if not s.get("fully_resident", False))
        ce0 = self.layers[0].cache_engine
        true_miss = sum(s.get("true_miss", 0) for s in per_layer)
        wasted = sum(s.get("wasted_reload", 0) for s in per_layer)
        out = {
            "miss_blocks": miss,
            "total_blocks": total,
            "hit_rate": 1.0 - miss / total if total > 0 else 0.0,
            "h2d_bytes": h2d_bytes,
            "true_miss": true_miss,
            "wasted_reload": wasted,
            "block_size": ce0.block_size,
            "head_dim": ce0.head_dim,
            "elem_size": 2 if ce0.dtype in [torch.float16, torch.bfloat16] else 4,
            "num_layers": len(self.layers),
            "per_layer": per_layer,
        }
        return out

    def reset_h2d_stats(self):
        """Reset H2D stats across all layers."""
        for l in self.layers:
            l.cache_engine.reset_h2d_stats()

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].get_seq_length()
    
    def get_max_length(self):
        raise NotImplementedError
