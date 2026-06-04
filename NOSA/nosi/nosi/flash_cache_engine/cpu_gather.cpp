// This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
// Copyright (c) 2026 THUNLP.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <torch/extension.h>
#include <omp.h>
#include <immintrin.h>
#include <cstring>

/*
 * Multithreaded CPU gather with software prefetching.
 *
 * Two variants:
 *   cpu_gather(src, indices)  — allocates output, gathers one tensor
 *   cpu_gather_kv_blocks(...) — block-oriented fused K+V gather with
 *       fixed-stride inner loop for HW prefetcher, writes to compact
 *       pinned staging buffers.
 */

#define PREFETCH_DISTANCE 5

torch::Tensor cpu_gather(
    const torch::Tensor& src,      // (N, D) CPU, contiguous
    const torch::Tensor& indices    // (M,) int64 CPU
) {
    TORCH_CHECK(src.is_cpu(), "src must be a CPU tensor");
    TORCH_CHECK(indices.is_cpu(), "indices must be a CPU tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");

    const int64_t M = indices.size(0);
    const int64_t D = src.size(1);
    const int64_t row_bytes = D * src.element_size();

    auto dst = torch::empty({M, D}, src.options());
    if (M == 0) return dst;

    const char* src_ptr = reinterpret_cast<const char*>(src.data_ptr());
    char* dst_ptr = reinterpret_cast<char*>(dst.data_ptr());
    const int64_t* idx_ptr = indices.data_ptr<int64_t>();

    const int num_threads = std::min((int)M, omp_get_max_threads());

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const int64_t chunk = (M + num_threads - 1) / num_threads;
        const int64_t start = tid * chunk;
        const int64_t end = std::min(start + chunk, M);

        for (int64_t i = start; i < end; ++i) {
            const int64_t src_row = idx_ptr[i];
            const char* src_row_ptr = src_ptr + src_row * row_bytes;

            if (i + PREFETCH_DISTANCE < end) {
                const int64_t pf_row = idx_ptr[i + PREFETCH_DISTANCE];
                _mm_prefetch(src_ptr + pf_row * row_bytes, _MM_HINT_NTA);
            }

            std::memcpy(dst_ptr + i * row_bytes, src_row_ptr, row_bytes);
        }
    }

    return dst;
}

/*
 * Block-oriented fused K+V gather into compact pinned staging buffers.
 *
 * Processes one complete block at a time: the inner loop has a fixed
 * source stride (H * row_bytes), enabling the hardware prefetcher to
 * pipeline DRAM accesses.  Output is compact: miss-block 0 occupies
 * rows [0, block_size), miss-block 1 occupies [block_size, 2*block_size),
 * etc. — matching the dst_linear order computed in Python for GPU scatter.
 *
 * Source layout: (B, S, H, D) flattened to (B*S*H, D).
 * Dest layout:   compact (num_miss * block_size, D).
 */
void cpu_gather_kv_blocks(
    const torch::Tensor& k_src,     // (B*S*H, D) pinned CPU KV cache
    const torch::Tensor& v_src,     // (B*S*H, D)
    const torch::Tensor& miss_b,    // (num_miss,) int64
    const torch::Tensor& miss_h,    // (num_miss,) int64
    const torch::Tensor& miss_blk,  // (num_miss,) int64 — CPU block IDs
    const torch::Tensor& miss_slot, // (num_miss,) int64 — GPU slot IDs (unused here)
    torch::Tensor& k_dst,           // (num_miss * block_size, D) pinned
    torch::Tensor& v_dst,           // (num_miss * block_size, D) pinned
    int64_t S,                       // CPU seq length (dim 1 of B,S,H,D)
    int64_t gpu_S,                   // unused (compact output)
    int64_t H,                       // num_kv_heads
    int64_t block_size,              // tokens per block (e.g. 64)
    int64_t seq_length               // actual filled seq length (clamp)
) {
    const int64_t num_miss = miss_b.size(0);
    if (num_miss == 0) return;

    const int64_t D = k_src.size(1);
    const int64_t row_bytes = D * k_src.element_size();
    const int64_t src_stride = H * row_bytes;  // between consecutive tokens, same (b,h)

    const char* k_ptr = reinterpret_cast<const char*>(k_src.data_ptr());
    const char* v_ptr = reinterpret_cast<const char*>(v_src.data_ptr());
    char* k_out = reinterpret_cast<char*>(k_dst.data_ptr());
    char* v_out = reinterpret_cast<char*>(v_dst.data_ptr());

    const int64_t* b_ptr   = miss_b.data_ptr<int64_t>();
    const int64_t* h_ptr   = miss_h.data_ptr<int64_t>();
    const int64_t* blk_ptr = miss_blk.data_ptr<int64_t>();

    const int64_t max_src_token = seq_length - 1;

    const int num_threads = std::min((int)num_miss, omp_get_max_threads());

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const int64_t chunk = (num_miss + num_threads - 1) / num_threads;
        const int64_t start = tid * chunk;
        const int64_t end = std::min(start + chunk, num_miss);

        for (int64_t mi = start; mi < end; ++mi) {
            const int64_t b   = b_ptr[mi];
            const int64_t h   = h_ptr[mi];
            const int64_t blk = blk_ptr[mi];

            // Source: b * S * H + blk * block_size * H + h  (strided by H)
            const int64_t src_base = b * S * H + blk * block_size * H + h;
            const char* k_src_base = k_ptr + src_base * row_bytes;
            const char* v_src_base = v_ptr + src_base * row_bytes;

            // Dest: compact, mi-th block at rows [mi*block_size, (mi+1)*block_size)
            char* k_dst_base = k_out + mi * block_size * row_bytes;
            char* v_dst_base = v_out + mi * block_size * row_bytes;

            // Prefetch first rows of NEXT block
            if (mi + 1 < end) {
                const int64_t nb = b_ptr[mi + 1];
                const int64_t nh = h_ptr[mi + 1];
                const int64_t nblk = blk_ptr[mi + 1];
                const int64_t next_base = nb * S * H + nblk * block_size * H + nh;
                _mm_prefetch(k_ptr + next_base * row_bytes, _MM_HINT_T0);
                _mm_prefetch(v_ptr + next_base * row_bytes, _MM_HINT_T0);
            }

            // Inner loop: fixed source stride, sequential dest writes
            for (int64_t t = 0; t < block_size; ++t) {
                if (blk * block_size + t > max_src_token) break;

                const size_t s_off = t * src_stride;
                const size_t d_off = t * row_bytes;  // compact: sequential

                if (t + PREFETCH_DISTANCE < block_size) {
                    _mm_prefetch(k_src_base + (t + PREFETCH_DISTANCE) * src_stride, _MM_HINT_NTA);
                    _mm_prefetch(v_src_base + (t + PREFETCH_DISTANCE) * src_stride, _MM_HINT_NTA);
                }

                std::memcpy(k_dst_base + d_off, k_src_base + s_off, row_bytes);
                std::memcpy(v_dst_base + d_off, v_src_base + s_off, row_bytes);
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_gather", &cpu_gather,
          "Multithreaded CPU gather with software prefetching",
          py::arg("src"), py::arg("indices"),
          py::call_guard<py::gil_scoped_release>());
    m.def("cpu_gather_kv_blocks", &cpu_gather_kv_blocks,
          "Block-oriented fused K+V gather — fixed-stride inner loop for HW prefetcher",
          py::arg("k_src"), py::arg("v_src"),
          py::arg("miss_b"), py::arg("miss_h"), py::arg("miss_blk"), py::arg("miss_slot"),
          py::arg("k_dst"), py::arg("v_dst"),
          py::arg("S"), py::arg("gpu_S"), py::arg("H"),
          py::arg("block_size"), py::arg("seq_length"),
          py::call_guard<py::gil_scoped_release>());
}
