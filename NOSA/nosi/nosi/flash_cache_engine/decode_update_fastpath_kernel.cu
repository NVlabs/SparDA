// This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
// Copyright (c) 2026 THUNLP.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void decode_update_tail_write_kernel(
    scalar_t* __restrict__ k_gpu,      // [B, S_GPU, H, D]
    scalar_t* __restrict__ v_gpu,      // [B, S_GPU, H, D]
    const scalar_t* __restrict__ key_new, // [B, H, D]
    const scalar_t* __restrict__ val_new, // [B, H, D]
    int32_t* __restrict__ cache_lens,  // [B]
    scalar_t* __restrict__ bias_gpu,   // [B, S_GPU, H] or nullptr
    const scalar_t* __restrict__ bias_src, // [B, S_total, H] or nullptr
    int64_t B,
    int64_t H,
    int64_t D,
    int64_t tail_write_pos,
    int64_t bias_src_pos,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3,
    int64_t v_s0, int64_t v_s1, int64_t v_s2, int64_t v_s3,
    int64_t kn_s0, int64_t kn_s1, int64_t kn_s2,
    int64_t vn_s0, int64_t vn_s1, int64_t vn_s2,
    int64_t bg_s0, int64_t bg_s1, int64_t bg_s2,
    int64_t bs_s0, int64_t bs_s1, int64_t bs_s2
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_kv = B * H * D;

    if (idx < B) {
        cache_lens[idx] += 1;
    }

    if (idx >= total_kv) {
        return;
    }

    int64_t d = idx % D;
    int64_t h = (idx / D) % H;
    int64_t b = idx / (H * D);

    int64_t k_dst = b * k_s0 + tail_write_pos * k_s1 + h * k_s2 + d * k_s3;
    int64_t v_dst = b * v_s0 + tail_write_pos * v_s1 + h * v_s2 + d * v_s3;
    int64_t k_src = b * kn_s0 + h * kn_s1 + d * kn_s2;
    int64_t v_src = b * vn_s0 + h * vn_s1 + d * vn_s2;

    k_gpu[k_dst] = key_new[k_src];
    v_gpu[v_dst] = val_new[v_src];

    if (d == 0 && bias_gpu != nullptr) {
        bias_gpu[b * bg_s0 + tail_write_pos * bg_s1 + h * bg_s2] =
            bias_src[b * bs_s0 + bias_src_pos * bs_s1 + h * bs_s2];
    }
}

void decode_update_fastpath_cuda(
    at::Tensor k_gpu,      // [B, S_GPU, H, D]
    at::Tensor v_gpu,      // [B, S_GPU, H, D]
    at::Tensor key_new,    // [B, H, D]
    at::Tensor val_new,    // [B, H, D]
    at::Tensor cache_lens, // [B] int32
    int64_t tail_write_pos,
    c10::optional<at::Tensor> bias_gpu_opt,  // [B, S_GPU, H] or None
    c10::optional<at::Tensor> bias_src_opt,  // [B, S_total, H] or None
    int64_t bias_src_pos
) {
    TORCH_CHECK(k_gpu.is_cuda(), "k_gpu must be CUDA tensor");
    TORCH_CHECK(v_gpu.is_cuda(), "v_gpu must be CUDA tensor");
    TORCH_CHECK(key_new.is_cuda(), "key_new must be CUDA tensor");
    TORCH_CHECK(val_new.is_cuda(), "val_new must be CUDA tensor");
    TORCH_CHECK(cache_lens.is_cuda(), "cache_lens must be CUDA tensor");

    TORCH_CHECK(k_gpu.dim() == 4, "k_gpu must be [B,S_GPU,H,D]");
    TORCH_CHECK(v_gpu.dim() == 4, "v_gpu must be [B,S_GPU,H,D]");
    TORCH_CHECK(key_new.dim() == 3, "key_new must be [B,H,D]");
    TORCH_CHECK(val_new.dim() == 3, "val_new must be [B,H,D]");
    TORCH_CHECK(cache_lens.dim() == 1, "cache_lens must be [B]");

    TORCH_CHECK(
        k_gpu.scalar_type() == v_gpu.scalar_type() &&
            k_gpu.scalar_type() == key_new.scalar_type() &&
            k_gpu.scalar_type() == val_new.scalar_type(),
        "k/v/key/val dtypes must match"
    );
    TORCH_CHECK(cache_lens.scalar_type() == at::kInt, "cache_lens must be int32");

    int64_t B = key_new.size(0);
    int64_t H = key_new.size(1);
    int64_t D = key_new.size(2);
    TORCH_CHECK(val_new.size(0) == B && val_new.size(1) == H && val_new.size(2) == D,
                "val_new shape must match key_new");

    TORCH_CHECK(k_gpu.size(0) == B && k_gpu.size(2) == H && k_gpu.size(3) == D,
                "k_gpu shape must match [B,*,H,D] for key_new");
    TORCH_CHECK(v_gpu.size(0) == B && v_gpu.size(2) == H && v_gpu.size(3) == D,
                "v_gpu shape must match [B,*,H,D] for val_new");
    TORCH_CHECK(cache_lens.size(0) == B, "cache_lens size must be B");
    TORCH_CHECK(tail_write_pos >= 0 && tail_write_pos < k_gpu.size(1),
                "tail_write_pos out of range");

    TORCH_CHECK(
        bias_gpu_opt.has_value() == bias_src_opt.has_value(),
        "bias_gpu and bias_src must be both provided or both omitted"
    );
    bool has_bias = bias_gpu_opt.has_value();
    at::Tensor bias_gpu, bias_src;
    int64_t bg_s0 = 0, bg_s1 = 0, bg_s2 = 0;
    int64_t bs_s0 = 0, bs_s1 = 0, bs_s2 = 0;

    if (has_bias) {
        bias_gpu = *bias_gpu_opt;
        bias_src = *bias_src_opt;
        TORCH_CHECK(bias_gpu.is_cuda(), "bias_gpu must be CUDA tensor");
        TORCH_CHECK(bias_src.is_cuda(), "bias_src must be CUDA tensor");
        TORCH_CHECK(bias_gpu.dim() == 3, "bias_gpu must be [B,S_GPU,H]");
        TORCH_CHECK(bias_src.dim() == 3, "bias_src must be [B,S_total,H]");
        TORCH_CHECK(
            bias_gpu.size(0) == B && bias_gpu.size(1) == k_gpu.size(1) && bias_gpu.size(2) == H,
            "bias_gpu shape must be [B,S_GPU,H] matching k_gpu"
        );
        TORCH_CHECK(
            bias_src.size(0) == B && bias_src.size(2) == H,
            "bias_src shape must be [B,S_total,H] matching key/value heads"
        );
        TORCH_CHECK(bias_gpu.scalar_type() == k_gpu.scalar_type(),
                     "bias_gpu dtype must match k_gpu");
        TORCH_CHECK(bias_src.scalar_type() == k_gpu.scalar_type(),
                     "bias_src dtype must match k_gpu");
        TORCH_CHECK(bias_src_pos >= 0 && bias_src_pos < bias_src.size(1),
                     "bias_src_pos out of range");
        auto bg_str = bias_gpu.strides();
        auto bs_str = bias_src.strides();
        bg_s0 = bg_str[0]; bg_s1 = bg_str[1]; bg_s2 = bg_str[2];
        bs_s0 = bs_str[0]; bs_s1 = bs_str[1]; bs_s2 = bs_str[2];
    }

    auto k_str = k_gpu.strides();
    auto v_str = v_gpu.strides();
    auto kn_str = key_new.strides();
    auto vn_str = val_new.strides();

    int64_t total = std::max<int64_t>(B, B * H * D);
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k_gpu.scalar_type(),
        "decode_update_fastpath_cuda",
        [&] {
            scalar_t* bg_ptr = has_bias ? bias_gpu.data_ptr<scalar_t>() : nullptr;
            scalar_t* bs_ptr = has_bias ? bias_src.data_ptr<scalar_t>() : nullptr;
            decode_update_tail_write_kernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(
                    k_gpu.data_ptr<scalar_t>(),
                    v_gpu.data_ptr<scalar_t>(),
                    key_new.data_ptr<scalar_t>(),
                    val_new.data_ptr<scalar_t>(),
                    cache_lens.data_ptr<int32_t>(),
                    bg_ptr,
                    bs_ptr,
                    B,
                    H,
                    D,
                    tail_write_pos,
                    bias_src_pos,
                    k_str[0], k_str[1], k_str[2], k_str[3],
                    v_str[0], v_str[1], v_str[2], v_str[3],
                    kn_str[0], kn_str[1], kn_str[2],
                    vn_str[0], vn_str[1], vn_str[2],
                    bg_s0, bg_s1, bg_s2,
                    bs_s0, bs_s1, bs_s2
                );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
