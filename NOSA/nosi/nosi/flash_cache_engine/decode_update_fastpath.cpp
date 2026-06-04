// This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
// Copyright (c) 2026 THUNLP.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <torch/extension.h>

namespace py = pybind11;

void decode_update_fastpath_cuda(
    at::Tensor k_gpu,      // [B, S_GPU, H, D]
    at::Tensor v_gpu,      // [B, S_GPU, H, D]
    at::Tensor key_new,    // [B, H, D]
    at::Tensor val_new,    // [B, H, D]
    at::Tensor cache_lens, // [B] int32
    int64_t tail_write_pos,
    c10::optional<at::Tensor> bias_gpu,  // [B, S_GPU, H] or None
    c10::optional<at::Tensor> bias_src,  // [B, S_total, H] or None
    int64_t bias_src_pos
);

void decode_update_fastpath(
    torch::Tensor k_gpu,
    torch::Tensor v_gpu,
    torch::Tensor key_new,
    torch::Tensor val_new,
    torch::Tensor cache_lens,
    int64_t tail_write_pos,
    c10::optional<torch::Tensor> bias_gpu = c10::nullopt,
    c10::optional<torch::Tensor> bias_src = c10::nullopt,
    int64_t bias_src_pos = -1
) {
    decode_update_fastpath_cuda(
        k_gpu, v_gpu, key_new, val_new, cache_lens, tail_write_pos,
        bias_gpu, bias_src, bias_src_pos
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_update_fastpath", &decode_update_fastpath, "",
          py::arg("k_gpu"),
          py::arg("v_gpu"),
          py::arg("key_new"),
          py::arg("val_new"),
          py::arg("cache_lens"),
          py::arg("tail_write_pos"),
          py::arg("bias_gpu") = py::none(),
          py::arg("bias_src") = py::none(),
          py::arg("bias_src_pos") = -1);
}
