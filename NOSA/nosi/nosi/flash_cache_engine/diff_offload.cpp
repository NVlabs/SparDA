// This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
// Copyright (c) 2026 THUNLP.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <torch/extension.h>

void diff_offload_cuda(
    at::Tensor old_map,     // [H,B,M] in
    at::Tensor new_act,     // [H,B,M] in 
    at::Tensor new_map,     // [H,B,M] out
    at::Tensor load_map     // [H,B,M] out
);
void diff_offload_fused_cuda(
    at::Tensor old_map,     // [H,B,M] in
    at::Tensor new_act,     // [H,B,M] in
    at::Tensor new_map,     // [H,B,M] out
    at::Tensor load_map     // [H,B,M] out
);

void diff_offload(
    torch::Tensor old_map,     // [H,B,M] in
    torch::Tensor new_act,     // [H,B,M] in 
    torch::Tensor new_map,     // [H,B,M] out
    torch::Tensor load_map     // [H,B,M] out
){
    diff_offload_cuda(old_map, new_act, new_map, load_map);
    return;
}

void diff_offload_fused(
    torch::Tensor old_map,     // [H,B,M] in
    torch::Tensor new_act,     // [H,B,M] in
    torch::Tensor new_map,     // [H,B,M] out
    torch::Tensor load_map     // [H,B,M] out
){
    diff_offload_fused_cuda(old_map, new_act, new_map, load_map);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diff_offload", &diff_offload, "");
    m.def("diff_offload_fused", &diff_offload_fused, "");
}
