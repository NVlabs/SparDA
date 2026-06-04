// This file is copied/adapted from FlashAttention (https://github.com/Dao-AILab/flash-attention).
// Copyright (c) 2022-2024 Tri Dao and contributors.
// SPDX-License-Identifier: BSD-3-Clause

// This is purely so that it works with torch 2.1. For torch 2.2+ we can include ATen/cuda/PhiloxUtils.cuh

#pragma once
#include <ATen/cuda/detail/UnpackRaw.cuh>
