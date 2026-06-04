// This file is copied/adapted from FlashAttention (https://github.com/Dao-AILab/flash-attention).
// Copyright (c) 2022-2024 Tri Dao and contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <ATen/ATen.h>

#define DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, ...)                   \
switch(TYPE)                                                        \
{                                                                   \
case at::ScalarType::Half:                                          \
    {                                                               \
using scalar_t = at::Half;                                          \
__VA_ARGS__;                                                        \
break;                                                              \
    }                                                               \
case at::ScalarType::BFloat16:                                      \
    {                                                               \
using scalar_t = at::BFloat16;                                      \
__VA_ARGS__;                                                        \
break;                                                              \
    }                                                               \
default:                                                            \
    AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");	\
}
