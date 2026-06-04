// This file is copied/adapted from FlashAttention (https://github.com/Dao-AILab/flash-attention).
// Copyright (c) 2022-2024 Tri Dao and contributors.
// SPDX-License-Identifier: BSD-3-Clause

// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                           \
    [&] {                                                                            \
        if (COND) {                                                                  \
            constexpr bool CONST_NAME = true;                                        \
            return __VA_ARGS__();                                                    \
        } else {                                                                     \
            constexpr bool CONST_NAME = false;                                       \
            return __VA_ARGS__();                                                    \
        }                                                                            \
    }()
