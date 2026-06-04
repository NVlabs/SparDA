# This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
# Copyright (c) 2026 THUNLP.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Lazy loader for the C++ multithreaded CPU gather extension.

Raises RuntimeError if the C++ extension fails to build (requires a C++
compiler with OpenMP support).

Usage:
    from .flash_cache_engine.cpu_gather_loader import get_cpu_gather

    mod = get_cpu_gather()  # returns compiled extension module
    gathered = mod.cpu_gather(src_flat, indices)
    mod.cpu_gather_kv_blocks(k_src, v_src, miss_b, miss_h, ...)
"""

import os

_cpu_gather_mod = None
_cpu_gather_err = None
_loaded = False


def get_cpu_gather():
    """Return the compiled cpu_gather extension module. Raises on build failure."""
    global _cpu_gather_mod, _cpu_gather_err, _loaded
    if _loaded:
        if _cpu_gather_err is not None:
            raise _cpu_gather_err
        return _cpu_gather_mod

    _loaded = True
    try:
        from torch.utils.cpp_extension import load
        this_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(this_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        mod = load(
            name="cpu_gather",
            sources=[os.path.join(this_dir, "cpu_gather.cpp")],
            build_directory=build_dir,
            extra_cflags=["-O3", "-fopenmp", "-march=native"],
            extra_ldflags=["-lgomp"],
            verbose=False,
        )
        _cpu_gather_mod = mod
    except Exception as e:
        _cpu_gather_err = RuntimeError(
            f"Failed to build cpu_gather C++ extension: {e}\n"
            f"Ensure a C++ compiler with OpenMP support is available."
        )
        raise _cpu_gather_err

    return _cpu_gather_mod
