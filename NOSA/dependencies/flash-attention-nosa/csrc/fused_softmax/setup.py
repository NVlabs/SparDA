# This file is copied/adapted from FlashAttention (https://github.com/Dao-AILab/flash-attention).
# It includes code copied from NVIDIA Apex (https://github.com/NVIDIA/apex).
# Copyright (c) 2022-2024 Tri Dao and contributors.
# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

# Copied from https://github.com/NVIDIA/apex/tree/master/csrc/megatron
# We add the case where seqlen = 4k and seqlen = 8k
import os
import re
import subprocess

import torch
from packaging.version import parse, Version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        nvcc_threads = os.getenv("NVCC_THREADS") or "4"
        return nvcc_extra_args + ["--threads", nvcc_threads]
    return nvcc_extra_args


def _normalize_cuda_arch(arch: str):
    arch = arch.strip()
    if not arch:
        return None
    emit_ptx = arch.endswith("+PTX")
    if emit_ptx:
        arch = arch[:-4]
    arch_aliases = {
        "7.0": "70",
        "7.5": "75",
        "8.0": "80",
        "8.6": "86",
        "8.9": "89",
        "9.0": "90",
        "10.0": "100",
    }
    arch_code = arch_aliases.get(arch, arch.replace(".", ""))
    if not arch_code.isdigit():
        return None
    return arch_code, emit_ptx


def _cuda_supports_arch(arch_code: str, cuda_version: Version) -> bool:
    arch_num = int(arch_code)
    if arch_num >= 100:
        return cuda_version >= Version("12.8")
    if arch_num >= 90:
        return cuda_version >= Version("11.8")
    return cuda_version >= Version("11.0")


def get_cuda_arch_flags(cuda_version: Version, default_arches):
    env_arch_list = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
    if env_arch_list:
        requested_arches = re.split(r"[; ]+", env_arch_list)
    else:
        requested_arches = list(default_arches)
        if cuda_version >= Version("11.8") and "9.0" not in requested_arches:
            requested_arches.append("9.0")
        if cuda_version >= Version("12.8") and "10.0" not in requested_arches:
            requested_arches.append("10.0")

    cc_flag = []
    seen = set()
    for arch in requested_arches:
        normalized = _normalize_cuda_arch(arch)
        if normalized is None:
            continue
        arch_code, emit_ptx = normalized
        if not _cuda_supports_arch(arch_code, cuda_version):
            continue
        if arch_code not in seen:
            cc_flag.extend(["-gencode", f"arch=compute_{arch_code},code=sm_{arch_code}"])
            seen.add(arch_code)
        if emit_ptx:
            cc_flag.extend(["-gencode", f"arch=compute_{arch_code},code=compute_{arch_code}"])
    return cc_flag


_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
cc_flag = get_cuda_arch_flags(bare_metal_version, ["7.0", "8.0"])

setup(
    name='fused_softmax_lib',
    ext_modules=[
        CUDAExtension(
            name='fused_softmax_lib',
            sources=['fused_softmax.cpp', 'scaled_masked_softmax_cuda.cu', 'scaled_upper_triang_masked_softmax_cuda.cu'],
            extra_compile_args={
                               'cxx': ['-O3',],
                               'nvcc': append_nvcc_threads(['-O3', '--use_fast_math'] + cc_flag)
                               }
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
