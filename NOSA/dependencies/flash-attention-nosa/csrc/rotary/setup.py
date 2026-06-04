# This file is copied/adapted from FlashAttention (https://github.com/Dao-AILab/flash-attention).
# It includes code adapted from NVIDIA Apex (https://github.com/NVIDIA/apex).
# Copyright (c) 2022-2024 Tri Dao and contributors.
# Copyright (c) NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
import re
from packaging.version import parse, Version

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
from setuptools import setup, find_packages
import subprocess

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


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
            warnings.warn(f"Skipping unsupported CUDA arch entry: {arch}")
            continue
        arch_code, emit_ptx = normalized
        if not _cuda_supports_arch(arch_code, cuda_version):
            warnings.warn(
                f"Skipping arch {arch_code}: nvcc {cuda_version} is too old for it."
            )
            continue
        if arch_code not in seen:
            cc_flag.extend(["-gencode", f"arch=compute_{arch_code},code=sm_{arch_code}"])
            seen.add(arch_code)
        if emit_ptx:
            cc_flag.extend(["-gencode", f"arch=compute_{arch_code},code=compute_{arch_code}"])
    return cc_flag


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("12.8"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;10.0"
        elif bare_metal_version >= Version("11.8"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        elif bare_metal_version >= Version("11.1"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        elif bare_metal_version == Version("11.0"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

cmdclass = {}
ext_modules = []

raise_if_cuda_home_none("rotary_emb")
# Check, if CUDA11 is installed for compute capability 8.0
cc_flag = []
_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_metal_version < Version("11.0"):
    raise RuntimeError("rotary_emb is only supported on CUDA 11 and above")
cc_flag.extend(get_cuda_arch_flags(bare_metal_version, ["7.0", "8.0"]))

ext_modules.append(
    CUDAExtension(
        'rotary_emb', [
            'rotary.cpp',
            'rotary_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-march=native', '-funroll-loops'],
                            'nvcc': append_nvcc_threads([
                                '-O3', '--use_fast_math', '--expt-extended-lambda'
                            ] + cc_flag)
                           }
    )
)

setup(
    name="rotary_emb",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)
