# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOSA (Native and Offloadable Sparse Attention) model
#
# modeling_llama_nosa.py uses bare `from cis_pooling import ...` (not relative),
# so we must ensure this directory is on sys.path before the relative import.
import os, sys
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from .modeling_llama_nosa import (
    SparseLlamaForCausalLM,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaSdpaAttention,
)

__all__ = [
    "SparseLlamaForCausalLM",
    "LlamaModel",
    "LlamaDecoderLayer",
    "LlamaSdpaAttention",
]
