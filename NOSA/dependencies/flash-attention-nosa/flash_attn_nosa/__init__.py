# This file is copied/adapted from FlashAttention (https://github.com/Dao-AILab/flash-attention).
# Copyright (c) 2022-2024 Tri Dao and contributors.
# SPDX-License-Identifier: BSD-3-Clause

__version__ = "2.6.3"

from flash_attn_nosa.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
