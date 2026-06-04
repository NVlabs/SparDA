# This file is copied/adapted from FlashAttention (https://github.com/Dao-AILab/flash-attention).
# Copyright (c) 2022-2024 Tri Dao and contributors.
# SPDX-License-Identifier: BSD-3-Clause

CUDA_VISIBLE_DEVICES=1 nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --sample=none \
  --cpuctxsw=none \
  --output=trace_nosa_pooling \
  python tests/test_nosa_decode.py
