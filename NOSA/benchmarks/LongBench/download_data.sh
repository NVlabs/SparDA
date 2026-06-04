#!/bin/bash
# This file is copied/adapted from LongBench (https://github.com/THUDM/LongBench).
# Copyright (c) 2023 THU-KEG & Zhipu AI.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0


URL="https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"
OUTPUT="data.zip"

curl -L "$URL" -o "$OUTPUT"

if [ $? -eq 0 ]; then
    echo "Download success"
    unzip $OUTPUT
    echo "Unzip success"
else
    echo "Download failed"
    exit 1
fi
