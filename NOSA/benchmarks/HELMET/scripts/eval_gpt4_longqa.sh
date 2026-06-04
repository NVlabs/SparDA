# This file is copied/adapted from HELMET (https://github.com/princeton-nlp/HELMET).
# Copyright (c) 2024 Princeton Natural Language Processing.
# SPDX-License-Identifier: MIT

shards=10; for i in $(seq 0 $shards); do python scripts/eval_gpt4_longqa.py --num_shards $shards --shard_idx $i & done
