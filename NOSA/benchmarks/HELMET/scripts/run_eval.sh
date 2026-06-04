# This file is copied/adapted from HELMET (https://github.com/princeton-nlp/HELMET).
# Copyright (c) 2024 Princeton Natural Language Processing.
# SPDX-License-Identifier: MIT

for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    python eval.py --config configs/${task}.yaml
done

this will run the 8k to 64k versions
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    python eval.py --config configs/${task}_short.yaml
done