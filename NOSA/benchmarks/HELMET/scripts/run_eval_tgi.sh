# This file is copied/adapted from HELMET (https://github.com/princeton-nlp/HELMET).
# Copyright (c) 2024 Princeton Natural Language Processing.
# SPDX-License-Identifier: MIT

export host_ip=$(hostname -I | awk '{print $1}')
export LLM_ENDPOINT_PORT=8085 # change this to the port you want to use
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}/v1"

python eval.py --config configs/recall_demo.yaml --endpoint_url $LLM_ENDPOINT
