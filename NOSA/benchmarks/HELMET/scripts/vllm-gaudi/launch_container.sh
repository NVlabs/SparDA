# This file is copied/adapted from HELMET (https://github.com/princeton-nlp/HELMET).
# Copyright (c) 2024 Princeton Natural Language Processing.
# SPDX-License-Identifier: MIT

export host_ip=$(hostname -I | awk '{print $1}')
export LLM_ENDPOINT_PORT=8010
export HF_TOKEN=${HF_TOKEN}
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}/v1"
export DATA_PATH="~/.cache/huggingface"
export MAX_MODEL_LEN=131072

# single node 
# export LLM_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
# export NUM_CARDS=1

# multiple nodes 
export LLM_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
export NUM_CARDS=8

docker compose up