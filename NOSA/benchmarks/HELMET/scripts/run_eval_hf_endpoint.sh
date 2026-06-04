# This file is copied/adapted from HELMET (https://github.com/princeton-nlp/HELMET).
# Copyright (c) 2024 Princeton Natural Language Processing.
# SPDX-License-Identifier: MIT


LLM_ENDPOINT="https://${hf_inference_point_url}/v1" # fill in your endpoint url
API_KEY=$HF_TOKEN

python eval.py --config configs/recall_demo.yaml --endpoint_url $LLM_ENDPOINT --api_key $API_KEY