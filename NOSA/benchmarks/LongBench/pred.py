# This file is copied/adapted from LongBench (https://github.com/THUDM/LongBench).
# Copyright (c) 2023 THU-KEG & Zhipu AI.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import os
import sys
import time
import logging
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse

BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)

from torch_load_compat import torch_load_compat

DATASETS = [
    "gov_report", "triviaqa", "narrativeqa", "qasper", "qmsum", "musique",
    "2wikimqa", "multifieldqa_en", "repobench-p", "hotpotqa",
    "trec", "passage_retrieval_en", "passage_count", "samsum",
]


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model-path', type=str, default=None,
                        help="Override config/model2path.json with a Hugging Face ID or local model path.")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Single dataset to predict. If omitted, runs all datasets sequentially.")
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Limit each selected dataset to the first N samples for smoke tests.")
    parser.add_argument('--max-length', type=int, default=None,
                        help="Max input length for truncation. Falls back to model2maxlen.json.")
    parser.add_argument('--output-dir', type=str, default='.',
                        help="Root directory for pred/ or pred_e/ outputs (default: current directory).")
    parser.add_argument('--progress-file', type=str, default=None,
                        help="File to write progress (current total) for external monitoring.")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--sparda', dest='enable_sparda',
                        action='store_true',
                        help="Enable SparDA sparse attention with trained q_future indexer. "
                             "Use with --indexer_path.")
    parser.add_argument('--infinigen', action='store_true',
                        help="Enable InfiniGen decode (InfLLMv2 sparse prefill + InfiniGen masked decode). "
                             "Only for InfLLMv2/MiniCPM models. Overrides --sparda.")
    parser.add_argument('--indexer_path', type=str, default=None,
                        help="Path to .pt checkpoint with indexer weights for --sparda.")
    parser.add_argument('--resume', action='store_true',
                        help="Resume: skip samples that already have predictions.")
    parser.add_argument('--collect_block_hit_rate', action='store_true',
                        help="Collect block hit-rate stats (sparse models only).")
    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "minicpm" in model_name or "infllmv2" in model_name or "nosa" in model_name:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        prompt = f"<|im_start|>user\n{prompt} /no_think<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "minicpm" in model_name or "infllmv2" in model_name or "nosa" in model_name:
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
        response = response.replace("<think>", "").replace("</think>", "")
        response = response.replace("<thinking>", "").replace("</thinking>", "")
        response = re.sub(r"^(?:Final\s+)?Answer:\s*", "", response)
        response = re.sub(r"\n(?:Extracted|Relevant|Key)\s+Information:.*", "", response, flags=re.DOTALL)
        response = re.sub(r"\n+Passage:.*", "", response, flags=re.DOTALL)
        response = response.strip()
    return response


@torch.no_grad()
def gen(model, input_ids, max_new_tokens, eos_token_ids):
    output = model(input_ids, use_cache=True)
    generated_ids = []
    next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids.append(next_id.item())
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, past_key_values=past_key_values)
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_id.item())
        past_key_values = output.past_key_values
        if next_id.item() in eos_token_ids:
            break
    return generated_ids


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def _is_minicpm_family(model_name):
    return any(k in model_name for k in ("minicpm", "infllmv2", "nosa", "fullattn"))


def _detect_dense(model_name):
    return "dense" in model_name


def load_model_and_tokenizer(path, model_name, device,
                              enable_sparda=False, enable_infinigen=False,
                              indexer_path=None):
    if "nosa" in model_name and not _detect_dense(model_name):
        import sys
        nosa_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "nosa"))
        if nosa_dir not in sys.path:
            sys.path.insert(0, nosa_dir)
        import modeling_llama_nosa as _nosa_mod
        from transformers import AutoConfig

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model_sparse_cfg = dict(getattr(config, 'sparse_config', None) or {})
        checkpoint_sparse_cfg = {}
        checkpoint_state = None
        if indexer_path:
            checkpoint_state = torch_load_compat(indexer_path, map_location='cpu')
            if isinstance(checkpoint_state, dict):
                checkpoint_sparse_cfg = dict(checkpoint_state.get('sparse_config') or {})

        default_sparse = {
            "kernel_size": 32,
            "kernel_stride": 16,
            "init_blocks": 1,
            "block_size": 64,
            "window_size": 1024,  # NOSA local window = 16 blocks
            "topk": 80,           # Effective topk = 80 + 16 = 96
            "use_nope": False,
            "dense_len": 8192,
        }
        sparse_cfg = default_sparse.copy()
        sparse_cfg.update({k: v for k, v in model_sparse_cfg.items() if v is not None})
        if checkpoint_sparse_cfg:
            sparse_cfg.update(checkpoint_sparse_cfg)

        if enable_infinigen:
            if enable_sparda:
                logging.warning("LongBench NOSA: --infinigen overrides --sparda. Ignoring --sparda.")
            sparse_cfg['infinigen_enabled'] = True
            sparse_cfg['infinigen_num_channels'] = 32
            sparse_cfg['dense_len'] = -1
            sparse_cfg['use_q_future_for_topk'] = False
            sparse_cfg['create_indexer'] = False
        elif enable_sparda:
            sparse_cfg['dense_len'] = -1
            sparse_cfg['use_q_future_for_topk'] = True
            sparse_cfg['create_indexer'] = True
        else:
            sparse_cfg['dense_len'] = -1
            sparse_cfg['use_q_future_for_topk'] = False
            sparse_cfg['create_indexer'] = False
        config.sparse_config = sparse_cfg
        model = _nosa_mod.SparseLlamaForCausalLM.from_pretrained(
            path, config=config, device_map=str(device), torch_dtype=torch.bfloat16)
        if enable_infinigen:
            warmup_ids = torch.randint(0, config.vocab_size, (1, 2048), device=model.device)
            model.infinigen_warmup(input_ids=warmup_ids, max_new_tokens=128)
            logging.info(f"LongBench NOSA InfiniGen: sparse_config={config.sparse_config}")
        elif enable_sparda and indexer_path:
            sd = checkpoint_state.get('model_state_dict', checkpoint_state) if isinstance(checkpoint_state, dict) else {}
            iw = {k: v for k, v in sd.items() if 'q_future_proj' in k or 'q_curr_proj' in k}
            if iw:
                model.load_state_dict(iw, strict=False)
                logging.info(f"LongBench NOSA SparDA: loaded {len(iw)} indexer weights")
            logging.info(f"NOSA sparda: sparse_config={config.sparse_config}")
        else:
            logging.info(f"NOSA sparse: sparse_config={config.sparse_config}")
    elif _is_minicpm_family(model_name) and enable_sparda:
        import sys
        from transformers import AutoConfig
        models_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models", "minicpm"))
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        from modeling_minicpm import MiniCPMForCausalLM as MiniCPMFuture

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model_sparse_cfg = dict(getattr(config, 'sparse_config', None) or {})
        default_sparse = {
            "kernel_size": 32, "kernel_stride": 16, "init_blocks": 1,
            "block_size": 64, "window_size": 2048, "topk": 96,
            "use_nope": False, "dense_len": -1,
        }
        config.sparse_config = default_sparse.copy()
        config.sparse_config.update(
            {k: v for k, v in model_sparse_cfg.items() if v is not None})
        config.sparse_config['dense_len'] = -1
        config.sparse_config['topk'] = 96
        block_size = int(config.sparse_config.get("block_size", 64))
        window_size = int(config.sparse_config.get("window_size", 2048))
        local_blocks = (window_size // block_size) if block_size > 0 else 0
        nosi_topk = int(config.sparse_config["topk"])
        config.sparse_config["topk"] = max(1, nosi_topk - local_blocks)
        config.sparse_config['use_q_future_for_topk'] = True
        config._attn_implementation = "flash_attention_2"

        model = MiniCPMFuture.from_pretrained(
            path, config=config, trust_remote_code=True,
            device_map=str(device), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2")
        model.eval()
        model.config.sparse_config = config.sparse_config
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'use_q_future_for_topk'):
                layer.self_attn.use_q_future_for_topk = True
            if hasattr(layer.self_attn, 'dense_len'):
                layer.self_attn.dense_len = -1

        if indexer_path:
            ckpt = torch_load_compat(indexer_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            iw = {k: v for k, v in sd.items()
                  if 'q_future_proj' in k or 'q_curr_proj' in k}
            if iw:
                model.load_state_dict(iw, strict=False)
                logging.info(f"LongBench InfLLMv2 SparDA: loaded {len(iw)} indexer weights")
        logging.info(f"InfLLMv2 SparDA sparse_config: {config.sparse_config}")
    elif _is_minicpm_family(model_name) and _detect_dense(model_name):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if "nosa" in model_name:
            import sys
            from transformers import AutoConfig
            nosa_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "nosa"))
            if nosa_dir not in sys.path:
                sys.path.insert(0, nosa_dir)
            import modeling_llama_nosa as _nosa_mod

            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            sparse_cfg = dict(getattr(config, 'sparse_config', None) or {})
            sparse_cfg['use_q_future_for_topk'] = False
            sparse_cfg['create_indexer'] = False
            sparse_cfg['force_dense_inference'] = True
            config.sparse_config = sparse_cfg
            model = _nosa_mod.SparseLlamaForCausalLM.from_pretrained(
                path, config=config, device_map=str(device), torch_dtype=torch.bfloat16)
            logging.info("NOSA dense mode: modeling_llama_nosa with force_dense_inference=True")
        else:
            import sys
            from transformers import AutoConfig
            models_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "models", "minicpm"))
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            from modeling_minicpm import MiniCPMForCausalLM as MiniCPMFuture

            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            config.sparse_config = {"dense_len": 999999}
            config._attn_implementation = "flash_attention_2"
            model = MiniCPMFuture.from_pretrained(
                path, config=config, trust_remote_code=True,
                device_map=str(device), torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2")
            model.eval()
            model.config.sparse_config = config.sparse_config
            for layer in model.model.layers:
                if hasattr(layer.self_attn, 'dense_len'):
                    layer.self_attn.dense_len = 999999
            logging.info("InfLLMv2 dense mode: modeling_minicpm with dense_len=999999")
    elif enable_infinigen and _is_minicpm_family(model_name):
        if enable_sparda:
            logging.warning("--infinigen overrides --sparda. Ignoring --sparda.")
        import sys
        from transformers import AutoConfig
        models_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models", "minicpm"))
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        from modeling_minicpm_infinigen import MiniCPMForCausalLM_InfiniGen

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model_sparse_cfg = dict(getattr(config, 'sparse_config', None) or {})
        default_sparse = {
            "kernel_size": 32, "kernel_stride": 16, "init_blocks": 1,
            "block_size": 64, "window_size": 2048, "topk": 96,
            "use_nope": False, "dense_len": -1,
        }
        config.sparse_config = default_sparse.copy()
        config.sparse_config.update(
            {k: v for k, v in model_sparse_cfg.items() if v is not None})
        config.sparse_config['dense_len'] = -1
        config.sparse_config['use_q_future_for_topk'] = False
        config.sparse_config['create_indexer'] = False
        block_size = int(config.sparse_config.get("block_size", 64))
        window_size = int(config.sparse_config.get("window_size", 2048))
        local_blocks = (window_size // block_size) if block_size > 0 else 0
        nosi_topk = int(config.sparse_config["topk"])
        config.sparse_config["topk"] = max(1, nosi_topk - local_blocks)
        config._attn_implementation = "flash_attention_2"

        model = MiniCPMForCausalLM_InfiniGen.from_pretrained(
            path, config=config, trust_remote_code=True,
            device_map=str(device), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2")
        model.eval()
        model.config.sparse_config = config.sparse_config
        logging.info(f"InfiniGen: sparse_config={config.sparse_config}")
    elif enable_infinigen:
        raise ValueError("--infinigen is only supported for InfLLMv2/MiniCPM models, "
                         f"but model_name={model_name!r}")
    elif _is_minicpm_family(model_name):
        import sys
        from transformers import AutoConfig
        models_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models", "minicpm"))
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        from modeling_minicpm import MiniCPMForCausalLM as MiniCPMSparse

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model_sparse_cfg = dict(getattr(config, 'sparse_config', None) or {})
        default_sparse = {
            "kernel_size": 32, "kernel_stride": 16, "init_blocks": 1,
            "block_size": 64, "window_size": 2048, "topk": 96,
            "use_nope": False, "dense_len": -1,
        }
        config.sparse_config = default_sparse.copy()
        config.sparse_config.update(
            {k: v for k, v in model_sparse_cfg.items() if v is not None})
        config.sparse_config['dense_len'] = -1
        # Non-SparDA: use q_actual (original Q from QKV) for block selection,
        # not the untrained q_future_proj which would have random weights.
        config.sparse_config['use_q_future_for_topk'] = False
        config.sparse_config['create_indexer'] = False
        block_size = int(config.sparse_config.get("block_size", 64))
        window_size = int(config.sparse_config.get("window_size", 2048))
        local_blocks = (window_size // block_size) if block_size > 0 else 0
        nosi_topk = int(config.sparse_config["topk"])
        config.sparse_config["topk"] = max(1, nosi_topk - local_blocks)
        config._attn_implementation = "flash_attention_2"

        model = MiniCPMSparse.from_pretrained(
            path, config=config, trust_remote_code=True,
            device_map=str(device), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2")
        model.eval()
        model.config.sparse_config = config.sparse_config
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'dense_len'):
                layer.self_attn.dense_len = -1
        logging.info(f"InfLLMv2 sparse: sparse_config={config.sparse_config}")
    elif "shadowkv" in model_name:
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        shadowkv_root = os.environ.get("SHADOWKV_ROOT")
        if not shadowkv_root:
            raise ValueError("ShadowKV is not bundled. Set SHADOWKV_ROOT to an external ShadowKV checkout.")
        sys.path.insert(0, shadowkv_root)
        from models import Llama

        if "minf" in model_name:
            print("use minference")

        model = Llama(
            model_name=path,
            sparse_budget=3072,
            attn_mode='shadowkv',
            rank=40,
            device=str(device),
            chunk_size=64,
            minference=True if "minf" in model_name else False,
        )
    elif "arkvale" in model_name:
        try:
            from arkvale import adapter
        except Exception:
            raise ValueError("no arkvale in environment")

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
        adapter.enable_arkvale(
            model,
            dtype=torch.float16,
            device=device,
            page_size=32,
            page_budgets=128,
            page_topks=65,
            n_sink_pages=2,
            n_win_pages=32,
            n_max_bytes=40 * (1 << 30),
            n_max_cpu_bytes=60 * (1 << 30),
        )
    elif "infllmv1" in model_name:
        raise ValueError("InfLLMv1 baseline support is not bundled in this release.")
    elif "llama2" in model_name:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    else:
        raise ValueError("Unknown")

    model = model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_dataset(model, tokenizer, model_name, dataset_name, data,
                    max_length, max_gen, prompt_format, out_path,
                    progress_file=None, resume=False, collector=None):
    """Run prediction on all samples for a single dataset."""
    # Resume: skip already-predicted samples
    skip_count = 0
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            skip_count = sum(1 for line in f if line.strip())
        if skip_count >= len(data):
            print(f"  [{dataset_name}] all {skip_count} samples already predicted, skipping")
            return
        if skip_count > 0:
            print(f"  [{dataset_name}] resuming from sample {skip_count}/{len(data)}")
            data = data[skip_count:]

    device = next(model.parameters()).device
    total = len(data)
    data_iter = data if progress_file else tqdm(data, desc=dataset_name)

    for i, json_obj in enumerate(data_iter):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                     tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset_name in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]
        if collector is not None:
            collector.begin_sample()

        if "shadowkv" in model_name:
            q_len = input.input_ids.shape[-1]
            if q_len <= 4 * 1024:
                model.set_mode("full")
            else:
                model.set_mode("shadowkv")
            pred = model.generate(input.input_ids, gen_len=max_gen, temperature=0, top_k=-1, top_p=1.0)[0]
        elif "arkvale" in model_name:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=0.0,
                use_cache=False,
            )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=0.0,
            )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        if collector is not None:
            collector.end_sample()

        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"],
                        "all_classes": json_obj["all_classes"],
                        "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
        if progress_file:
            with open(progress_file, 'w') as pf:
                pf.write(f"{i + 1} {total}\n")


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model_name = args.model

    if args.max_length is not None:
        max_length = args.max_length
    else:
        model2maxlen = json.load(open("config/model2maxlen.json", "r"))
        max_length = model2maxlen[model_name]

    datasets = [args.dataset] if args.dataset else DATASETS

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[pred] model={model_name}, max_length={max_length}, "
          f"datasets={datasets}, device={device}")

    model_path = args.model_path or model2path[model_name]
    model, tokenizer = load_model_and_tokenizer(
        model_path, model_name, device,
        enable_sparda=args.enable_sparda,
        enable_infinigen=getattr(args, 'infinigen', False),
        indexer_path=args.indexer_path)

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    pred_root = os.path.join(args.output_dir, "pred_e" if args.e else "pred")
    os.makedirs(f"{pred_root}/{model_name}", exist_ok=True)

    # Set up block hit-rate collector if requested
    collector = None
    if args.collect_block_hit_rate:
        try:
            import sys as _sys
            _bench_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if _bench_dir not in _sys.path:
                _sys.path.insert(0, _bench_dir)
            from block_hit_rate import BlockHitRateCollector
            collector = BlockHitRateCollector(model)
            if not collector.available:
                print("[pred] WARNING: block hit-rate not available for this model (no sparse_config)")
                collector = None
            else:
                collector.install()
                print("[pred] Block hit-rate collection enabled")
        except ImportError as e:
            print(f"[pred] WARNING: cannot import block_hit_rate module: {e}")

    for dataset_name in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset_name}_e", split='test')
        else:
            data = load_dataset('json', data_files=f"./data/{dataset_name}.jsonl")['train']

        out_path = f"{pred_root}/{model_name}/{dataset_name}.jsonl"
        prompt_format = dataset2prompt[dataset_name]
        max_gen = dataset2maxlen[dataset_name]
        data_all = list(data)
        if args.max_samples is not None:
            data_all = data_all[:max(0, args.max_samples)]

        print(f"  [{dataset_name}] {len(data_all)} samples")
        t0 = time.time()
        predict_dataset(model, tokenizer, model_name, dataset_name, data_all,
                        max_length, max_gen, prompt_format, out_path,
                        progress_file=args.progress_file,
                        resume=args.resume, collector=collector)
        elapsed = time.time() - t0
        print(f"  [{dataset_name}] done in {elapsed:.1f}s")

        # Save per-dataset block hit-rate stats and reset for next dataset
        if collector is not None:
            stats = collector.get_stats()
            if stats and stats.get("total_blocks", 0) > 0:
                stats_path = f"{pred_root}/{model_name}/{dataset_name}_block_stats.json"
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                print(f"  [{dataset_name}] block hit-rate: {stats['hit_rate'] * 100:.1f}%  "
                      f"(saved to {stats_path})")
            # Reset tracker for next dataset
            collector._tracker.reset()

    if collector is not None:
        collector.uninstall()
