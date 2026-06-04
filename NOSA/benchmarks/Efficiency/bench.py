#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified efficiency benchmark for NOSA and MiniCPM models.

Usage:
    # NOSA with offloading (default)
    python bench.py --model-path openbmb/NOSA-8B

    # NOSA without offloading (GPU-only)
    python bench.py --model-path openbmb/NOSA-8B --no-offload --batch-size 4 --seq-len 16K

    # NOSA with SparDA indexer
    python bench.py --model-path openbmb/NOSA-8B --sparda --indexer-path /path/to/indexer_weights

    # NOSA with HuggingFace backend
    python bench.py --model-path openbmb/NOSA-8B --backend hf --batch-size 4 --seq-len 16K

    # MiniCPM with offloading
    python bench.py --model-path openbmb/MiniCPM4.1-8B

    # MiniCPM with SparDA indexer
    python bench.py --model-path openbmb/MiniCPM4.1-8B --sparda --indexer-path /path/to/indexer_weights

    # MiniCPM with SparDA indexer but no persistent decode prefetch
    python bench.py --model-path openbmb/MiniCPM4.1-8B --sparda --sparda-no-prefetch --indexer-path /path/to/indexer_weights

    # MiniCPM dense baseline
    python bench.py --model-path openbmb/MiniCPM4.1-8B --dense

    # MiniCPM with InfiniGen
    python bench.py --model-path openbmb/MiniCPM4.1-8B --infinigen

    # Custom settings
    python bench.py --model-path openbmb/NOSA-8B --batch-size 8 --seq-len 64K --max-new-tokens 8 --test-n 6

Seq-len accepts: 16K, 32K, 64K, 128K, or raw integers (e.g., 16384).
"""

import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path

# Ensure local nosi package is importable (prioritize repo over site-packages).
# nosi package lives at NOSA/nosi/nosi/, so NOSA/nosi/ must be on sys.path.
_NOSI_ROOT = str(Path(__file__).resolve().parents[2] / "nosi")
if _NOSI_ROOT not in sys.path:
    sys.path.insert(0, _NOSI_ROOT)

_BENCHMARKS_ROOT = str(Path(__file__).resolve().parents[1])
if _BENCHMARKS_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARKS_ROOT)

# ─── Suppress noisy warnings before any imports ────────────────────────────
warnings.filterwarnings("ignore", message=".*rope_scaling.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*new version of the following files.*")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*malicious code.*")

import builtins
_original_print = builtins.print

def _quiet_print(*args, **kwargs):
    if args and "Use InfLLMv2" in str(args[0]):
        return
    _original_print(*args, **kwargs)

builtins.print = _quiet_print

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from bench_utils import parse_seq_len, prepare_input_batches
from torch_load_compat import torch_load_compat


# ─── Helpers ───────────────────────────────────────────────────────────────

def _print_input_fingerprint(tokenizer, input_batches, tag=""):
    """Print a compact fingerprint of each input batch for cross-backend verification."""
    import hashlib
    _original_print(f"[Input fingerprint ({tag})]  {len(input_batches)} batches")
    for bi, batch in enumerate(input_batches):
        # Hash the raw bytes of the tensor for exact comparison
        raw = batch.cpu().numpy().tobytes()
        h = hashlib.sha256(raw).hexdigest()[:16]
        B, L = batch.shape
        first5 = batch[0, :5].tolist()
        last5 = batch[0, -5:].tolist()
        _original_print(
            f"  batch {bi}: shape=({B},{L})  sha256={h}  "
            f"sample0_first5={first5}  sample0_last5={last5}"
        )


def _print_generated(tokenizer, gen_ids, tag=""):
    """Print generated tokens for each sample in a batch.

    Args:
        tokenizer: HuggingFace tokenizer.
        gen_ids: Tensor of shape (B, T) containing generated token IDs.
        tag: Optional label for the print header.
    """
    B = gen_ids.shape[0]
    header = f"[Generated tokens ({tag})]" if tag else "[Generated tokens]"
    _original_print(header)
    for b in range(B):
        ids = gen_ids[b].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        tokens = [tokenizer.decode([t], skip_special_tokens=False) for t in ids]
        _original_print(f"  sample {b}: ids={ids}")
        _original_print(f"           tokens={tokens}")
        _original_print(f"           text={text!r}")


NATIVE_MAX = {"nosa": 32768, "minicpm": 65536}


def _compute_benchmark_metrics(total_elapsed_s, decode_tok_per_s, batch_size, input_len, max_new_tokens):
    decode_tokens = batch_size * max_new_tokens
    prefill_tokens = batch_size * input_len
    decode_time = decode_tokens / decode_tok_per_s if decode_tok_per_s > 0 else 0.0
    prefill_time = max(0.0, total_elapsed_s - decode_time)
    prefill_tok_per_s = prefill_tokens / prefill_time if prefill_time > 0 else 0.0
    return {
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "prefill_tok_per_s": prefill_tok_per_s,
        "decode_tok_per_s": decode_tok_per_s,
    }


def _compute_split_benchmark_metrics(prefill_time_s, decode_time_s, batch_size, input_len, max_new_tokens):
    decode_tokens = batch_size * max_new_tokens
    prefill_tokens = batch_size * input_len
    prefill_tok_per_s = prefill_tokens / prefill_time_s if prefill_time_s > 0 else 0.0
    decode_tok_per_s = decode_tokens / decode_time_s if decode_time_s > 0 else 0.0
    return {
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_time": prefill_time_s,
        "decode_time": decode_time_s,
        "prefill_tok_per_s": prefill_tok_per_s,
        "decode_tok_per_s": decode_tok_per_s,
    }


def _compute_benchmark_metrics_from_model(model, total_elapsed_s, decode_tok_per_s, batch_size, input_len, max_new_tokens):
    prefill_time = getattr(model, "_last_prefill_time_s", None)
    decode_time = getattr(model, "_last_decode_time_s", None)
    if prefill_time is not None and decode_time is not None and prefill_time >= 0 and decode_time > 0:
        return _compute_split_benchmark_metrics(
            prefill_time,
            decode_time,
            batch_size,
            input_len,
            max_new_tokens,
        )
    return _compute_benchmark_metrics(
        total_elapsed_s,
        decode_tok_per_s,
        batch_size,
        input_len,
        max_new_tokens,
    )


def _print_iter_metrics(iter_idx, metrics, warmup=False):
    suffix = " (warmup, excluded)" if warmup else ""
    print(
        f"  iter {iter_idx}: prefill = {metrics['prefill_tok_per_s']:.2f} tok/s, "
        f"decode = {metrics['decode_tok_per_s']:.2f} tok/s{suffix}"
    )


def _print_benchmark_summary(prefill_tokens, decode_tokens, total_prefill_time, total_decode_time, test_n):
    avg_prefill_time = total_prefill_time / test_n if test_n > 0 else 0.0
    avg_decode_time = total_decode_time / test_n if test_n > 0 else 0.0
    prefill_tok_per_s = (prefill_tokens * test_n / total_prefill_time) if total_prefill_time > 0 else 0.0
    decode_tok_per_s = (decode_tokens * test_n / total_decode_time) if total_decode_time > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f"  Prefill {prefill_tokens} tokens in {avg_prefill_time:.3f} s")
    print(f"  Prefill throughput: {prefill_tok_per_s:.2f} tok/s (avg over {test_n} runs)")
    print(f"  Decode {decode_tokens} tokens in {avg_decode_time:.3f} s")
    print(f"  {decode_tok_per_s:.2f} tokens/s (avg over {test_n} runs)")
    print(f"{'=' * 50}")


def detect_method(model_path: str) -> str:
    combined = model_path.lower()
    if "nosa" in combined:
        return "nosa"
    if "minicpm" in combined:
        return "minicpm"
    return "other"


# ─── Backend: NOSI (nosa / minicpm) ───────────────────────────────────────

def run_nosi(args):
    method = detect_method(args.model_path)
    if args.infinigen and method == "minicpm":
        from nosi import InfLLMv2Llama as Llama
    elif args.infinigen and method == "nosa":
        from nosi import NOSALlama as Llama
    elif method == "nosa":
        from nosi import NOSALlama as Llama
    elif method == "minicpm":
        from nosi import InfLLMv2Llama as Llama
    else:
        raise ValueError(f"NOSI backend: cannot detect method from model path {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    offload = not args.no_offload

    if args.infinigen and method == "minicpm":
        model = Llama(
            model_name=args.model_path,
            device="cuda",
            offload=offload,
            offload_layer0=args.offload_layer0,
            long_context=args.long_context,
            infinigen=True,
        )
    elif args.infinigen and method == "nosa":
        model = Llama(
            model_name=args.model_path,
            device="cuda",
            offload=offload,
            offload_layer0=args.offload_layer0,
            long_context=args.long_context,
            yarn=getattr(args, 'yarn', False),
            infinigen=True,
        )
    else:
        model_kwargs = dict(
            model_name=args.model_path,
            device="cuda",
            offload=offload,
            offload_layer0=args.offload_layer0,
            long_context=args.long_context,
        )
        if method == "nosa":
            model_kwargs["yarn"] = getattr(args, 'yarn', False)
        if args.sparda:
            model_kwargs["decoupled"] = True
            model_kwargs["decoupled_no_prefetch"] = args.sparda_no_prefetch
        if args.indexer_path:
            model_kwargs["indexer_path"] = args.indexer_path
        model = Llama(**model_kwargs)
        if args.dense:
            model.force_dense_inference = True
            sparse_cfg = dict(getattr(model.config, "sparse_config", None) or {})
            sparse_cfg["force_dense_inference"] = True
            model.config.sparse_config = sparse_cfg
    setattr(model, "_collect_prefill_decode_timings", True)

    builtins.print = _original_print

    B = args.batch_size
    L = args.seq_len
    max_new_tokens = args.max_new_tokens
    test_n = args.test_n

    lc_str = "128K" if args.long_context else "off"
    mode = "dense" if args.dense else ("infinigen" if args.infinigen else method)
    print(f"\n[Config] method={mode}, backend=nosi, offload={offload}, "
          f"sparda={args.sparda}, sparda_no_prefetch={args.sparda_no_prefetch}, "
          f"layer0_cpu_offload={args.offload_layer0}, "
          f"long_context={lc_str}")
    if args.indexer_path:
        print(f"[Indexer] {args.indexer_path}")
    print(f"[Setup]  batch={B}, input_len={L}, max_new_tokens={max_new_tokens}, "
          f"test_n={test_n}, model={args.model_path}")

    def test_time(input_ids):
        torch.cuda.synchronize()
        total_beg = time.time()
        gen_ids, decode_tok_per_s = model.batch_generate_benchmark(
            input_ids, max_new_tokens=max_new_tokens + 2
        )
        torch.cuda.synchronize()
        total_end = time.time()
        if args.print_output and gen_ids is not None:
            _print_generated(tokenizer, gen_ids, tag="nosi")
        return _compute_benchmark_metrics_from_model(
            model,
            total_end - total_beg,
            decode_tok_per_s,
            input_ids.shape[0],
            input_ids.shape[1],
            max_new_tokens,
        )

    input_batches = prepare_input_batches(
        args.dataset_name, args.dataset_split, tokenizer,
        B, L, 1 + test_n,
        model_path=args.model_path,
    )
    if args.print_output:
        _print_input_fingerprint(tokenizer, input_batches, tag="nosi")

    # ── Warmup ────────────────────────────────────────────────────────
    metrics = test_time(input_batches[0])
    gc.collect()
    torch.cuda.empty_cache()
    _print_iter_metrics(0, metrics, warmup=True)

    # ── Timed iterations ──────────────────────────────────────────────
    total_prefill_time = 0.0
    total_decode_time = 0.0
    for r in range(test_n):
        metrics = test_time(input_batches[1 + r])
        gc.collect()
        torch.cuda.empty_cache()
        total_prefill_time += metrics["prefill_time"]
        total_decode_time += metrics["decode_time"]
        _print_iter_metrics(r + 1, metrics)

    _print_benchmark_summary(B * L, B * max_new_tokens, total_prefill_time, total_decode_time, test_n)


# ─── Backend: HuggingFace (nosa-hf / minicpm-hf / dense) ─────────────────

def run_hf(args):
    import logging
    from transformers import AutoConfig
    method = detect_method(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    probed_cfg = None
    is_minicpm_model = False
    if method == "minicpm":
        try:
            probed_cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
            is_minicpm_model = (getattr(probed_cfg, "model_type", "") == "minicpm")
        except Exception:
            is_minicpm_model = ("minicpm" in str(args.model_path).lower())
        if args.indexer_path and not args.sparda:
            print("[SparDA] --indexer-path provided; enabling SparDA mode for HF MiniCPM.")
            args.sparda = True

    use_minicpm_infinigen = (
        method == "minicpm" and
        getattr(args, 'infinigen', False) and
        not args.dense
    )

    use_minicpm_loader = (
        method == "minicpm" and
        not args.dense and
        not use_minicpm_infinigen and
        (args.sparda or is_minicpm_model)
    )

    if use_minicpm_infinigen:
        # HF backend InfiniGen for MiniCPM
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "minicpm")))
        from modeling_minicpm_infinigen import MiniCPMForCausalLM_InfiniGen
        from modeling_minicpm import apply_minicpm41_128k_longrope

        config = probed_cfg if probed_cfg is not None else AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if args.long_context:
            apply_minicpm41_128k_longrope(config)
        config.sparse_config = config.sparse_config if hasattr(config, 'sparse_config') and config.sparse_config else {}
        config.sparse_config['use_q_future_for_topk'] = False
        config.sparse_config['create_indexer'] = False
        config._attn_implementation = "flash_attention_2"

        model = MiniCPMForCausalLM_InfiniGen.from_pretrained(
            args.model_path, config=config, trust_remote_code=True,
            device_map="auto", torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2")
        print("[HF Loader] MiniCPM InfiniGen model loaded")
    elif args.dense and method != "nosa":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
    elif use_minicpm_loader:
        # Explicitly load modeling_minicpm.py for SparDA InfLLMv2
        mode_note = "sparda mode" if args.sparda else "auto MiniCPM mode"
        print(f"[HF Loader] using MiniCPM loader ({mode_note})")
        logging.info(f"Loading MiniCPM infllmv2 from modeling_minicpm.py ({mode_note})")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "minicpm")))
        from modeling_minicpm import (
            MiniCPMForCausalLM as MiniCPMForCausalLM_Future,
            apply_minicpm41_128k_longrope,
        )
        from transformers import GenerationConfig

        config = probed_cfg if probed_cfg is not None else AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if args.long_context:
            apply_minicpm41_128k_longrope(config)
            print("[LongContext] infllmv2: applied official MiniCPM4.1 128K LongRoPE factors.")

        # Default sparse_config; checkpoint values override when present
        default_sparse_config = {
            "kernel_size": 32,
            "kernel_stride": 16,
            "init_blocks": 1,
            "block_size": 64,
            "window_size": 2048,  # local window blocks = 32
            "topk": 96,
            "use_nope": False,
            "dense_len": -1,  # Force pure sparse
        }
        config.sparse_config = default_sparse_config.copy()
        # Align MiniCPM-future topk semantics with nosi:
        #   MiniCPM-future effective_topk = sparse_config["topk"] + local_blocks
        #   nosi effective_topk           = sparse_config["topk"]
        # So set HF base topk = max(1, nosi_topk - local_blocks).
        block_size = int(config.sparse_config.get("block_size", 64))
        window_size = int(config.sparse_config.get("window_size", 2048))
        local_blocks = (window_size // block_size) if block_size > 0 else 0
        nosi_topk = int(config.sparse_config.get("topk", 64))
        hf_base_topk = max(1, nosi_topk - local_blocks)
        if hf_base_topk != nosi_topk:
            print(
                f"[HF Loader] topk semantics aligned: "
                f"nosi_topk={nosi_topk}, local_blocks={local_blocks}, "
                f"hf_base_topk={hf_base_topk}, hf_effective_topk={hf_base_topk + local_blocks}"
            )
        config.sparse_config["topk"] = hf_base_topk
        config.sparse_config['use_q_future_for_topk'] = bool(args.sparda)
        config._attn_implementation = "flash_attention_2"

        logging.info(f"sparse_config: {config.sparse_config}")

        model = MiniCPMForCausalLM_Future.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Ensure sparse_config is applied to attention layers
        model.config.sparse_config = config.sparse_config
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'use_q_future_for_topk'):
                layer.self_attn.use_q_future_for_topk = bool(args.sparda)
            if hasattr(layer.self_attn, 'dense_len'):
                layer.self_attn.dense_len = -1

        # Load indexer weights if provided
        if args.indexer_path:
            checkpoint = torch_load_compat(args.indexer_path, map_location='cpu')
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model_state_dict'), dict):
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and isinstance(checkpoint.get('state_dict'), dict):
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                # Fallback: root dict itself may be a state_dict (+ optional metadata keys)
                tensor_items = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
                state_dict = tensor_items
            else:
                state_dict = {}
            is_full = bool(checkpoint.get('full_model', False)) if isinstance(checkpoint, dict) else False

            def _map_state_dict_to_model_keys(src_state_dict, model_state_keys):
                """Map checkpoint keys to target model keys by stripping common wrappers."""
                prefixes = (
                    "module.", "model.", "model.model.",
                    "module.model.", "module.model.model.",
                    "_orig_mod.", "base_model.model.",
                )
                mapped = {}
                for src_k, v in src_state_dict.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    variants = [src_k]
                    # Repeatedly strip known prefixes (e.g., module.model.)
                    changed = True
                    while changed:
                        changed = False
                        for p in prefixes:
                            cur = variants[-1]
                            if cur.startswith(p):
                                nxt = cur[len(p):]
                                variants.append(nxt)
                                changed = True
                                break
                    candidates = []
                    for k in variants:
                        candidates.append(k)
                        if not k.startswith("model."):
                            candidates.append("model." + k)
                        if k.startswith("model."):
                            candidates.append(k[len("model."):])
                    tgt_k = next((c for c in candidates if c in model_state_keys), None)
                    if tgt_k is not None:
                        mapped[tgt_k] = v
                return mapped

            if is_full:
                mapped_full = _map_state_dict_to_model_keys(state_dict, set(model.state_dict().keys()))
                model.load_state_dict(mapped_full, strict=False)
                logging.info(f"Loaded FULL model weights ({len(state_dict)} tensors)")
            else:
                indexer_weights = {k: v for k, v in state_dict.items() if 'q_future_proj' in k or 'q_curr_proj' in k}
                if indexer_weights:
                    model_state_keys = set(model.state_dict().keys())
                    mapped_indexer = _map_state_dict_to_model_keys(indexer_weights, model_state_keys)
                    expected_indexer = sorted(
                        k for k in model_state_keys
                        if ('q_future_proj' in k or 'q_curr_proj' in k)
                    )
                    missing_indexer = [k for k in expected_indexer if k not in mapped_indexer]
                    if missing_indexer:
                        preview = ", ".join(missing_indexer[:5])
                        raise RuntimeError(
                            f"Indexer checkpoint missing {len(missing_indexer)} expected tensors. "
                            f"Examples: {preview}"
                        )
                    model.load_state_dict(mapped_indexer, strict=False)
                    print(
                        f"[Indexer] loaded {len(mapped_indexer)}/{len(expected_indexer)} tensors "
                        f"from {args.indexer_path}"
                    )
                    logging.info(
                        f"Loaded {len(mapped_indexer)}/{len(expected_indexer)} indexer tensors "
                        f"(source tensors: {len(indexer_weights)})"
                    )
                else:
                    raise RuntimeError(
                        f"No q_future_proj/q_curr_proj tensors found in indexer checkpoint: {args.indexer_path}"
                    )

        if model.generation_config is None:
            model.generation_config = GenerationConfig.from_model_config(config)
    else:
        if method == "nosa":
            nosa_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "nosa"))
            if nosa_dir not in sys.path:
                sys.path.insert(0, nosa_dir)
            import modeling_llama_nosa as _hf_mod
        else:
            raise ValueError(f"HF backend: cannot detect method from model path {args.model_path}")

        if method == "nosa" and args.long_context:
            if getattr(args, 'yarn', False):
                _hf_mod.LONG_CONTEXT_ENABLED = True
            else:
                _hf_mod.LONG_CONTEXT_ROPE_THETA = 40000
        SparseLlamaForCausalLM = _hf_mod.SparseLlamaForCausalLM

        hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model_sparse_cfg = dict(getattr(hf_config, 'sparse_config', None) or {})
        checkpoint_state = None
        checkpoint_sparse_cfg = {}
        if getattr(args, "indexer_path", None):
            checkpoint_state = torch_load_compat(args.indexer_path, map_location='cpu')
            if isinstance(checkpoint_state, dict):
                checkpoint_sparse_cfg = dict(checkpoint_state.get("sparse_config") or {})

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

        if getattr(args, 'infinigen', False) and method == "nosa":
            sparse_cfg['infinigen_enabled'] = True
            sparse_cfg['infinigen_num_channels'] = 32
            sparse_cfg['use_q_future_for_topk'] = False
            sparse_cfg['create_indexer'] = False
            hf_config.sparse_config = sparse_cfg
            print(f"[HF Loader] NOSA InfiniGen sparse_config: {hf_config.sparse_config}")
        elif getattr(args, 'sparda', False) and method == "nosa":
            sparse_cfg['use_q_future_for_topk'] = True
            sparse_cfg['create_indexer'] = True
            hf_config.sparse_config = sparse_cfg
            print(f"[HF Loader] NOSA SparDA sparse_config: {hf_config.sparse_config}")
        else:
            sparse_cfg['use_q_future_for_topk'] = False
            sparse_cfg['create_indexer'] = False
            if args.dense and method == "nosa":
                sparse_cfg['force_dense_inference'] = True
            hf_config.sparse_config = sparse_cfg

        model = SparseLlamaForCausalLM.from_pretrained(
            args.model_path,
            config=hf_config,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        if getattr(args, 'infinigen', False) and method == "nosa":
            # InfiniGen warmup for NOSA HF
            warmup_ids = torch.randint(0, hf_config.vocab_size, (1, 2048),
                                        device=model.device)
            model.infinigen_warmup(input_ids=warmup_ids,
                                   max_new_tokens=args.max_new_tokens)
            print("[HF Loader] NOSA InfiniGen warmup complete")
        elif getattr(args, 'sparda', False) and getattr(args, 'indexer_path', None) and method == "nosa":
            sd = checkpoint_state.get('model_state_dict', checkpoint_state) if isinstance(checkpoint_state, dict) else {}
            iw = {k: v for k, v in sd.items() if 'q_future_proj' in k or 'q_curr_proj' in k}
            if iw:
                model.load_state_dict(iw, strict=False)
                print(f"[HF Loader] NOSA: loaded {len(iw)} indexer weights")

    model.eval()
    builtins.print = _original_print

    B = args.batch_size
    L = args.seq_len
    max_new_tokens = args.max_new_tokens
    test_n = args.test_n

    lc_str = "128K" if args.long_context else "off"
    mode = "dense" if args.dense else method
    print(f"\n[Config] method={mode}, backend=hf, long_context={lc_str}")
    print(f"[Setup]  batch={B}, input_len={L}, max_new_tokens={max_new_tokens}, "
          f"test_n={test_n}, model={args.model_path}")

    if args.dense:
        @torch.inference_mode()
        def test_time(input_ids):
            prefill_start = torch.cuda.Event(enable_timing=True)
            prefill_end = torch.cuda.Event(enable_timing=True)
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)

            prefill_start.record()
            output = model(input_ids=input_ids, use_cache=True, num_logits_to_keep=1)
            all_gen = [output.logits[:, -1, :].argmax(dim=-1, keepdim=True)]
            past_key_values = output.past_key_values
            prefill_end.record()

            torch.cuda.synchronize()
            decode_start.record()
            for _ in range(max_new_tokens):
                output = model(input_ids=all_gen[-1], past_key_values=past_key_values)
                all_gen.append(output.logits[:, -1, :].argmax(dim=-1, keepdim=True))
                past_key_values = output.past_key_values
            decode_end.record()
            torch.cuda.synchronize()
            prefill_time = prefill_start.elapsed_time(prefill_end) / 1000
            decode_time = decode_start.elapsed_time(decode_end) / 1000
            if args.print_output:
                gen_ids = torch.cat(all_gen, dim=-1)  # (B, max_new_tokens+1)
                _print_generated(tokenizer, gen_ids, tag="hf-full")
            return {
                "prefill_tokens": input_ids.shape[0] * input_ids.shape[1],
                "decode_tokens": input_ids.shape[0] * max_new_tokens,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "prefill_tok_per_s": (input_ids.shape[0] * input_ids.shape[1] / prefill_time) if prefill_time > 0 else 0.0,
                "decode_tok_per_s": (input_ids.shape[0] * max_new_tokens / decode_time) if decode_time > 0 else 0.0,
            }
    else:
        def test_time(input_ids):
            torch.cuda.synchronize()
            total_beg = time.time()
            model.timer_beg = 0
            model.timer_end = 0
            model.passed_iters = 0
            output_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens + 2, do_sample=False,
            )
            torch.cuda.synchronize()
            total_end = time.time()
            model.timer_end = total_end
            gc.collect()
            if args.print_output:
                gen_ids = output_ids[:, input_ids.shape[1]:]  # strip prompt
                _print_generated(tokenizer, gen_ids, tag="hf")
            decode_elapsed = model.timer_end - model.timer_beg
            decode_tok_per_s = (B * max_new_tokens / decode_elapsed) if decode_elapsed > 0 else 0.0
            return _compute_benchmark_metrics(
                total_end - total_beg,
                decode_tok_per_s,
                input_ids.shape[0],
                input_ids.shape[1],
                max_new_tokens,
            )

    input_batches = prepare_input_batches(
        args.dataset_name, args.dataset_split, tokenizer,
        B, L, 1 + test_n,
        model_path=args.model_path,
    )
    if args.print_output:
        _print_input_fingerprint(tokenizer, input_batches, tag="hf")

    # ── Warmup ────────────────────────────────────────────────────────
    metrics = test_time(input_batches[0])
    gc.collect()
    torch.cuda.empty_cache()
    _print_iter_metrics(0, metrics, warmup=True)

    # ── Timed iterations ──────────────────────────────────────────────
    total_prefill_time = 0.0
    total_decode_time = 0.0
    for r in range(test_n):
        metrics = test_time(input_batches[1 + r])
        gc.collect()
        torch.cuda.empty_cache()
        total_prefill_time += metrics["prefill_time"]
        total_decode_time += metrics["decode_time"]
        _print_iter_metrics(r + 1, metrics)

    _print_benchmark_summary(B * L, B * max_new_tokens, total_prefill_time, total_decode_time, test_n)


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified efficiency benchmark for NOSA / MiniCPM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-path", required=True,
        help="HuggingFace model path (e.g. openbmb/NOSA-8B, openbmb/MiniCPM4.1-8B).",
    )
    parser.add_argument(
        "--backend", default="nosi",
        choices=["nosi", "hf"],
        help="Inference backend for the efficiency harness (default: nosi). Use nosi here, not in RULER.",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Force dense (full) attention on the selected backend.",
    )
    parser.add_argument(
        "--infinigen", action="store_true",
        help="Enable InfiniGen decode for MiniCPM or NOSA models.",
    )
    parser.add_argument(
        "--no-offload", action="store_true",
        help="Disable KV cache offloading (GPU-only mode).",
    )
    parser.add_argument(
        "--offload-layer0", action="store_true",
        help="With NOSI offload enabled, also store layer-0 KV cache on CPU instead of "
             "keeping it resident on GPU. This makes layer 0 follow the same offload path "
             "as sparse attention.",
    )
    parser.add_argument(
        "--sparda", action="store_true",
        help="Enable SparDA indexer (q_future/q_curr projections for prefetch overlap). "
             "With --backend nosi, requires --indexer-path.",
    )
    parser.add_argument(
        "--sparda-no-prefetch", action="store_true",
        help="Disable the persistent decode prefetch path for SpARDA on the nosi backend. "
             "Implies --sparda. "
             "Decode will still use q_future/q_curr for block selection, but layers fetch KV via "
             "the normal sparse decode_update_kv pipeline instead of the persistent UVA kernel.",
    )
    parser.add_argument(
        "--indexer-path", default=None,
        help="Path to checkpoint with q_future_proj/q_curr_proj weights. "
             "Required for --sparda with --backend nosi.",
    )
    parser.add_argument(
        "--batch-size", "-B", type=int, default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--seq-len", "-L", type=str, default="128K",
        help="Input sequence length, e.g. 16K, 32K, 128K, or 131072 (default: 128K).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=4,
        help="Number of tokens to generate per sample (default: 4).",
    )
    parser.add_argument(
        "--test-n", type=int, default=4,
        help="Number of timed iterations (excludes 1 warmup) (default: 4).",
    )
    parser.add_argument(
        "--dataset", dest="dataset_name", type=str, default="emozilla/pg19",
        help="Input source: HuggingFace dataset name (default: emozilla/pg19) "
             "or 'ruler' / 'ruler_<task>' for exact RULER prompt text "
             "(default task: niah_single_3).",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="test",
        help="Dataset split to use (default: test).",
    )
    parser.add_argument(
        "--print-output", action="store_true",
        help="Print generated tokens for each sample in the batch (for correctness verification).",
    )
    parser.add_argument(
        "--long-context", action="store_true", default=False,
        help="Enable 128K long-context extension. "
             "For MiniCPM: applies official 128K LongRoPE factors. "
             "For NOSA: increases rope_theta to 40000 (default).",
    )
    parser.add_argument(
        "--yarn", action="store_true", default=False,
        help="Use YaRN (factor=4.0) instead of rope_theta increase for NOSA long-context. "
             "Only effective with --long-context on NOSA models.",
    )

    args = parser.parse_args()
    args.seq_len = parse_seq_len(args.seq_len)

    if args.sparda_no_prefetch:
        args.sparda = True

    method = detect_method(args.model_path)

    # Validate flag combinations
    if args.dense and args.sparda:
        parser.error("--dense and --sparda are mutually exclusive.")
    if args.sparda_no_prefetch and args.backend != "nosi":
        parser.error("--sparda-no-prefetch is only supported with --backend nosi.")
    # InfiniGen now supports both MiniCPM and NOSA models
    # InfiniGen works with both HF and NOSI backends
    if args.infinigen and args.dense:
        parser.error("--dense and --infinigen are mutually exclusive.")

    # Auto-enable long-context when seq_len exceeds native max
    native_max = NATIVE_MAX.get(method, 0)
    if not args.long_context and native_max and args.seq_len > native_max:
        args.long_context = True
        _original_print(
            f"[Auto] --long-context enabled: seq_len {args.seq_len} > "
            f"native max {native_max} for {method}"
        )

    # Validate SparDA + indexer-path (only required for nosi backend)
    if args.sparda and args.indexer_path is None and args.backend == "nosi":
        parser.error("--indexer-path is required when --sparda is set with nosi backend")
    if args.sparda and args.indexer_path is not None:
        if not os.path.exists(args.indexer_path):
            parser.error(f"--indexer-path does not exist: {args.indexer_path}")

    if args.backend == "nosi":
        run_nosi(args)
    else:
        run_hf(args)


if __name__ == "__main__":
    main()
