# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import importlib.util
import logging
import os
import sys
import requests
import torch
from typing import Dict, List, Optional

BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torch_load_compat import torch_load_compat


def _require_visible_cuda() -> None:
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "Local HF RULER eval requires a visible CUDA device, but "
        f"torch.cuda.is_available() is False in {sys.executable}. "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. "
        "If you are running inside a sandboxed shell, launch the benchmark on "
        "a real GPU node or outside the sandbox."
    )
    

def _move_to_visible_cuda(model):
    # Avoid transformers' device_map path, which requires accelerate.
    _require_visible_cuda()
    return model.to("cuda")


def _resolve_attn_implementation() -> str:
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    logging.warning(
        "flash_attn is not installed; falling back to attn_implementation='sdpa'."
    )
    return "sdpa"


class HuggingFaceModel:
    def __init__(
        self,
        name_or_path: str,
        enable_sparse: bool = False,
        enable_sparda: bool = False,
        enable_dense: bool = False,
        enable_infinigen: bool = False,
        indexer_path: str = None,
        long_context: bool = False,
        yarn: bool = False,
        max_seq_length: int = 32768,
        **generation_kwargs,
    ) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig

        self.max_seq_length = max_seq_length
        self._use_num_logits_to_keep = False

        # --- Resolve base model path vs .pt indexer checkpoint ---
        indexer_checkpoint_path = None
        checkpoint = None
        checkpoint_sparse_config = None
        checkpoint_is_full_model = False
        base_model_path = name_or_path

        if name_or_path.endswith('.pt'):
            indexer_checkpoint_path = name_or_path
            checkpoint = torch_load_compat(indexer_checkpoint_path, map_location='cpu')
            checkpoint_sparse_config = checkpoint.get('sparse_config')
            checkpoint_is_full_model = bool(checkpoint.get('full_model', False))
            base_model_path = checkpoint.get('base_model_path', 'openbmb/MiniCPM4.1-8B')
            logging.info(f"Detected .pt checkpoint file: {indexer_checkpoint_path}")
            logging.info(f"Base model path: {base_model_path}")
        elif indexer_path:
            indexer_checkpoint_path = indexer_path
            checkpoint = torch_load_compat(indexer_checkpoint_path, map_location='cpu')
            checkpoint_sparse_config = checkpoint.get('sparse_config')
            checkpoint_is_full_model = bool(checkpoint.get('full_model', False))
            logging.info(f"Separate indexer checkpoint: {indexer_checkpoint_path}")
            logging.info(f"Base model path: {base_model_path}")
        else:
            logging.info(f"Loading model or path: {base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        # Detect NOSA vs InfLLMv2 from model name/path.
        # NOSA 8B uses modeling_llama_nosa (Llama-based, two-stage QK+CIS topk).
        # InfLLMv2 uses modeling_minicpm (MiniCPM-based, single-stage topk).
        is_nosa = 'nosa' in str(name_or_path).lower()

        if is_nosa:
            self._use_num_logits_to_keep = True
            use_infinigen = bool(enable_infinigen)
            use_q_future = bool(enable_sparda) and not use_infinigen
            if use_infinigen and enable_sparda:
                logging.warning("NOSA --infinigen overrides --sparda. Ignoring --sparda.")
            if enable_dense:
                mode_label = "dense"
            elif use_infinigen:
                mode_label = "infinigen"
            elif use_q_future:
                mode_label = "sparda (q_future)"
            else:
                mode_label = "default (QK+CIS)"
            logging.info(f"Loading NOSA model via modeling_llama_nosa.py "
                         f"(Llama-based, two-stage QK+CIS selection, mode={mode_label})")
            import sys
            nosa_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "..", "models", "nosa"))
            if nosa_dir not in sys.path:
                sys.path.insert(0, nosa_dir)
            import modeling_llama_nosa as _nosa_mod
            from transformers import AutoConfig

            if long_context and yarn:
                _nosa_mod.LONG_CONTEXT_ENABLED = True
                _nosa_mod.LONG_CONTEXT_ROPE_THETA = None
                logging.info("NOSA: enabled long-context (YaRN).")
            elif long_context:
                _nosa_mod.LONG_CONTEXT_ENABLED = False
                _nosa_mod.LONG_CONTEXT_ROPE_THETA = 40000
                logging.info("NOSA: enabled long-context (rope_theta=40000).")
            else:
                _nosa_mod.LONG_CONTEXT_ENABLED = False
                _nosa_mod.LONG_CONTEXT_ROPE_THETA = None

            config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
            sparse_cfg = dict(getattr(config, 'sparse_config', None) or {})
            if enable_dense:
                if checkpoint_sparse_config:
                    sparse_cfg.update(checkpoint_sparse_config)
                sparse_cfg['use_q_future_for_topk'] = False
                sparse_cfg['create_indexer'] = False
                sparse_cfg['force_dense_inference'] = True
                logging.info(f"NOSA dense sparse_config: {sparse_cfg}")
            elif use_infinigen:
                if checkpoint_sparse_config:
                    sparse_cfg.update(checkpoint_sparse_config)
                sparse_cfg['infinigen_enabled'] = True
                sparse_cfg['infinigen_num_channels'] = 32
                sparse_cfg['dense_len'] = -1
                sparse_cfg['use_q_future_for_topk'] = False
                sparse_cfg['create_indexer'] = False
                logging.info(f"NOSA InfiniGen sparse_config: {sparse_cfg}")
            elif use_q_future:
                sparse_cfg['dense_len'] = -1
                sparse_cfg['use_q_future_for_topk'] = True
                sparse_cfg['create_indexer'] = True
                if checkpoint_sparse_config:
                    sparse_cfg.update(checkpoint_sparse_config)
                sparse_cfg['dense_len'] = -1
                logging.info(f"NOSA SparDA sparse_config: {sparse_cfg}")
            else:
                # Explicitly disable SparDA indexer path for default NOSA mode.
                sparse_cfg['dense_len'] = -1
                sparse_cfg['use_q_future_for_topk'] = False
                sparse_cfg['create_indexer'] = False
            config.sparse_config = sparse_cfg

            self.pipeline = None
            self.model = _nosa_mod.SparseLlamaForCausalLM.from_pretrained(
                base_model_path,
                config=config,
                torch_dtype=torch.bfloat16,
            )

            if use_q_future and indexer_checkpoint_path:
                if checkpoint is None:
                    checkpoint = torch_load_compat(indexer_checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', {})
                if not state_dict and isinstance(checkpoint, dict) and any(
                        'q_future_proj' in k or 'q_curr_proj' in k for k in checkpoint):
                    state_dict = checkpoint
                    logging.info("NOSA: checkpoint is a raw state_dict (no 'model_state_dict' wrapper)")
                if checkpoint_is_full_model:
                    if state_dict:
                        self.model.load_state_dict(state_dict, strict=False)
                        logging.info(f"NOSA: loaded FULL model weights from checkpoint ({len(state_dict)} tensors)")
                else:
                    indexer_weights = {k: v for k, v in state_dict.items()
                                       if 'q_future_proj' in k or 'q_curr_proj' in k}
                    if indexer_weights:
                        self.model.load_state_dict(indexer_weights, strict=False)
                        logging.info(f"NOSA: loaded {len(indexer_weights)} indexer weight tensors from checkpoint")
                    else:
                        logging.warning("NOSA: no q_future_proj/q_curr_proj weights found in checkpoint")

            self.model = _move_to_visible_cuda(self.model)
            self.model.eval()
            if enable_dense:
                logging.info("NOSA dense mode: modeling_llama_nosa with force_dense_inference=True")
            elif use_infinigen:
                warmup_len = min(self.max_seq_length, 2048)
                warmup_budget = int(generation_kwargs.get("max_new_tokens", 128) or 128)
                logging.info("NOSA InfiniGen: running warmup "
                             f"(warmup_len={warmup_len}, max_new_tokens={warmup_budget})")
                warmup_ids = torch.randint(
                    0, config.vocab_size, (1, warmup_len), device=self.model.device
                )
                self.model.infinigen_warmup(
                    input_ids=warmup_ids,
                    max_new_tokens=warmup_budget,
                )
                logging.info("NOSA InfiniGen warmup complete.")
            else:
                logging.info("NOSA sparse config: window_size=1024, local_blocks=16, "
                             "topk=96, qk_select=41 (1+16+24), two-stage QK+CIS")

        elif enable_sparda or enable_sparse:
            use_q_future = bool(enable_sparda)
            mode_label = "sparda (q_future)" if use_q_future else "sparse (q_current)"
            logging.info(f"Enabling infllmv2 {mode_label} sparse attention (modeling_minicpm.py)")
            import sys
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "models", "minicpm")))
            from modeling_minicpm import (
                MiniCPMForCausalLM as MiniCPMForCausalLM_Future,
                apply_minicpm41_128k_longrope,
            )

            config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

            if long_context:
                apply_minicpm41_128k_longrope(config)
                logging.info("Applied MiniCPM4.1 128K LongRoPE factors.")

            model_sparse_cfg = dict(getattr(config, 'sparse_config', None) or {})

            default_sparse_config = {
                "kernel_size": 32,
                "kernel_stride": 16,
                "init_blocks": 1,
                "block_size": 64,
                "window_size": 2048,
                "topk": 96,
                "use_nope": False,
                "dense_len": -1,
            }

            # Layer: defaults < model config < checkpoint config
            config.sparse_config = default_sparse_config.copy()
            config.sparse_config.update(
                {k: v for k, v in model_sparse_cfg.items() if v is not None})
            if checkpoint_sparse_config:
                logging.info(f"Loading sparse_config from checkpoint: {checkpoint_sparse_config}")
                config.sparse_config.update(checkpoint_sparse_config)
            config.sparse_config['dense_len'] = -1
            config.sparse_config['topk'] = 96

            # Align topk semantics with nosi backend:
            #   MiniCPM-future effective_topk = sparse_config["topk"] + local_blocks
            #   nosi effective_topk           = sparse_config["topk"]
            block_size = int(config.sparse_config.get("block_size", 64))
            window_size = int(config.sparse_config.get("window_size", 2048))
            local_blocks = (window_size // block_size) if block_size > 0 else 0
            nosi_topk = int(config.sparse_config["topk"])
            hf_base_topk = max(1, nosi_topk - local_blocks)
            config.sparse_config["topk"] = hf_base_topk

            config.sparse_config['use_q_future_for_topk'] = use_q_future
            if not use_q_future:
                config.sparse_config['create_indexer'] = False
            attn_impl = _resolve_attn_implementation()
            config._attn_implementation = attn_impl

            logging.info(
                f"Final sparse_config: {config.sparse_config} "
                f"(nosi_topk={nosi_topk}, local_blocks={local_blocks})")

            self.pipeline = None
            self.model = MiniCPMForCausalLM_Future.from_pretrained(
                base_model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )

            # Ensure sparse_config is applied to attention layers
            self.model.config.sparse_config = config.sparse_config
            for layer in self.model.model.layers:
                if hasattr(layer.self_attn, 'use_q_future_for_topk'):
                    layer.self_attn.use_q_future_for_topk = use_q_future
                if hasattr(layer.self_attn, 'dense_len'):
                    layer.self_attn.dense_len = -1

            # Load weights from .pt checkpoint (SparDA mode only)
            if use_q_future and indexer_checkpoint_path:
                if checkpoint is None:
                    checkpoint = torch_load_compat(indexer_checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', {})
                if checkpoint_is_full_model:
                    if state_dict:
                        self.model.load_state_dict(state_dict, strict=False)
                        logging.info(f"Loaded FULL model weights from checkpoint ({len(state_dict)} tensors)")
                else:
                    indexer_weights = {k: v for k, v in state_dict.items() if 'q_future_proj' in k or 'q_curr_proj' in k}
                    if indexer_weights:
                        self.model.load_state_dict(indexer_weights, strict=False)
                        logging.info(f"Loaded {len(indexer_weights)} indexer weight tensors from checkpoint")

            if self.model.generation_config is None:
                self.model.generation_config = GenerationConfig.from_model_config(config)
            self.model = _move_to_visible_cuda(self.model)
            self.model.eval()

        elif enable_infinigen:
            if enable_sparda or enable_sparse:
                logging.warning("--infinigen overrides --enable_sparse/--sparda. "
                                "Ignoring those flags.")
            if 'nosa' in str(name_or_path).lower():
                logging.warning("--infinigen is only supported for InfLLMv2/MiniCPM models, "
                                f"but model path contains 'nosa': {name_or_path}")
            logging.info("Enabling InfiniGen decode (modeling_minicpm_infinigen.py)")
            import sys
            sys.path.append(os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "..", "models", "minicpm")))
            from modeling_minicpm_infinigen import MiniCPMForCausalLM_InfiniGen
            from modeling_minicpm import apply_minicpm41_128k_longrope

            config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
            if long_context:
                apply_minicpm41_128k_longrope(config)
                logging.info("Applied MiniCPM4.1 128K LongRoPE factors.")

            model_sparse_cfg = dict(getattr(config, 'sparse_config', None) or {})
            default_sparse_config = {
                "kernel_size": 32, "kernel_stride": 16, "init_blocks": 1,
                "block_size": 64, "window_size": 2048, "topk": 96,
                "use_nope": False, "dense_len": -1,
            }
            config.sparse_config = default_sparse_config.copy()
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
            attn_impl = _resolve_attn_implementation()
            config._attn_implementation = attn_impl

            self.pipeline = None
            self.model = MiniCPMForCausalLM_InfiniGen.from_pretrained(
                base_model_path, config=config, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl)
            self.model = _move_to_visible_cuda(self.model)
            self.model.eval()
            self.model.config.sparse_config = config.sparse_config
            logging.info(f"InfiniGen: sparse_config={config.sparse_config}")

            if self.model.generation_config is None:
                self.model.generation_config = GenerationConfig.from_model_config(config)

        else:
            # Dense mode (default) - use standard AutoModelForCausalLM
            logging.info(f"Loading model in dense mode from: {base_model_path}")

            if 'Yarn-Llama' in base_model_path:
                model_kwargs = {}
            else:
                model_kwargs = {"attn_implementation": "sdpa"}

            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                **model_kwargs,
            )
            self.model = _move_to_visible_cuda(self.model)

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        import re
        np_pattern = re.compile(r'[\x00-\x1f]')

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            )
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": self.generation_kwargs.get('max_new_tokens', 128),
                    "do_sample": self.generation_kwargs.get('do_sample', False),
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                }
                if self._use_num_logits_to_keep:
                    generate_kwargs["num_logits_to_keep"] = 1
                generated_ids = self.model.generate(
                    **generate_kwargs,
                )

            text = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
            text = np_pattern.sub('\n', text).strip()

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
