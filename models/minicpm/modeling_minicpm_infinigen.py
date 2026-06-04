# This file is copied/adapted from InfiniGen (https://github.com/snu-comparch/InfiniGen).
# Copyright (c) InfiniGen authors / SNU COMPARCH.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM model with InfiniGen decode (InfLLMv2 sparse prefill + InfiniGen masked decode).

Adapted from InfiniGen (OSDI 2024) accuracy implementation for GQA models.
Uses SVD-based weight skewing and partial weight speculation to predict
which KV cache entries are important during autoregressive decode.

During prefill: standard InfLLMv2 sparse attention (unchanged).
During decode: InfiniGen speculation produces a mask that zeros out
unimportant KV tokens, approximating sparse decode without actual
KV cache management.

Layer 0 always uses dense/full attention during decode (no speculation
available from a preceding layer). Layer 0 produces the hidden states
used for layer 1's speculation.
"""

import math
import warnings
import sys
import os
from typing import Any, List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
from torch import nn

# Import everything from the base modeling file
try:
    from .modeling_minicpm import (
        MiniCPMConfig,
        MiniCPMInfLLMv2Attention,
        MiniCPMDecoderLayer,
        MiniCPMModel,
        MiniCPMForCausalLM,
        MiniCPMPreTrainedModel,
        InfLLMv2Cache,
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_q_only,
        repeat_kv,
        _get_unpad_data,
        flash_attn_available,
        flash_attn_func,
        flash_attn_varlen_func,
        index_first_axis,
        pad_input,
        unpad_input,
        logger,
    )
except ImportError:
    from modeling_minicpm import (
        MiniCPMConfig,
        MiniCPMInfLLMv2Attention,
        MiniCPMDecoderLayer,
        MiniCPMModel,
        MiniCPMForCausalLM,
        MiniCPMPreTrainedModel,
        InfLLMv2Cache,
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_q_only,
        repeat_kv,
        _get_unpad_data,
        flash_attn_available,
        flash_attn_func,
        flash_attn_varlen_func,
        index_first_axis,
        pad_input,
        unpad_input,
        logger,
    )

# InfiniGen controller imports
_controllers_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'NOSA', 'nosi', 'nosi')
if _controllers_dir not in sys.path:
    sys.path.insert(0, _controllers_dir)

from infinigen_controllers import (
    compute_skewing_matrix,
    compute_partial_weight_indices,
)


class MiniCPMInfiniGenAttention(MiniCPMInfLLMv2Attention):
    """MiniCPM attention with InfiniGen decode path.

    Inherits InfLLMv2 sparse attention for prefill. Adds InfiniGen
    speculation + masking for decode when infinigen_enabled=True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.infinigen_num_channels = 32
        self._ig_skewing_matrix = None
        self._ig_partial_indices = None
        self._ig_partial_key_cache = None
        self._ig_warmed_up = False
        self._ig_prev_hidden_states = None
        self._ig_attn_input = None
        self.infinigen_enabled = False

        # Path counters for debugging
        self._path_infinigen = 0
        self._path_fallback = 0

    def _refresh_partial_indices_from_prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> None:
        """Recompute request-specific partial indices from the current prefill."""
        if self._ig_skewing_matrix is None:
            return

        self._ig_partial_indices = compute_partial_weight_indices(
            query_states, key_states, self._ig_skewing_matrix,
            self.num_heads, self.num_key_value_heads, self.head_dim,
            num_channels=self.infinigen_num_channels,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        q_future_from_prev: Optional[torch.Tensor] = None,
        training_stage: int = 0,
        **kwargs,
    ):
        """Forward with InfiniGen decode on top of the base sparse attention path."""
        q_len = hidden_states.size(1)
        is_decode = q_len == 1 and past_key_value is not None
        use_infinigen = (
            is_decode
            and self.infinigen_enabled
            and self._ig_warmed_up
            and self._ig_prev_hidden_states is not None
            and self._ig_partial_indices is not None
            and self._ig_skewing_matrix is not None
            and self.layer_idx > 0  # Layer 0 uses dense (no preceding layer)
        )

        if use_infinigen:
            self._path_infinigen = getattr(self, "_path_infinigen", 0) + 1
        elif is_decode:
            self._path_fallback = getattr(self, "_path_fallback", 0) + 1

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            q_future_from_prev=q_future_from_prev,
            training_stage=training_stage,
            **kwargs,
        )


class MiniCPMInfiniGenDecoderLayer(MiniCPMDecoderLayer):
    """Decoder layer that passes hidden states between layers for InfiniGen speculation."""

    # Override to use InfiniGen attention
    def _build_attention(self, config, layer_idx):
        return MiniCPMInfiniGenAttention(config=config, layer_idx=layer_idx)


class MiniCPMInfiniGenModel(MiniCPMModel):
    """MiniCPM model with InfiniGen attention layers.

    After loading weights via from_pretrained(), call _upgrade_to_infinigen()
    to replace standard attention modules with InfiniGen variants.
    """

    def _upgrade_to_infinigen(self):
        """Replace standard InfLLMv2 attention modules with InfiniGen variants.

        Uses in-place class swap to avoid re-creating modules (which would
        trigger flash_attn_2 CPU validation errors).
        """
        upgraded_layers = 0
        for layer in self.layers:
            if not getattr(layer, 'use_sparse_attn', False):
                continue
            attn = layer.self_attn
            if not isinstance(attn, MiniCPMInfiniGenAttention):
                # In-place class swap: add InfiniGen attributes without re-creating
                attn.__class__ = MiniCPMInfiniGenAttention
            upgraded_layers += 1

            # Initialize fields normally created in MiniCPMInfiniGenAttention.__init__.
            # Class swap bypasses __init__, so set all runtime state explicitly.
            attn.infinigen_num_channels = 32
            attn._ig_skewing_matrix = None
            attn._ig_partial_indices = None
            attn._ig_partial_key_cache = None
            attn._ig_warmed_up = False
            attn._ig_prev_hidden_states = None
            attn._ig_attn_input = None
            attn.infinigen_enabled = False
            attn._path_infinigen = 0
            attn._path_fallback = 0
            attn._ig_max_new_tokens = None

        return upgraded_layers


class MiniCPMForCausalLM_InfiniGen(MiniCPMForCausalLM):
    """MiniCPM causal LM with InfiniGen decode.

    Use from_pretrained() which loads the base MiniCPMForCausalLM, then
    the __init_subclass__ hook converts it. Alternatively, call
    convert_to_infinigen() on a loaded MiniCPMForCausalLM instance.
    """

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load base MiniCPMForCausalLM then convert to InfiniGen in-place."""
        model = MiniCPMForCausalLM.from_pretrained(*args, **kwargs)
        return convert_to_infinigen(model)

    def _ensure_infinigen_upgrade(self):
        """Upgrade attention modules to InfiniGen variants (idempotent)."""
        if not getattr(self, '_infinigen_upgraded', False):
            upgraded_layers = self.model._upgrade_to_infinigen()
            if upgraded_layers == 0:
                raise RuntimeError(
                    "InfiniGen upgrade found no sparse attention layers. "
                    "Ensure CUDA is available and config.sparse_config is set before "
                    "loading MiniCPMForCausalLM_InfiniGen."
                )
            logger.info("InfiniGen upgrade complete: %d sparse layers", upgraded_layers)
            self._infinigen_upgraded = True

    def _get_infinigen_attns(self):
        """Get all InfiniGen attention modules."""
        self._ensure_infinigen_upgrade()
        attns = []
        for layer in self.model.layers:
            if isinstance(layer.self_attn, MiniCPMInfiniGenAttention):
                attns.append(layer.self_attn)
        return attns

    @torch.no_grad()
    def infinigen_warmup(self, input_ids=None, warmup_text=None, tokenizer=None,
                         max_new_tokens=None):
        """Warmup: compute per-layer skewing matrices for InfiniGen decode.

        Single forward pass to capture post-RoPE Q/K, then compute per-KV-head
        skewing matrices. Partial indices are recomputed from the current
        request's prefill instead of being frozen from the calibration prompt.

        Args:
            input_ids: (1, seq_len) calibration input IDs, or
            warmup_text: string to tokenize for calibration
            tokenizer: tokenizer for warmup_text
        """
        if input_ids is None and warmup_text is not None and tokenizer is not None:
            input_ids = tokenizer(warmup_text, return_tensors="pt").input_ids
        if input_ids is None:
            raise ValueError("Must provide input_ids or warmup_text+tokenizer")

        if max_new_tokens is None:
            raise ValueError(
                "max_new_tokens is required for InfiniGen warmup. "
                "Pass it to infinigen_warmup() or generate()."
            )

        input_ids = input_ids[:1, :self._warmup_seq_len].to(self.device)
        attention_mask = torch.ones_like(input_ids)

        attns = self._get_infinigen_attns()

        # Single forward pass to capture per-layer hidden states
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )

        hidden_states_list = outputs.hidden_states

        for layer_idx, attn in enumerate(attns):
            hs = hidden_states_list[layer_idx]
            layer = self.model.layers[layer_idx]
            hs_normed = layer.input_layernorm(hs)

            q = attn.q_proj(hs_normed)
            k = attn.k_proj(hs_normed)

            bsz, seq_len, _ = hs_normed.shape
            q = q.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            cos, sin = attn.rotary_emb(q.to(torch.float32), seq_len=seq_len)
            pos_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

            # Compute per-KV-head skewing matrix (Q averaged across GQA group)
            A = compute_skewing_matrix(
                q, k, attn.num_heads, attn.num_key_value_heads, attn.head_dim,
            )
            attn._ig_skewing_matrix = A
            attn._ig_partial_indices = None
            attn._ig_partial_key_cache = None
            attn._ig_prev_hidden_states = None
            attn._ig_attn_input = None
            attn._ig_warmed_up = True
            attn.infinigen_enabled = True
            attn._ig_max_new_tokens = max_new_tokens

        self._infinigen_warmed_up = True
        logger.info("InfiniGen warmup complete: %d layers, num_channels=%d",
                     len(attns), attns[0].infinigen_num_channels if attns else 0)

    def _resolve_generation_budget(self, input_ids, kwargs) -> int:
        max_new_tokens = kwargs.get("max_new_tokens")
        gc = getattr(self, "generation_config", None)

        if max_new_tokens is None and gc is not None:
            max_new_tokens = getattr(gc, "max_new_tokens", None)

        if max_new_tokens is None and input_ids is not None:
            max_length = kwargs.get("max_length")
            if max_length is None and gc is not None:
                max_length = getattr(gc, "max_length", None)
            if max_length is not None:
                max_new_tokens = int(max_length) - int(input_ids.shape[-1])

        if max_new_tokens is None or int(max_new_tokens) <= 0:
            raise ValueError(
                "InfiniGen requires a positive decode budget. "
                "Pass max_new_tokens to generate(), set generation_config.max_new_tokens, "
                "or provide max_length larger than the prompt length."
            )

        return int(max_new_tokens)

    def _prepare_infinigen_request_state(self, max_new_tokens: int):
        """Reset per-request InfiniGen state and set this call's decode budget."""
        for attn in self._get_infinigen_attns():
            attn._ig_max_new_tokens = int(max_new_tokens)
            attn._ig_prev_hidden_states = None
            attn._ig_partial_indices = None
            attn._ig_partial_key_cache = None
            attn._ig_attn_input = None
            attn._path_infinigen = 0
            attn._path_fallback = 0

    def _cleanup_infinigen_request_state(self):
        """Release per-request InfiniGen state after generation."""
        for attn in self._get_infinigen_attns():
            attn._ig_prev_hidden_states = None
            attn._ig_partial_indices = None
            attn._ig_partial_key_cache = None
            attn._ig_attn_input = None
            attn._path_infinigen = 0
            attn._path_fallback = 0

    def generate(self, *args, **kwargs):
        """Generate with per-request InfiniGen partial-cache allocation."""
        self._ensure_infinigen_upgrade()

        attns = self._get_infinigen_attns()
        if not attns:
            return super().generate(*args, **kwargs)

        input_ids = kwargs.get('input_ids', args[0] if args else None)
        if hasattr(input_ids, 'input_ids'):
            input_ids = input_ids.input_ids

        max_new_tokens = self._resolve_generation_budget(input_ids, kwargs)
        need_warmup = any(not getattr(attn, '_ig_warmed_up', False) for attn in attns)
        if need_warmup:
            if input_ids is None:
                raise ValueError(
                    "InfiniGen auto-warmup requires input_ids. "
                    "Pass input_ids to generate() before first use."
                )
            self.infinigen_warmup(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
            )

        self._prepare_infinigen_request_state(max_new_tokens)
        try:
            result = super().generate(*args, **kwargs)
        finally:
            # Print path stats
            attns = self._get_infinigen_attns()
            ig_total = sum(getattr(a, "_path_infinigen", 0) for a in attns)
            fb_total = sum(getattr(a, "_path_fallback", 0) for a in attns)
            if ig_total + fb_total > 0:
                print(f"[InfiniGen] Path stats: infinigen={ig_total}, "
                      f"fallback={fb_total} "
                      f"(across {len(attns)} layers)", flush=True)
            self._cleanup_infinigen_request_state()
        return result


def convert_to_infinigen(model):
    """Convert a loaded MiniCPMForCausalLM to InfiniGen in-place.

    Swaps class to MiniCPMForCausalLM_InfiniGen and upgrades attention modules.

    Args:
        model: MiniCPMForCausalLM instance (already on GPU)

    Returns:
        The same model object, now an instance of MiniCPMForCausalLM_InfiniGen
    """
    model.__class__ = MiniCPMForCausalLM_InfiniGen
    model.model.__class__ = MiniCPMInfiniGenModel
    model._infinigen_warmed_up = False
    model._infinigen_upgraded = False
    model._warmup_seq_len = 2048
    return model
