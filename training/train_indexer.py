# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train SparDA indexer weights (q_future_proj, q_curr_proj) for MiniCPM or NOSA models.

Unified training script that auto-detects model architecture from config.model_type:
  - "minicpm" → MiniCPMForCausalLM (flash_attention_2, scale_emb, InfLLMv2 sparse attention)
  - "llama"   → SparseLlamaForCausalLM (SDPA, CIS pooling, NOSA sparse attention)

Both paths train only indexer weights via per-layer KL divergence loss.
"""
import glob
import os
import random
import re
import sys
import time
import warnings

# Disable tokenizers parallelism to avoid fork warnings in multi-GPU training
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress various warnings - must be before other imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*doesn't directly inherit from `GenerationMixin`.*")
warnings.filterwarnings("ignore", message=".*`torch_dtype` is deprecated.*")
warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*A new version of the following files was downloaded.*")
warnings.filterwarnings("ignore", message=".*Make sure to double-check they do not contain.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import argparse
import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional here
    np = None

def _torch_load_compat(path, map_location="cpu", *, allow_unsafe_fallback: bool = True):
    """Compatibility wrapper for torch.load() across PyTorch versions.

    PyTorch 2.6 changed torch.load default `weights_only` from False -> True.

    Strategy:
    - Try weights_only=True (safer).
    - If that fails due to weights_only restrictions, optionally fall back to weights_only=False
      for trusted local checkpoints.
    """
    serialization = getattr(torch, "serialization", None)

    def _try_weights_only_true():
        # Only attempt if this torch.load supports the kwarg (PyTorch 2.6+).
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            # Older torch that doesn't know weights_only; just do default behavior.
            return torch.load(path, map_location=map_location)

    try:
        return _try_weights_only_true()
    except Exception as e:
        msg = str(e)
        looks_like_weights_only_issue = (
            "Weights only load failed" in msg
            or "weights_only" in msg
            or "WeightsUnpickler" in msg
            or "Unsupported global" in msg
        )
        if allow_unsafe_fallback and looks_like_weights_only_issue:
            warnings.warn(
                f"torch.load(weights_only=True) failed for checkpoint {path}. "
                "Falling back to weights_only=False. Only do this if you trust the checkpoint source.",
                RuntimeWarning,
            )
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


from transformers import AutoConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# Optional dependency: Weights & Biases (only required when --enable_wandb)
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

# Suppress transformers logging
import transformers
transformers.logging.set_verbosity_error()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TRAINING_KERNEL_SIZE = 2
DEFAULT_TRAINING_KERNEL_STRIDE = 1
LEGACY_RUN_NAME_KERNEL_SIZE = 32
LEGACY_RUN_NAME_KERNEL_STRIDE = 16

# Extracted helpers (keeps this training script focused).
# Support both `python training/train_indexer.py` (script) and `python -m training.train_indexer` (module).
try:
    from .minicpm_data import (
        DEFAULT_DATA_PATH,
        PROLONG_DATASET_ID,
        PROLONG_MINICPM_DATA_PATH,
        cleanup_streaming_shm,
        download_prolong_if_needed,
        get_dataloader,
        has_mds_index,
        infinite_dataloader,
        is_prolong_dataset,
    )
except ImportError:
    from minicpm_data import (
        DEFAULT_DATA_PATH,
        PROLONG_DATASET_ID,
        PROLONG_MINICPM_DATA_PATH,
        cleanup_streaming_shm,
        download_prolong_if_needed,
        get_dataloader,
        has_mds_index,
        infinite_dataloader,
        is_prolong_dataset,
    )

try:
    from .ruler_in_memory_eval import evaluate_ruler_accuracy_in_memory, load_tokenizer_for_ruler_eval
except ImportError:
    from ruler_in_memory_eval import evaluate_ruler_accuracy_in_memory, load_tokenizer_for_ruler_eval

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        try:
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def capture_rng_state_by_rank(accelerator):
    local_state = capture_rng_state()
    world_size = getattr(accelerator, "num_processes", 1) or 1
    if world_size <= 1:
        return [local_state]

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_state)
            return gathered
    except Exception:
        pass
    return [local_state]


def restore_rng_state(state):
    if not state:
        return False

    try:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"])
        if np is not None and "numpy" in state:
            np.random.set_state(state["numpy"])
        if torch.cuda.is_available() and "torch_cuda_all" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
        return True
    except Exception:
        return False


def select_rank_rng_state(checkpoint, rank):
    per_rank = checkpoint.get("rng_state_by_rank")
    if isinstance(per_rank, list) and per_rank:
        if 0 <= rank < len(per_rank) and per_rank[rank]:
            return per_rank[rank]
        if per_rank[0]:
            return per_rank[0]
    return checkpoint.get("rng_state")

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint based on modification time.

    Returns:
        (checkpoint_path, step)
        - checkpoint_path: Path to the checkpoint file, or None if not found
        - step: The step number from the checkpoint filename
    """
    patterns = [
        # IMPORTANT: only match numeric regular checkpoints (exclude ckpt-best-acc...).
        os.path.join(output_dir, "ckpt-[0-9]*.pt"),
        os.path.join(output_dir, "checkpoint-step*.pt"),
    ]

    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern))

    all_checkpoints = []
    for ckpt in checkpoints:
        match = re.search(r'ckpt-(\d+)\.pt', ckpt)
        if match:
            all_checkpoints.append((ckpt, int(match.group(1))))
            continue
        match = re.search(r'checkpoint-step(\d+)\.pt', ckpt)
        if match:
            all_checkpoints.append((ckpt, int(match.group(1))))

    if not all_checkpoints:
        return None, 0

    # Sort by modification time (most recent last)
    all_checkpoints.sort(key=lambda x: os.path.getmtime(x[0]))
    return all_checkpoints[-1]

def find_best_accuracy(output_dir):
    """Find the best accuracy from existing best checkpoints."""
    pattern = os.path.join(output_dir, "ckpt-best-acc*.pt")
    best_accuracy = 0.0
    for ckpt in glob.glob(pattern):
        match = re.search(r'best-acc([\d.]+)', ckpt)
        if match:
            best_accuracy = max(best_accuracy, float(match.group(1)))
    return best_accuracy


def count_best_checkpoints(output_dir):
    """Count existing best checkpoints."""
    return len(glob.glob(os.path.join(output_dir, "ckpt-best-acc*.pt")))


def load_indexer_weights(model, checkpoint_path, accelerator):
    """Load indexer weights only from a checkpoint (no optimizer state, no step resume)."""
    if accelerator.is_main_process:
        print(f"Loading indexer weights from {checkpoint_path}")

    checkpoint = _torch_load_compat(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    unwrapped = accelerator.unwrap_model(model)
    model_state = unwrapped.state_dict()

    loaded = 0
    for key, value in model_state_dict.items():
        if key in model_state and ("q_future_proj" in key or "q_curr_proj" in key):
            model_state[key].copy_(value)
            loaded += 1
    if accelerator.is_main_process:
        print(f"Loaded {loaded} indexer parameters from {os.path.basename(checkpoint_path)}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, accelerator):
    """Load checkpoint (indexer weights + optimizer/scheduler/runtime state)."""
    if accelerator.is_main_process:
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = _torch_load_compat(checkpoint_path, map_location="cpu")

    # Handle both old format (just state_dict) and new format (dict with keys)
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        opt_state_dict = checkpoint.get("indexer_optimizer_state_dict") or checkpoint.get("optimizer_state_dict")
        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        step = int(checkpoint.get("step", 0) or 0)
    else:
        model_state_dict = checkpoint
        opt_state_dict = None
        scheduler_state_dict = None
        step = 0

    unwrapped = accelerator.unwrap_model(model)
    model_state = unwrapped.state_dict()

    loaded = 0
    for key, value in model_state_dict.items():
        if key in model_state and ("q_future_proj" in key or "q_curr_proj" in key):
            model_state[key].copy_(value)
            loaded += 1
    if accelerator.is_main_process:
        print(f"Loaded {loaded} indexer parameters from checkpoint")

    if opt_state_dict is not None:
        try:
            optimizer.load_state_dict(opt_state_dict)
            if accelerator.is_main_process:
                print("Loaded optimizer state")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Warning: Could not load optimizer state: {e}")

    scheduler_loaded = False
    if scheduler_state_dict is not None and scheduler is not None:
        try:
            scheduler.load_state_dict(scheduler_state_dict)
            scheduler_loaded = True
            if accelerator.is_main_process:
                print("Loaded scheduler state")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Warning: Could not load scheduler state: {e}")

    return {
        "step": step,
        "micro_batches_seen": int(checkpoint.get("micro_batches_seen", 0) or 0),
        "gradient_accumulation_steps": checkpoint.get("gradient_accumulation_steps"),
        "world_size": checkpoint.get("world_size"),
        "scheduler_loaded": scheduler_loaded,
        "rng_state": select_rank_rng_state(checkpoint, accelerator.process_index or 0),
    }

def save_checkpoint(
    accelerator,
    model,
    optimizer,
    scheduler,
    output_dir,
    step,
    gradient_accumulation_steps,
    base_model_path=None,
    keep_last_n=5,
):
    rng_state_by_rank = capture_rng_state_by_rank(accelerator)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model)
        state_dict = unwrapped.state_dict()

        trainable_state_dict = {k: v for k, v in state_dict.items() if "q_future_proj" in k or "q_curr_proj" in k}

        sparse_config = getattr(unwrapped.config, 'sparse_config', None)
        if sparse_config is not None:
            sparse_config = sparse_config.copy()

        checkpoint = {
            'step': step,
            'micro_batches_seen': int(step) * int(gradient_accumulation_steps),
            'gradient_accumulation_steps': int(gradient_accumulation_steps),
            'world_size': int(getattr(accelerator, "num_processes", 1) or 1),
            'model_state_dict': trainable_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'indexer_optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'sparse_config': sparse_config,
            'base_model_path': base_model_path,
            'rng_state': rng_state_by_rank[0] if rng_state_by_rank else None,
            'rng_state_by_rank': rng_state_by_rank,
        }

        ckpt_path = os.path.join(output_dir, f"ckpt-{step}.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path} ({len(trainable_state_dict)} model keys)")

        # Rotation: only rotate numeric regular checkpoints (exclude ckpt-best-acc...).
        checkpoints = glob.glob(os.path.join(output_dir, "ckpt-[0-9]*.pt"))
        if len(checkpoints) > keep_last_n:
            checkpoints.sort(key=os.path.getmtime)
            for old in checkpoints[:-keep_last_n]:
                try:
                    os.remove(old)
                    print(f"Removed old checkpoint: {old}")
                except OSError as e:
                    print(f"Error deleting checkpoint {old}: {e}")

def save_best_checkpoint(
    accelerator,
    model,
    optimizer,
    scheduler,
    output_dir,
    step,
    accuracy,
    gradient_accumulation_steps,
    base_model_path=None,
    keep_top_n=5,
):
    """Save checkpoint if it's in top N best accuracies."""
    rng_state_by_rank = capture_rng_state_by_rank(accelerator)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

        best_pattern = os.path.join(output_dir, "ckpt-best-acc*.pt")
        existing_best = glob.glob(best_pattern)

        existing_accuracies = []
        for ckpt_path in existing_best:
            match = re.search(r'best-acc([\d.]+)-step\d+\.pt', ckpt_path)
            if match:
                existing_accuracies.append((float(match.group(1)), ckpt_path))

        existing_accuracies.sort(key=lambda x: x[0], reverse=True)

        if len(existing_accuracies) >= keep_top_n and accuracy <= existing_accuracies[-1][0]:
            return False

        unwrapped = accelerator.unwrap_model(model)
        state_dict = unwrapped.state_dict()
        trainable_state_dict = {k: v for k, v in state_dict.items() if "q_future_proj" in k or "q_curr_proj" in k}

        sparse_config = getattr(unwrapped.config, 'sparse_config', None)
        if sparse_config is not None:
            sparse_config = sparse_config.copy()

        checkpoint = {
            'step': step,
            'accuracy': accuracy,
            'micro_batches_seen': int(step) * int(gradient_accumulation_steps),
            'gradient_accumulation_steps': int(gradient_accumulation_steps),
            'world_size': int(getattr(accelerator, "num_processes", 1) or 1),
            'model_state_dict': trainable_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'sparse_config': sparse_config,
            'base_model_path': base_model_path,
            'rng_state': rng_state_by_rank[0] if rng_state_by_rank else None,
            'rng_state_by_rank': rng_state_by_rank,
        }

        ckpt_path = os.path.join(output_dir, f"ckpt-best-acc{accuracy:.4f}-step{step:06d}.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"Saved best checkpoint to {ckpt_path} (accuracy={accuracy:.4f})")

        existing_accuracies.append((accuracy, ckpt_path))
        existing_accuracies.sort(key=lambda x: x[0], reverse=True)
        for acc, path in existing_accuracies[keep_top_n:]:
            try:
                os.remove(path)
                print(f"Removed checkpoint {os.path.basename(path)} (accuracy={acc:.4f})")
            except OSError:
                pass
        return True
    return False


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def get_dataset_short_name(data_path):
    """Get a short identifier for the dataset based on data_path."""
    if is_prolong_dataset(data_path):
        return "prolong64k"
    basename = os.path.basename(data_path.rstrip('/'))
    if len(basename) > 15:
        basename = basename[:15]
    return basename


def _format_seq_len(seq_len):
    if seq_len >= 1024 and seq_len % 1024 == 0:
        return f"{seq_len // 1024}k"
    return str(seq_len)

def _format_lr(lr):
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")

def _resolve_training_kernels(args, model_type):
    """Return training-kernel params from the CLI teacher-grid args."""
    return int(args.training_kernel_size), int(args.training_kernel_stride)


def _resolve_kl_target_align_pooling(args):
    """Return KL-target alignment pooling mode from args."""
    mode = str(args.kl_target_align_pooling).lower()
    if mode not in {"max", "mean"}:
        raise ValueError(f"Unsupported kl_target_align_pooling={args.kl_target_align_pooling!r}")
    return mode


def _build_progressive_training_kernel_schedule(
    kernel_size: int,
    kernel_stride: int,
    training_kernel_size: int,
    training_kernel_stride: int,
    enabled: bool,
):
    """Return the active training-kernel schedule from deployed grid to target grid."""
    kernel_size = int(kernel_size)
    kernel_stride = int(kernel_stride)
    training_kernel_size = int(training_kernel_size)
    training_kernel_stride = int(training_kernel_stride)
    enabled = bool(enabled)

    if not enabled or (
        kernel_size == training_kernel_size and kernel_stride == training_kernel_stride
    ):
        return [(training_kernel_size, training_kernel_stride)]

    if training_kernel_size > kernel_size or training_kernel_stride > kernel_stride:
        raise ValueError(
            "progressive_training_kernel requires training_kernel_size/training_kernel_stride "
            "to be less than or equal to the deployed kernel grid."
        )

    schedule = [(kernel_size, kernel_stride)]
    current_kernel_size = kernel_size
    current_kernel_stride = kernel_stride

    while (
        current_kernel_size != training_kernel_size
        or current_kernel_stride != training_kernel_stride
    ):
        next_kernel_size = current_kernel_size
        next_kernel_stride = current_kernel_stride

        if current_kernel_size > training_kernel_size:
            next_kernel_size = max(training_kernel_size, current_kernel_size // 2)
        if current_kernel_stride > training_kernel_stride:
            next_kernel_stride = max(training_kernel_stride, current_kernel_stride // 2)

        if (
            next_kernel_size == current_kernel_size
            and next_kernel_stride == current_kernel_stride
        ):
            raise ValueError(
                "Unable to build progressive training-kernel schedule from "
                f"({kernel_size}, {kernel_stride}) to ({training_kernel_size}, {training_kernel_stride})."
            )

        current_kernel_size = next_kernel_size
        current_kernel_stride = next_kernel_stride
        schedule.append((current_kernel_size, current_kernel_stride))

    return schedule


def _get_progressive_training_kernel_for_step(schedule, current_step: int, total_steps: int):
    """Return active training-kernel pair and schedule index for the given optimizer step."""
    if not schedule:
        raise ValueError("Progressive training-kernel schedule cannot be empty.")
    if len(schedule) == 1:
        return schedule[0], 0
    if total_steps <= 1:
        return schedule[-1], len(schedule) - 1
    if total_steps < len(schedule):
        stage_idx = min(
            len(schedule) - 1,
            (max(int(current_step), 0) * (len(schedule) - 1)) // max(int(total_steps) - 1, 1),
        )
        return schedule[stage_idx], stage_idx
    if total_steps <= 0:
        return schedule[-1], len(schedule) - 1
    stage_idx = min(len(schedule) - 1, (max(int(current_step), 0) * len(schedule)) // int(total_steps))
    return schedule[stage_idx], stage_idx


def _set_progressive_training_kernel_state(sparse_config, stage_idx, kernel_size, kernel_stride):
    """Persist the active progressive training-kernel state into sparse_config."""
    if sparse_config is None:
        return
    sparse_config["progressive_training_kernel_active_stage"] = int(stage_idx)
    sparse_config["progressive_training_kernel_active_size"] = int(kernel_size)
    sparse_config["progressive_training_kernel_active_stride"] = int(kernel_stride)


def _collect_explicit_cli_args(argv, parser):
    """Return argparse dest names explicitly provided on the CLI."""
    explicit = set()
    option_actions = getattr(parser, "_option_string_actions", {})
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None and getattr(action, "dest", None):
            explicit.add(action.dest)
    return explicit

def get_run_name(args, model_type="minicpm", *, legacy_training_kernel_naming=False):
    """Generate a compact run name based on training configuration."""
    seq_str = _format_seq_len(args.seq_len)
    lr_str = _format_lr(args.lr)
    parts = [f"lr{lr_str}", f"st{args.steps}", f"seq{seq_str}"]

    if args.batch_size != 1:
        parts.append(f"bs{args.batch_size}")
    if args.gradient_accumulation_steps != 1:
        parts.append(f"ga{args.gradient_accumulation_steps}")
    if args.warmup_steps != 0:
        parts.append(f"wu{args.warmup_steps}")

    if args.weight_decay != 0.01:
        parts.append(f"wd{args.weight_decay}")
    if args.max_grad_norm != 0.5:
        parts.append(f"gn{args.max_grad_norm}")

    # Teacher/training-grid params shared by MiniCPM and NOSA.
    if legacy_training_kernel_naming:
        include_training_kernel_size = (
            int(args.training_kernel_size) != LEGACY_RUN_NAME_KERNEL_SIZE
        )
        include_training_kernel_stride = (
            int(args.training_kernel_stride) != LEGACY_RUN_NAME_KERNEL_STRIDE
        )
    else:
        include_training_kernel_size = True
        include_training_kernel_stride = True

    if include_training_kernel_size:
        parts.append(f"tks{args.training_kernel_size}")
    if include_training_kernel_stride:
        parts.append(f"tkst{args.training_kernel_stride}")
    kl_target_align_pooling = _resolve_kl_target_align_pooling(args)
    if kl_target_align_pooling != "max":
        parts.append(f"tkpool{kl_target_align_pooling}")
    if args.progressive_training_kernel:
        parts.append("ptk")

    if args.run_name:
        parts.append(args.run_name)

    # Include dataset tag for both models and a model suffix to keep folders unambiguous.
    parts.append(get_dataset_short_name(args.data_path))
    parts.append("nosa" if model_type == "llama" else "minicpm")

    return "_".join(parts)


def _find_resume_checkpoint_path(args, model_type, explicit_cli_args):
    """Find the latest checkpoint for auto-resume, including pooling-mode fallback."""
    candidate_modes = [_resolve_kl_target_align_pooling(args)]
    if "kl_target_align_pooling" not in explicit_cli_args:
        for mode in ("max", "mean"):
            if mode not in candidate_modes:
                candidate_modes.append(mode)
    candidate_progressive_flags = [bool(args.progressive_training_kernel)]
    if "progressive_training_kernel" not in explicit_cli_args:
        alt_flag = not bool(args.progressive_training_kernel)
        if alt_flag not in candidate_progressive_flags:
            candidate_progressive_flags.append(alt_flag)
    candidate_legacy_training_kernel_naming = [False]
    if (
        int(args.training_kernel_size) == LEGACY_RUN_NAME_KERNEL_SIZE
        or int(args.training_kernel_stride) == LEGACY_RUN_NAME_KERNEL_STRIDE
    ):
        candidate_legacy_training_kernel_naming.append(True)

    original_mode = args.kl_target_align_pooling
    original_progressive = args.progressive_training_kernel
    try:
        for progressive_flag in candidate_progressive_flags:
            args.progressive_training_kernel = progressive_flag
            for mode in candidate_modes:
                args.kl_target_align_pooling = mode
                for legacy_training_kernel_naming in candidate_legacy_training_kernel_naming:
                    run_name = get_run_name(
                        args,
                        model_type=model_type,
                        legacy_training_kernel_naming=legacy_training_kernel_naming,
                    )
                    run_output_dir = os.path.join(args.output_dir, run_name)
                    ckpt_path, ckpt_step = find_latest_checkpoint(run_output_dir)
                    if ckpt_path:
                        return ckpt_path, ckpt_step, run_output_dir, mode
    finally:
        args.kl_target_align_pooling = original_mode
        args.progressive_training_kernel = original_progressive

    run_name = get_run_name(args, model_type=model_type)
    return None, 0, os.path.join(args.output_dir, run_name), None


# ---------------------------------------------------------------------------
# NOSA data helpers
# ---------------------------------------------------------------------------

def _resolve_nosa_data_path(data_path):
    """Resolve data_path for NOSA training.

    NOSA-8B shares the same tokenizer as MiniCPM4.1-8B, so we use the
    MiniCPM-retokenized ProLong dataset (avoids out-of-range token clamping
    from the original Llama-3-tokenized version).
    """
    if data_path in ("prolong-64k", "prolong", PROLONG_DATASET_ID):
        return PROLONG_MINICPM_DATA_PATH
    return data_path


def _is_builtin_prolong_data_path(data_path):
    return data_path in ("prolong-64k", "prolong", PROLONG_DATASET_ID)


def _require_prepared_prolong_minicpm():
    if has_mds_index(PROLONG_MINICPM_DATA_PATH):
        return
    raise RuntimeError(
        "MiniCPM-tokenized ProLong data is required before training with "
        "--data_path prolong-64k. Prepare datasets/prolong-64k-minicpm first; "
        "see README Dataset Preparation."
    )


def _apply_nosa_sparse_runtime_config(
    model,
    sparse_config,
    *,
    training_kernel_size_override=None,
    training_kernel_stride_override=None,
):
    """Apply sparse/runtime settings to NOSA attention layer instances."""
    kernel_size = int(sparse_config.get("kernel_size", 32))
    kernel_stride = int(sparse_config.get("kernel_stride", 16))
    training_kernel_size = int(
        sparse_config.get("training_kernel_size", kernel_size)
        if training_kernel_size_override is None
        else training_kernel_size_override
    )
    training_kernel_stride = int(
        sparse_config.get("training_kernel_stride", kernel_stride)
        if training_kernel_stride_override is None
        else training_kernel_stride_override
    )
    kl_target_align_pooling = str(sparse_config.get("kl_target_align_pooling", "max")).lower()
    if kl_target_align_pooling not in {"max", "mean"}:
        raise ValueError(f"Unsupported kl_target_align_pooling={kl_target_align_pooling!r}")
    init_blocks = int(sparse_config.get("init_blocks", 1))
    block_size = int(sparse_config.get("block_size", 64))
    window_size = int(sparse_config.get("window_size", 1024))
    local_blocks = max(1, window_size // max(1, block_size))
    select_blocks = int(sparse_config.get("select_blocks", 24))
    # Keep effective topk consistent with MiniCPM default (96 including local blocks).
    topk = int(sparse_config.get("topk", 80)) + local_blocks
    dense_len = int(sparse_config.get("dense_len", 8192))
    use_q_future_for_topk = bool(sparse_config.get("use_q_future_for_topk", True))
    use_q_future_decode_only = bool(sparse_config.get("use_q_future_decode_only", False))
    use_nope = bool(sparse_config.get("use_nope", False))
    updated_layers = 0

    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "kernel_size"):
            attn.kernel_size = kernel_size
        if hasattr(attn, "kernel_stride"):
            attn.kernel_stride = kernel_stride
        if hasattr(attn, "init_blocks"):
            attn.init_blocks = init_blocks
        if hasattr(attn, "block_size"):
            attn.block_size = block_size
        if hasattr(attn, "window_size"):
            attn.window_size = window_size
        if hasattr(attn, "local_blocks"):
            attn.local_blocks = local_blocks
        if hasattr(attn, "select_blocks"):
            attn.select_blocks = select_blocks
        if hasattr(attn, "topk"):
            attn.topk = topk
        if hasattr(attn, "dense_len"):
            attn.dense_len = dense_len
        if hasattr(attn, "use_q_future_for_topk"):
            attn.use_q_future_for_topk = use_q_future_for_topk
        if hasattr(attn, "use_q_future_decode_only"):
            attn.use_q_future_decode_only = use_q_future_decode_only
        if hasattr(attn, "use_nope"):
            attn.use_nope = use_nope

        if hasattr(attn, "training_kernel_size"):
            attn.training_kernel_size = training_kernel_size
        if hasattr(attn, "training_kernel_stride"):
            attn.training_kernel_stride = training_kernel_stride
        if hasattr(attn, "kl_target_align_pooling"):
            attn.kl_target_align_pooling = kl_target_align_pooling
        if hasattr(attn, "_kl_target_compress_differs"):
            attn._kl_target_compress_differs = (
                training_kernel_size != kernel_size
                or training_kernel_stride != kernel_stride
            )

        # Compress modules cache their own kernel params.
        if hasattr(attn, "compress_k"):
            attn.compress_k.kernel_size = kernel_size
            attn.compress_k.kernel_stride = kernel_stride
        if (
            hasattr(attn, "compress_k_kl_target")
            and attn.compress_k_kl_target is not None
            and attn.compress_k_kl_target is not getattr(attn, "compress_k", None)
        ):
            attn.compress_k_kl_target.kernel_size = training_kernel_size
            attn.compress_k_kl_target.kernel_stride = training_kernel_stride
        if hasattr(attn, "compress_k2"):
            attn.compress_k2.kernel_size = kernel_size * 4
            attn.compress_k2.kernel_stride = kernel_stride * 4
        updated_layers += 1

    return (
        kernel_size,
        kernel_stride,
        training_kernel_size,
        training_kernel_stride,
        kl_target_align_pooling,
        init_blocks,
        block_size,
        window_size,
        select_blocks,
        topk,
        updated_layers,
    )


def _apply_minicpm_sparse_runtime_config(
    model,
    sparse_config,
    *,
    training_kernel_size_override=None,
    training_kernel_stride_override=None,
):
    """Apply training-kernel runtime settings to MiniCPM attention layer instances."""
    kernel_size = int(sparse_config.get("kernel_size", 32))
    kernel_stride = int(sparse_config.get("kernel_stride", 16))
    training_kernel_size = int(
        sparse_config.get("training_kernel_size", kernel_size)
        if training_kernel_size_override is None
        else training_kernel_size_override
    )
    training_kernel_stride = int(
        sparse_config.get("training_kernel_stride", kernel_stride)
        if training_kernel_stride_override is None
        else training_kernel_stride_override
    )
    kl_target_align_pooling = str(sparse_config.get("kl_target_align_pooling", "max")).lower()
    if kl_target_align_pooling not in {"max", "mean"}:
        raise ValueError(f"Unsupported kl_target_align_pooling={kl_target_align_pooling!r}")
    updated_layers = 0

    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "training_kernel_size"):
            attn.training_kernel_size = training_kernel_size
        if hasattr(attn, "training_kernel_stride"):
            attn.training_kernel_stride = training_kernel_stride
        if hasattr(attn, "kl_target_align_pooling"):
            attn.kl_target_align_pooling = kl_target_align_pooling
        if hasattr(attn, "_kl_target_compress_differs"):
            attn._kl_target_compress_differs = (
                training_kernel_size != kernel_size
                or training_kernel_stride != kernel_stride
            )
        if (
            hasattr(attn, "compress_k_kl_target")
            and attn.compress_k_kl_target is not None
            and attn.compress_k_kl_target is not getattr(attn, "compress_k", None)
        ):
            attn.compress_k_kl_target.kernel_size = training_kernel_size
            attn.compress_k_kl_target.kernel_stride = training_kernel_stride
        if (
            hasattr(attn, "compress_k2_kl_target")
            and attn.compress_k2_kl_target is not None
            and attn.compress_k2_kl_target is not getattr(attn, "compress_k2", None)
        ):
            attn.compress_k2_kl_target.kernel_size = training_kernel_size * 4
            attn.compress_k2_kl_target.kernel_stride = training_kernel_stride * 4
        updated_layers += 1

    return (
        kernel_size,
        kernel_stride,
        training_kernel_size,
        training_kernel_stride,
        kl_target_align_pooling,
        updated_layers,
    )


def _apply_active_training_kernel_runtime_config(
    model,
    sparse_config,
    model_type,
    *,
    training_kernel_size,
    training_kernel_stride,
):
    """Apply the active training-kernel grid to model attention layers."""
    if model_type == "llama":
        return _apply_nosa_sparse_runtime_config(
            model,
            sparse_config,
            training_kernel_size_override=training_kernel_size,
            training_kernel_stride_override=training_kernel_stride,
        )
    return _apply_minicpm_sparse_runtime_config(
        model,
        sparse_config,
        training_kernel_size_override=training_kernel_size,
        training_kernel_stride_override=training_kernel_stride,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_class(model_type):
    """Import and return the model class for the given model_type."""
    if model_type == "minicpm":
        sys.path.append(os.path.join(PROJECT_ROOT, "models", "minicpm"))
        from modeling_minicpm import MiniCPMForCausalLM
        return MiniCPMForCausalLM
    elif model_type == "llama":
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "nosa"))
        from modeling_llama_nosa import SparseLlamaForCausalLM
        return SparseLlamaForCausalLM
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}. Expected 'minicpm' or 'llama'.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # ------------------------------------------------------------------
    # Reject .pt as --model_path (migration from old behavior)
    # Must be checked before Accelerator() init to give a clear error.
    # ------------------------------------------------------------------
    if args.model_path.endswith('.pt') and os.path.isfile(args.model_path):
        raise ValueError(
            f"--model_path must be a HuggingFace model ID or local directory, not a checkpoint file.\n"
            f"Got: {args.model_path}\n"
            f"Migration: use --model_path <hf_model> --indexer_path {args.model_path} instead.\n"
            f"Or use --resume to auto-find the latest checkpoint in the output directory."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps if args.gradient_accumulation_steps > 0 else 1,
        split_batches=False,
    )
    set_seed(42)

    num_gpus = accelerator.num_processes

    if _is_builtin_prolong_data_path(args.data_path):
        _require_prepared_prolong_minicpm()

    if args.gradient_accumulation_steps == 0:
        args.gradient_accumulation_steps = max(1, 32 // (args.batch_size * num_gpus))
        accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps

    effective_batch = args.batch_size * args.gradient_accumulation_steps * num_gpus

    # ------------------------------------------------------------------
    # 1. Load config and detect model type
    # ------------------------------------------------------------------
    base_model_path = args.model_path
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    model_type = getattr(config, "model_type", "minicpm")
    if model_type not in ("minicpm", "llama"):
        raise ValueError(
            f"Unsupported config.model_type={model_type!r}. "
            "Expected 'minicpm' (MiniCPM) or 'llama' (NOSA)."
        )
    is_nosa = model_type == "llama"
    explicit_cli_args = set(getattr(args, "_explicit_cli_args", set()) or [])

    if args.seq_len is None:
        model_max_seq_len = getattr(config, "max_position_embeddings", None)
        if not isinstance(model_max_seq_len, int) or model_max_seq_len <= 0:
            raise ValueError(
                "Could not infer default --seq_len from model config. "
                f"Expected a positive integer at config.max_position_embeddings, got: {model_max_seq_len!r}. "
                "Please pass --seq_len explicitly."
            )
        args.seq_len = int(model_max_seq_len)
        if accelerator.is_main_process:
            print(f"--seq_len not set; defaulting to model max seq len: {args.seq_len}")
    elif args.seq_len <= 0:
        raise ValueError(f"--seq_len must be > 0, got {args.seq_len}.")

    if accelerator.is_main_process:
        print(f"Model type: {model_type} ({'NOSA' if is_nosa else 'MiniCPM'})")
        print(f"Loading model: {base_model_path}")

    # Warn about args ignored by NOSA path
    if is_nosa:
        ignored_nosa_args = {
            "chunk_size": getattr(args, "chunk_size", 0),
        }
        active = {k: v for k, v in ignored_nosa_args.items() if v}
        if active and accelerator.is_main_process:
            print(f"Warning: Args ignored for NOSA model: {list(active.keys())}")

    # ------------------------------------------------------------------
    # Peek at resume checkpoint sparse_config BEFORE model creation
    # ------------------------------------------------------------------
    checkpoint_sparse_config = None
    resume_checkpoint_path = None

    if args.resume:
        if args.indexer_path:
            if not os.path.isfile(args.indexer_path):
                raise FileNotFoundError(f"--indexer_path not found: {args.indexer_path}")
            resume_checkpoint_path = args.indexer_path
            resume_ckpt = _torch_load_compat(resume_checkpoint_path, map_location="cpu")
            checkpoint_sparse_config = resume_ckpt.get('sparse_config')
            ckpt_step = int(resume_ckpt.get("step", 0) or 0)
            if accelerator.is_main_process:
                print(f"Using explicit resume checkpoint: {resume_checkpoint_path} (step {ckpt_step})")
        else:
            ckpt_path, ckpt_step, _run_output_dir, discovered_pool_mode = _find_resume_checkpoint_path(
                args, model_type, explicit_cli_args
            )
            if ckpt_path:
                resume_checkpoint_path = ckpt_path
                resume_ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
                checkpoint_sparse_config = resume_ckpt.get('sparse_config')
                if accelerator.is_main_process:
                    pool_msg = ""
                    if discovered_pool_mode is not None and discovered_pool_mode != _resolve_kl_target_align_pooling(args):
                        pool_msg = f", discovered via kl_target_align_pooling={discovered_pool_mode}"
                    print(f"Found checkpoint for resume: {ckpt_path} (step {ckpt_step}{pool_msg})")

    # ------------------------------------------------------------------
    # Configure sparse_config
    # ------------------------------------------------------------------
    eos_token_id = config.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]
    if config.pad_token_id is None or config.pad_token_id >= config.vocab_size:
        config.pad_token_id = eos_token_id if eos_token_id is not None else 0
    config.pad_token_id = int(config.pad_token_id)

    base_sparse_config = dict(getattr(config, "sparse_config", None) or {})
    base_kernel_size = int(base_sparse_config.get("kernel_size", 32))
    base_kernel_stride = int(base_sparse_config.get("kernel_stride", 16))

    training_kernel_size, training_kernel_stride = _resolve_training_kernels(args, model_type)
    kl_target_align_pooling = _resolve_kl_target_align_pooling(args)
    progressive_training_kernel = bool(args.progressive_training_kernel)

    if is_nosa:
        # NOSA sparse_config
        if not hasattr(config, 'sparse_config') or config.sparse_config is None:
            config.sparse_config = {
                "create_indexer": True,
                "kernel_size": base_kernel_size,
                "kernel_stride": base_kernel_stride,
                "training_kernel_size": training_kernel_size,
                "training_kernel_stride": training_kernel_stride,
                "progressive_training_kernel": progressive_training_kernel,
                "kl_target_align_pooling": kl_target_align_pooling,
                "init_blocks": 1,
                "block_size": 64,
                "window_size": 1024,
                "select_blocks": 24,
                # NOSA uses a smaller local window (1024 -> 16 local blocks),
                # so default selectable topk is 80 to keep effective total at 96.
                "topk": 80,
                "dense_len": 8192,
                "use_nope": False,
                "use_q_future_for_topk": True,
                "use_q_future_decode_only": False,
                "indexer_kaiming_init": True,
            }
        else:
            config.sparse_config["create_indexer"] = True
            config.sparse_config["kernel_size"] = base_kernel_size
            config.sparse_config["kernel_stride"] = base_kernel_stride
            config.sparse_config["training_kernel_size"] = training_kernel_size
            config.sparse_config["training_kernel_stride"] = training_kernel_stride
            config.sparse_config["progressive_training_kernel"] = progressive_training_kernel
            config.sparse_config["kl_target_align_pooling"] = kl_target_align_pooling
            config.sparse_config.setdefault("select_blocks", 24)
            config.sparse_config.setdefault("indexer_kaiming_init", True)
            config.sparse_config.setdefault("dense_len", 8192)
            config.sparse_config.setdefault("use_nope", False)
            config.sparse_config.setdefault("use_q_future_for_topk", True)
            config.sparse_config.setdefault("use_q_future_decode_only", False)
    else:
        # MiniCPM sparse_config
        if not hasattr(config, 'sparse_config') or config.sparse_config is None:
            config.sparse_config = {
                "kernel_size": base_kernel_size,
                "kernel_stride": base_kernel_stride,
                "training_kernel_size": training_kernel_size,
                "training_kernel_stride": training_kernel_stride,
                "progressive_training_kernel": progressive_training_kernel,
                "kl_target_align_pooling": kl_target_align_pooling,
                "init_blocks": 1,
                "block_size": 64,
                "window_size": 2048,
                "topk": 64,
                "use_nope": False,
                "dense_len": 8192,
                "use_q_future_for_topk": True,
                "use_q_future_decode_only": False,
                "indexer_kaiming_init": True,
            }
        else:
            config.sparse_config["kernel_size"] = base_kernel_size
            config.sparse_config["kernel_stride"] = base_kernel_stride
            config.sparse_config["training_kernel_size"] = training_kernel_size
            config.sparse_config["training_kernel_stride"] = training_kernel_stride
            config.sparse_config["progressive_training_kernel"] = progressive_training_kernel
            config.sparse_config["kl_target_align_pooling"] = kl_target_align_pooling
            config.sparse_config["indexer_kaiming_init"] = True
            config.sparse_config.setdefault("use_q_future_for_topk", True)
            config.sparse_config.setdefault("use_q_future_decode_only", False)

    # Restore sparse_config from resume checkpoint
    if checkpoint_sparse_config is not None:
        if config.sparse_config is None:
            config.sparse_config = {}
        for key, value in checkpoint_sparse_config.items():
            config.sparse_config[key] = value
        if accelerator.is_main_process:
            print(f"Restored sparse_config from checkpoint (keys={len(checkpoint_sparse_config)})")

        # Fill required training-only keys if missing from checkpoint.
        config.sparse_config.setdefault("training_kernel_size", training_kernel_size)
        config.sparse_config.setdefault("training_kernel_stride", training_kernel_stride)
        config.sparse_config.setdefault("progressive_training_kernel", progressive_training_kernel)
        config.sparse_config.setdefault("kl_target_align_pooling", "max")
        if is_nosa:
            config.sparse_config.setdefault("select_blocks", 24)

        # Keep args consistent with checkpoint for logging/naming. The CLI kernel
        # args represent the fine KL-teacher grid; deployed routing stays
        # in sparse_config["kernel_*"].
        ckpt_teacher_ks = int(
            config.sparse_config.get(
                "training_kernel_size",
                training_kernel_size,
            )
        )
        if accelerator.is_main_process and args.training_kernel_size != ckpt_teacher_ks:
            print(f"[resume] Overriding args.training_kernel_size={args.training_kernel_size} -> {ckpt_teacher_ks} (from checkpoint training kernel)")
        args.training_kernel_size = ckpt_teacher_ks
        ckpt_teacher_kst = int(
            config.sparse_config.get(
                "training_kernel_stride",
                training_kernel_stride,
            )
        )
        if accelerator.is_main_process and args.training_kernel_stride != ckpt_teacher_kst:
            print(f"[resume] Overriding args.training_kernel_stride={args.training_kernel_stride} -> {ckpt_teacher_kst} (from checkpoint training kernel)")
        args.training_kernel_stride = ckpt_teacher_kst

        _sync_keys = ["kl_target_align_pooling", "progressive_training_kernel"]
        if not is_nosa:
            _sync_keys.extend([
                "dense_len", "window_size", "block_size", "topk", "init_blocks",
                "use_q_future_for_topk", "use_q_future_decode_only", "use_nope",
            ])
        for k in _sync_keys:
            if hasattr(args, k) and k in config.sparse_config:
                old_v = getattr(args, k)
                new_v = config.sparse_config[k]
                if k in explicit_cli_args:
                    config.sparse_config[k] = old_v
                    if accelerator.is_main_process and old_v != new_v:
                        print(f"[resume] Keeping explicit args.{k}={old_v} instead of checkpoint value {new_v}")
                    continue
                if accelerator.is_main_process and old_v != new_v:
                    print(f"[resume] Overriding args.{k}={old_v} -> {new_v} (from checkpoint)")
                setattr(args, k, new_v)

    progressive_training_kernel_schedule = _build_progressive_training_kernel_schedule(
        kernel_size=int((config.sparse_config or {}).get("kernel_size", base_kernel_size)),
        kernel_stride=int((config.sparse_config or {}).get("kernel_stride", base_kernel_stride)),
        training_kernel_size=int((config.sparse_config or {}).get("training_kernel_size", training_kernel_size)),
        training_kernel_stride=int((config.sparse_config or {}).get("training_kernel_stride", training_kernel_stride)),
        enabled=bool((config.sparse_config or {}).get("progressive_training_kernel", False)),
    )

    # Set attention implementation
    if is_nosa:
        config._attn_implementation = "sdpa"
        attn_impl = "sdpa"
    else:
        config._attn_implementation = "flash_attention_2"
        attn_impl = "flash_attention_2"
        config.chunk_size = int(getattr(args, "chunk_size", 0))

    # ------------------------------------------------------------------
    # Build run name and output dir (after sparse_config is finalized)
    # ------------------------------------------------------------------
    run_name = get_run_name(args, model_type=model_type)
    run_output_dir = os.path.join(args.output_dir, run_name)

    if accelerator.is_main_process:
        print(f"Run name: {run_name}")
        print(f"Output directory: {run_output_dir}")
        print(f"Training config: {num_gpus} GPUs, batch_size={args.batch_size}, "
              f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
              f"effective_batch_size={effective_batch}, seq_len={args.seq_len}")

    if args.sanity_train:
        args.steps = min(args.steps, args.sanity_steps)
        if accelerator.is_main_process:
            print(f"Sanity train enabled: limiting to {args.steps} steps.")
    if accelerator.is_main_process and bool((config.sparse_config or {}).get("progressive_training_kernel", False)):
        schedule_boundaries = [
            (idx * args.steps) // len(progressive_training_kernel_schedule)
            for idx in range(len(progressive_training_kernel_schedule))
        ]
        schedule_desc = ", ".join(
            f"step>={boundary}:{ks}/{kst}"
            for boundary, (ks, kst) in zip(schedule_boundaries, progressive_training_kernel_schedule)
        )
        print(f"Progressive training-kernel schedule: {schedule_desc}")

    try:
        import streaming
        streaming.base.util.clean_stale_shared_memory()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Model Vocab Size: {config.vocab_size}")

    ModelClass = _load_model_class(model_type)
    model = ModelClass.from_pretrained(
        base_model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    model.gradient_checkpointing_enable()
    if accelerator.is_main_process:
        print("Gradient checkpointing ENABLED")

    # Load tokenizer for potential evaluation (shared helper keeps behavior consistent).
    tokenizer = load_tokenizer_for_ruler_eval(base_model_path)

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    world_size = accelerator.num_processes or 1
    rank = accelerator.process_index if accelerator.process_index is not None else 0

    total_steps = args.steps
    epoch_samples = total_steps * args.gradient_accumulation_steps * args.batch_size * world_size

    if is_nosa:
        actual_data_path = _resolve_nosa_data_path(args.data_path)
        use_builtin_prolong = _is_builtin_prolong_data_path(args.data_path)
        if accelerator.is_main_process:
            if use_builtin_prolong:
                print(f"Using ProLong-64K dataset (MiniCPM tokenized)")
            else:
                download_prolong_if_needed(actual_data_path, verbose=True)
            print(f"Data path: {actual_data_path}")
        accelerator.wait_for_everyone()
    else:
        # MiniCPM data setup
        use_builtin_prolong = _is_builtin_prolong_data_path(args.data_path)

        if accelerator.is_main_process:
            if use_builtin_prolong:
                print(f"Using ProLong-64K dataset (MiniCPM tokenized)")
                print(f"Data path: {PROLONG_MINICPM_DATA_PATH}")
            else:
                download_prolong_if_needed(args.data_path, verbose=True)
        accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # 4. Setup parameters and optimizer
    # ------------------------------------------------------------------
    num_layers = len(model.model.layers)
    last_layer_prefix = f"model.layers.{num_layers - 1}.self_attn.q_future_proj"

    indexer_params = []
    model_params = []
    trainable_param_count = 0
    total_param_count = 0

    for n, p in model.named_parameters():
        total_param_count += p.numel()
        is_indexer = "q_future_proj" in n or "q_curr_proj" in n
        is_last_layer_indexer = last_layer_prefix in n

        if is_indexer and not is_last_layer_indexer:
            indexer_params.append(p)
            p.requires_grad = True
            trainable_param_count += p.numel()
        else:
            model_params.append(p)
            p.requires_grad = False

    if accelerator.is_main_process:
        pct = 100 * trainable_param_count / max(total_param_count, 1)
        print(f"Training {len(indexer_params)} indexer parameter groups (q_future_proj, q_curr_proj)")
        print(f"Trainable parameters: {trainable_param_count:,} / {total_param_count:,} ({pct:.2f}%)")

    indexer_optimizer = torch.optim.AdamW(
        indexer_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    model = accelerator.prepare(model)
    unwrapped_model = accelerator.unwrap_model(model)

    # LR scheduler with warmup
    from torch.optim.lr_scheduler import LambdaLR

    def get_warmup_scheduler(opt, warmup, total):
        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return 1.0
        return LambdaLR(opt, lr_lambda)

    num_warmup = min(args.warmup_steps, args.steps // 10)
    scheduler = get_warmup_scheduler(indexer_optimizer, num_warmup, args.steps)

    # ------------------------------------------------------------------
    # Load weights: base model → --indexer_path (weights only) → --resume (full state)
    # ------------------------------------------------------------------
    start_step = 0
    resume_state = None
    resume_micro_batches = 0

    # --indexer_path: load weights only (no optimizer, no step) unless it is the explicit --resume checkpoint.
    if args.indexer_path and not args.resume:
        if not os.path.isfile(args.indexer_path):
            raise FileNotFoundError(f"--indexer_path not found: {args.indexer_path}")
        load_indexer_weights(model, args.indexer_path, accelerator)

    # --resume: load weights + optimizer + step (overrides indexer_path weights)
    if args.resume and resume_checkpoint_path:
        resume_state = load_checkpoint(model, indexer_optimizer, scheduler, resume_checkpoint_path, accelerator)
        start_step = int(resume_state.get("step", 0) or 0)
        saved_grad_accum = resume_state.get("gradient_accumulation_steps")
        if saved_grad_accum is not None and int(saved_grad_accum) != int(args.gradient_accumulation_steps):
            raise ValueError(
                "Resume checkpoint was created with gradient_accumulation_steps="
                f"{saved_grad_accum}, but current run uses {args.gradient_accumulation_steps}. "
                "Exact resume requires the same gradient accumulation setting."
            )
        saved_world_size = resume_state.get("world_size")
        current_world_size = int(getattr(accelerator, "num_processes", 1) or 1)
        if saved_world_size is not None and int(saved_world_size) != current_world_size:
            raise ValueError(
                f"Resume checkpoint was created with world_size={saved_world_size}, but current run uses "
                f"world_size={current_world_size}. Exact resume requires the same number of processes."
            )
        resume_micro_batches = int(
            resume_state.get("micro_batches_seen", 0) or (start_step * args.gradient_accumulation_steps)
        )
        if accelerator.is_main_process:
            print(f"Resuming from step {start_step}")
    elif args.resume:
        if accelerator.is_main_process:
            print("No checkpoint found, starting fresh training")

    if start_step > 0 and not (resume_state and resume_state.get("scheduler_loaded")):
        for _ in range(start_step):
            scheduler.step()
        if accelerator.is_main_process:
            print("Resume checkpoint has no scheduler state; reconstructed warmup scheduler position from step count")

    (active_training_kernel_size, active_training_kernel_stride), active_training_kernel_stage = (
        _get_progressive_training_kernel_for_step(
            progressive_training_kernel_schedule,
            current_step=start_step,
            total_steps=args.steps,
        )
    )
    saved_training_kernel_stage = int(
        (config.sparse_config or {}).get(
            "progressive_training_kernel_active_stage",
            active_training_kernel_stage,
        )
    )
    if saved_training_kernel_stage > active_training_kernel_stage:
        if accelerator.is_main_process:
            print(
                f"[resume] Keeping progressive training-kernel stage "
                f"{saved_training_kernel_stage + 1} instead of recomputed "
                f"stage {active_training_kernel_stage + 1} at step {start_step}"
            )
        active_training_kernel_stage = min(
            saved_training_kernel_stage,
            len(progressive_training_kernel_schedule) - 1,
        )
        active_training_kernel_size, active_training_kernel_stride = progressive_training_kernel_schedule[
            active_training_kernel_stage
        ]
    _set_progressive_training_kernel_state(
        config.sparse_config,
        active_training_kernel_stage,
        active_training_kernel_size,
        active_training_kernel_stride,
    )
    runtime_result = _apply_active_training_kernel_runtime_config(
        unwrapped_model,
        config.sparse_config or {},
        model_type,
        training_kernel_size=active_training_kernel_size,
        training_kernel_stride=active_training_kernel_stride,
    )
    if accelerator.is_main_process:
        if is_nosa:
            ks, kst, tks, tkst, tpool, ib, bs, ws, sb, tk, updated = runtime_result
            print(
                f"Applied NOSA sparse compression config: kernel_size={ks}, "
                f"kernel_stride={kst}, training_kernel_size={tks}, "
                f"training_kernel_stride={tkst}, kl_target_align_pooling={tpool}, init_blocks={ib}, block_size={bs}, "
                f"window_size={ws}, select_blocks={sb}, topk={tk} (updated {updated} layers)"
            )
        else:
            ks, kst, tks, tkst, tpool, updated = runtime_result
            print(
                f"Applied MiniCPM sparse compression config: kernel_size={ks}, "
                f"kernel_stride={kst}, training_kernel_size={tks}, "
                f"training_kernel_stride={tkst}, kl_target_align_pooling={tpool} (updated {updated} layers)"
            )
        if bool((config.sparse_config or {}).get("progressive_training_kernel", False)):
            print(
                f"Progressive training kernel active at step {start_step}: "
                f"{active_training_kernel_size}/{active_training_kernel_stride} "
                f"(stage {active_training_kernel_stage + 1}/{len(progressive_training_kernel_schedule)})"
            )

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    wandb_run = None
    is_resuming = start_step > 0

    if args.enable_wandb and accelerator.is_main_process:
        if wandb is None:
            print("Warning: wandb not installed.")
        else:
            try:
                wandb_api_key = os.environ.get("WANDB_API_KEY", "")
                if wandb_api_key:
                    wandb.login(key=wandb_api_key, relogin=True)
                else:
                    wandb.login(relogin=False)

                wandb_run_name = run_name + (f"_resumed_{start_step}" if is_resuming else "")
                resolved_sparse_config = dict(config.sparse_config or {})
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=wandb_run_name,
                    config={
                        **vars(args),
                        "model_type": model_type,
                        "run_output_dir": run_output_dir,
                        "kernel_size": int(resolved_sparse_config.get("kernel_size", base_kernel_size)),
                        "kernel_stride": int(resolved_sparse_config.get("kernel_stride", base_kernel_stride)),
                        "training_kernel_size": int(
                            resolved_sparse_config.get("training_kernel_size", args.training_kernel_size)
                        ),
                        "training_kernel_stride": int(
                            resolved_sparse_config.get("training_kernel_stride", args.training_kernel_stride)
                        ),
                        "progressive_training_kernel": bool(
                            resolved_sparse_config.get(
                                "progressive_training_kernel",
                                args.progressive_training_kernel,
                            )
                        ),
                        "kl_target_align_pooling": resolved_sparse_config.get(
                            "kl_target_align_pooling",
                            getattr(args, "kl_target_align_pooling", "max"),
                        ),
                    },
                    settings=wandb.Settings(init_timeout=60),
                )
                wandb.define_metric("step")
                wandb.define_metric("*", step_metric="step")
            except Exception as e:
                print(f"Warning: Failed to init wandb: {e}")
                wandb_run = None

    # ------------------------------------------------------------------
    # Training loop setup
    # ------------------------------------------------------------------
    model.train()

    _position_ids_cache = {}
    _ones_mask_cache = {}

    def _get_position_ids(seq_length, device):
        key = str(device)
        full = _position_ids_cache.get(key)
        if full is None or full.shape[1] < args.seq_len:
            full = torch.arange(args.seq_len, dtype=torch.long, device=device).unsqueeze(0)
            _position_ids_cache[key] = full
        return full[:, :seq_length]

    def _get_ones_attention_mask(batch_size, seq_length, device):
        key = (str(device), int(batch_size), int(seq_length))
        m = _ones_mask_cache.get(key)
        if m is None:
            m = torch.ones((batch_size, seq_length), device=device)
            _ones_mask_cache[key] = m
        return m

    def _batch_to_device(batch):
        return {
            k: v.to(accelerator.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _allreduce_indexer_grads(params):
        import torch.distributed as dist
        if not dist.is_initialized():
            return
        for p in params:
            if p is not None and getattr(p, "grad", None) is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    # Internal cache-flush policy
    empty_cache_every_steps = 10 if args.seq_len >= 65536 else 0

    def _maybe_empty_cache(step=None, *, force=False):
        if not torch.cuda.is_available():
            return
        if force:
            torch.cuda.empty_cache()
            return
        every = empty_cache_every_steps
        if every and step is not None and step % every == 0:
            torch.cuda.empty_cache()

    def _post_ruler_eval_cleanup(*, seq_len, tag=""):
        if seq_len < 65536:
            return
        try:
            import gc; gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _format_gib(x):
        return f"{x / (1024**3):.2f} GiB"

    def _cuda_mem_report(tag, step=None, *, reset_peak=False):
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        max_alloc, max_reserved, max_peak_alloc, max_peak_reserved = alloc, reserved, peak_alloc, peak_reserved
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                t = torch.tensor([alloc, reserved, peak_alloc, peak_reserved], device=accelerator.device, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                max_alloc, max_reserved, max_peak_alloc, max_peak_reserved = (int(t[0].item()), int(t[1].item()), int(t[2].item()), int(t[3].item()))
        except Exception:
            pass

        if accelerator.is_main_process:
            step_str = f" step={step}" if step is not None else ""
            print(
                f"[mem]{step_str} {tag} | "
                f"alloc={_format_gib(max_alloc)} reserved={_format_gib(max_reserved)} | "
                f"peak_alloc={_format_gib(max_peak_alloc)} peak_reserved={_format_gib(max_peak_reserved)}"
            )

        if reset_peak:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Layer-wise forward functions (model-type specific)
    # ------------------------------------------------------------------
    def run_layer_forward_minicpm(hidden_states, attention_mask, position_ids,
                                  grad_accum_steps, is_last_pass=True, training_stage=1):
        """Layer-wise forward with per-layer KL backward for MiniCPM."""
        q_future_prev = None
        loss_tensor = torch.tensor(0.0, device=accelerator.device)
        layers_with_loss = 0
        n_layers = len(unwrapped_model.model.layers)

        for layer_idx, layer in enumerate(unwrapped_model.model.layers):
            is_last_layer = (layer_idx == n_layers - 1)

            use_ckpt = (
                bool(getattr(unwrapped_model.model, "gradient_checkpointing", False))
                and unwrapped_model.model.training
            )
            if use_ckpt:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,   # past_key_value
                    False,  # output_attentions
                    False,  # use_cache
                    q_future_prev,
                    training_stage,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                    q_future_from_prev=q_future_prev,
                    training_stage=training_stage,
                )

            hidden_states_next = layer_outputs[0]
            q_future_curr = layer_outputs[-2]
            aux_scores = layer_outputs[-1]

            if aux_scores is not None:
                scores_pred, scores_target = aux_scores
                loss = scores_pred if scores_target == "precomputed" else None
                if loss is not None and loss.requires_grad:
                    loss = loss / grad_accum_steps
                    should_sync = is_last_layer and is_last_pass
                    if should_sync or not hasattr(model, 'no_sync'):
                        loss.backward()
                    else:
                        with model.no_sync():
                            loss.backward()
                    loss_tensor = loss_tensor + loss.detach() * grad_accum_steps
                    layers_with_loss += 1

            hidden_states = hidden_states_next.detach()
            q_future_prev = q_future_curr

        return loss_tensor, layers_with_loss

    def run_layer_forward_nosa(hidden_states, attention_mask, position_ids,
                               grad_accum_steps, is_last_pass=True, training_stage=1):
        """Layer-wise forward with per-layer KL backward for NOSA (Llama)."""
        q_future_prev = None
        loss_tensor = torch.tensor(0.0, device=accelerator.device)
        layers_with_loss = 0
        n_layers = len(unwrapped_model.model.layers)

        # Position embeddings computed once and shared across layers
        position_embeddings = unwrapped_model.model.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(unwrapped_model.model.layers):
            is_last_layer = (layer_idx == n_layers - 1)

            use_ckpt = (
                bool(getattr(unwrapped_model.model, "gradient_checkpointing", False))
                and unwrapped_model.model.training
            )
            if use_ckpt:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,   # past_key_value
                    False,  # output_attentions
                    False,  # use_cache
                    None,   # cache_position
                    position_embeddings,
                    q_future_prev,
                    training_stage,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                    q_future_from_prev=q_future_prev,
                    training_stage=training_stage,
                )

            hidden_states_next = layer_outputs[0]
            q_future_curr = layer_outputs[-2]
            kl_loss = layer_outputs[-1]

            # NOSA returns kl_loss as a scalar tensor directly
            if kl_loss is not None and isinstance(kl_loss, torch.Tensor) and kl_loss.requires_grad:
                loss = kl_loss / grad_accum_steps
                should_sync = is_last_layer and is_last_pass
                if should_sync or not hasattr(model, 'no_sync'):
                    loss.backward()
                else:
                    with model.no_sync():
                        loss.backward()
                loss_tensor = loss_tensor + loss.detach() * grad_accum_steps
                layers_with_loss += 1

            hidden_states = hidden_states_next.detach()
            q_future_prev = q_future_curr

        return loss_tensor, layers_with_loss

    # Select forward function based on model type
    run_layer_forward = run_layer_forward_nosa if is_nosa else run_layer_forward_minicpm

    # ------------------------------------------------------------------
    # Dataloader setup
    # ------------------------------------------------------------------
    _cuda_mem_report("after_prepare", reset_peak=True)

    if accelerator.is_main_process:
        print("Initializing dataloader...")

    is_node_leader = accelerator.is_local_main_process
    if is_node_leader:
        cleanup_streaming_shm(verbose=accelerator.is_main_process)
    accelerator.wait_for_everyone()

    data_path_for_loader = actual_data_path if is_nosa else args.data_path
    dataloader = get_dataloader(
        data_path_for_loader, args.batch_size, args.seq_len, epoch_samples,
        verbose=accelerator.is_main_process, rank=rank, world_size=world_size,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Dataloader ready (all ranks)")
    _cuda_mem_report("after_dataloader_init", reset_peak=True)
    if resume_state and resume_state.get("rng_state"):
        restored_rng = restore_rng_state(resume_state["rng_state"])
        if accelerator.is_main_process:
            if restored_rng:
                print("Restored RNG state from checkpoint")
            else:
                print("Warning: Failed to restore RNG state from checkpoint; resume may not be bitwise-consistent")
    elif args.resume and start_step > 0 and accelerator.is_main_process:
        print("Warning: Resume checkpoint has no RNG state; resume will be approximate")

    # ------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        if start_step > 0:
            print(f"Resuming training (step {start_step} -> {args.steps}, {args.steps - start_step} remaining)")
            print(f"Resuming data stream from microbatch {resume_micro_batches}")
        else:
            print(f"Starting training ({args.steps} steps, lr={args.lr})")
        print("Goal: Train indexer (q_future_proj, q_curr_proj) via KL divergence")

    remaining_steps = args.steps - start_step
    progress_bar = tqdm(range(remaining_steps), disable=not accelerator.is_main_process,
                        desc=f"Training (from step {start_step})" if start_step > 0 else "Training")
    if accelerator.is_main_process:
        existing_best = count_best_checkpoints(run_output_dir)
        if existing_best > 0:
            best_acc = find_best_accuracy(run_output_dir)
            print(f"Found {existing_best} existing best checkpoint(s) (top accuracy: {best_acc:.4f})")

    current_step = start_step
    accumulated_loss = 0.0
    accumulated_layers = 0
    torch.cuda.empty_cache()

    for step, batch in enumerate(infinite_dataloader(dataloader, accelerator, start_batch=resume_micro_batches)):
        if current_step >= args.steps:
            break

        (
            (active_training_kernel_size, active_training_kernel_stride),
            next_training_kernel_stage,
        ) = _get_progressive_training_kernel_for_step(
            progressive_training_kernel_schedule,
            current_step=current_step,
            total_steps=args.steps,
        )
        if next_training_kernel_stage > active_training_kernel_stage:
            active_training_kernel_stage = next_training_kernel_stage
            active_training_kernel_size, active_training_kernel_stride = progressive_training_kernel_schedule[
                active_training_kernel_stage
            ]
            _set_progressive_training_kernel_state(
                config.sparse_config,
                active_training_kernel_stage,
                active_training_kernel_size,
                active_training_kernel_stride,
            )
            _apply_active_training_kernel_runtime_config(
                unwrapped_model,
                config.sparse_config or {},
                model_type,
                training_kernel_size=active_training_kernel_size,
                training_kernel_stride=active_training_kernel_stride,
            )
            if accelerator.is_main_process:
                print(
                    f"Progressive training kernel switched at step {current_step}: "
                    f"{active_training_kernel_size}/{active_training_kernel_stride} "
                    f"(stage {active_training_kernel_stage + 1}/{len(progressive_training_kernel_schedule)})"
                )

        batch = _batch_to_device(batch)

        with accelerator.accumulate(model):
            input_ids_full = batch["input_ids"]
            seq_len = min(args.seq_len, input_ids_full.shape[1])
            input_ids = input_ids_full[:, :seq_len]

            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask[:, :seq_len]

            bsz, seqlen = input_ids.shape
            device = input_ids.device

            position_ids = _get_position_ids(seqlen, device)
            attention_mask_2d = attention_mask if attention_mask is not None else _get_ones_attention_mask(bsz, seqlen, device)

            # Embedding (model-type specific)
            if is_nosa:
                inputs_embeds = unwrapped_model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = unwrapped_model.model.embed_tokens(input_ids) * unwrapped_model.config.scale_emb

            # Attention mask preparation (model-type specific)
            if is_nosa:
                # NOSA uses passthrough 2D mask
                final_mask = attention_mask_2d
            else:
                use_sparse = unwrapped_model.model._use_sparse_attn
                use_flash = unwrapped_model.model._use_flash_attention_2
                if use_sparse:
                    final_mask = attention_mask_2d
                elif use_flash:
                    final_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                else:
                    final_mask = _prepare_4d_causal_attention_mask(
                        attention_mask_2d, (bsz, seqlen), inputs_embeds, 0
                    )

            loss_tensor, layers_count = run_layer_forward(
                inputs_embeds,
                final_mask,
                position_ids,
                grad_accum_steps=args.gradient_accumulation_steps,
                is_last_pass=accelerator.sync_gradients,
                training_stage=1,
            )

            grad_norm = None
            if accelerator.sync_gradients:
                _allreduce_indexer_grads(indexer_params)
                max_norm = args.max_grad_norm if args.max_grad_norm > 0 else float('inf')
                grad_norm = torch.nn.utils.clip_grad_norm_(indexer_params, max_norm)
                indexer_optimizer.step()
                scheduler.step()
                indexer_optimizer.zero_grad()

            micro_loss = loss_tensor.item()
            accumulated_loss += micro_loss
            accumulated_layers += layers_count

            del inputs_embeds

        if accelerator.sync_gradients:
            progress_bar.update(1)
            loss_value = accumulated_loss / max(accumulated_layers, 1)
            accumulated_loss = 0.0
            accumulated_layers = 0
            current_step += 1

            if wandb_run and accelerator.is_main_process:
                log_dict = {"step": current_step, "train/loss": loss_value}
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = grad_norm.item()
                log_every = max(int(getattr(args, "wandb_log_every_steps", 1)), 1)
                if current_step == 1 or current_step % log_every == 0 or current_step == args.steps:
                    wandb.log(log_dict, step=current_step)

            should_print = args.sanity_train or (current_step % args.save_every_steps == 0)
            if should_print and accelerator.is_main_process:
                print(f"Step {current_step}, Loss: {loss_value:.4f}")

            if current_step % args.save_every_steps == 0 and current_step > 0:
                save_checkpoint(
                    accelerator,
                    model,
                    indexer_optimizer,
                    scheduler,
                    run_output_dir,
                    current_step,
                    args.gradient_accumulation_steps,
                    base_model_path,
                )
                _cuda_mem_report("after_ckpt", step=current_step)
                _maybe_empty_cache(current_step)

                # Evaluate RULER accuracy for best checkpoint selection
                if args.eval_for_best_checkpoint:
                    eval_kernel_size = int((config.sparse_config or {}).get("kernel_size", base_kernel_size))
                    eval_kernel_stride = int((config.sparse_config or {}).get("kernel_stride", base_kernel_stride))
                    accuracy = evaluate_ruler_accuracy_in_memory(
                        model=model,
                        tokenizer=tokenizer,
                        accelerator=accelerator,
                        seq_len=args.seq_len,
                        num_samples_per_task=args.ruler_num_samples,
                        kernel_size=eval_kernel_size,
                        kernel_stride=eval_kernel_stride,
                        output_dir=run_output_dir,
                    )
                    _post_ruler_eval_cleanup(seq_len=args.seq_len, tag=f"step{current_step}")

                    if wandb_run and accelerator.is_main_process:
                        wandb.log({"step": current_step, "train/ruler_accuracy": accuracy}, step=current_step)

                    save_best_checkpoint(
                        accelerator,
                        model,
                        indexer_optimizer,
                        scheduler,
                        run_output_dir,
                        current_step,
                        accuracy,
                        args.gradient_accumulation_steps,
                        base_model_path,
                    )

                    accelerator.wait_for_everyone()

    # Final save
    save_checkpoint(
        accelerator,
        model,
        indexer_optimizer,
        scheduler,
        run_output_dir,
        current_step,
        args.gradient_accumulation_steps,
        base_model_path,
    )
    if accelerator.is_main_process:
        print(f"Training complete. Final checkpoint at step {current_step}")

    accelerator.wait_for_everyone()
    if wandb_run:
        wandb_run.finish()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SparDA indexer weights (q_future_proj, q_curr_proj). "
                    "Auto-detects MiniCPM vs NOSA from config.model_type."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model ID/path (e.g. openbmb/MiniCPM4.1-8B or openbmb/NOSA-8B). "
                             "Must be a model directory, NOT a .pt checkpoint file.")
    parser.add_argument("--indexer_path", type=str, default=None,
                        help="Optional path to a .pt checkpoint. "
                             "Without --resume, loads indexer weights only. "
                             "With --resume, uses this exact checkpoint for full-state resume.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help="Dataset path. 'prolong-64k' (default) for ProLong-64K, "
                             "or a local ProLong-format MDS path.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0,
                        help="Gradient accumulation steps (0=auto to maintain effective batch size of 32)")
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Chunk size for chunked CE+MLP (MiniCPM only, 0 disables).")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Sequence length for training. "
                             "Defaults to model config max_position_embeddings when omitted.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for indexer training.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Max gradient norm for clipping (0 to disable)")
    parser.add_argument("--save_every_steps", type=int, default=50, help="Save checkpoint every N steps.")
    parser.add_argument("--sanity_train", action="store_true", help="Run a short sanity training loop.")
    parser.add_argument("--sanity_steps", type=int, default=10, help="Steps in sanity mode.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint if available.")
    parser.add_argument("--run_name", type=str, default="",
                        help="Optional suffix to distinguish concurrent runs.")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="indexer_training",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_log_every_steps", type=int, default=1,
                        help="Log metrics to W&B every N optimizer steps.")
    parser.add_argument("--eval_for_best_checkpoint", action="store_true",
                        help="Evaluate RULER accuracy at each checkpoint for best model selection.")
    parser.add_argument("--ruler_num_samples", type=int, default=50,
                        help="Samples per RULER task for validation.")
    # Shared teacher-grid args (MiniCPM + NOSA).
    parser.add_argument("--training_kernel_size", type=int, default=DEFAULT_TRAINING_KERNEL_SIZE,
                        help="Fine KL-teacher kernel size used during training. Deployed routing grid comes from model sparse_config.")
    parser.add_argument("--training_kernel_stride", type=int, default=DEFAULT_TRAINING_KERNEL_STRIDE,
                        help="Fine KL-teacher kernel stride used during training. Deployed routing grid comes from model sparse_config.")
    parser.add_argument("--progressive_training_kernel", action="store_true",
                        help="Progressively anneal the active training kernel from deployed kernel_size/kernel_stride to training_kernel_size/training_kernel_stride across total training steps.")
    parser.add_argument("--kl_target_align_pooling", type=str, choices=("max", "mean"), default="max",
                        help="Pooling mode used when aligning finer KL targets to the pred compressed grid.")

    args = parser.parse_args()
    args._explicit_cli_args = _collect_explicit_cli_args(sys.argv[1:], parser)
    train(args)
