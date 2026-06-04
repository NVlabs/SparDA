# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory RULER evaluation helpers.

Extracted from the training script so training code stays readable.
"""

from __future__ import annotations

import os
import re
import sys
import time
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_tokenizer_for_ruler_eval(base_model_path: str):
    """Load tokenizer used for in-memory RULER evaluation."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def evaluate_ruler_accuracy_in_memory(
    model,
    tokenizer,
    accelerator,
    seq_len=32768,
    num_samples_per_task=50,
    kernel_size=32,
    kernel_stride=16,
    output_dir=None,
):
    """
    Evaluate RULER accuracy using the current in-memory model with sample-based
    distribution across all nodes/GPUs.
    
    This is functionally identical to run_minicpm.sh but uses the in-memory model
    directly instead of loading from checkpoint, enabling faster evaluation.
    
    Implements sample-based distribution: all samples across all tasks are pooled
    and distributed round-robin across all GPUs (matching call_api_distributed.py).
    
    Data Path: Uses NOSA/benchmarks/RULER/scripts/benchmark_root/RULER/data/{seq_len}_n{num_samples}/
               This matches run_minicpm.sh default and enables data sharing across runs.
    
    Args:
        model: The current model (wrapped by accelerator or unwrapped)
        tokenizer: The tokenizer for the model
        accelerator: Accelerator object for distributed training
        seq_len: Sequence length to evaluate at
        num_samples_per_task: Number of samples per RULER task
        kernel_size: Sparse attention kernel size (default: 32, matches run_minicpm.sh)
        kernel_stride: Sparse attention kernel stride (default: 16, matches run_minicpm.sh)
        output_dir: Unused (kept for API compatibility). Data path uses shared benchmark_root.
    
    Returns:
        float: Average accuracy across all tasks (0.0-100.0)
    """
    import json
    import yaml
    import tempfile
    import shutil
    import subprocess
    import socket
    from pathlib import Path
    
    eval_start_time = time.time()

    def _distributed_barrier(tag: str = ""):
        """Use explicit NCCL device_ids to avoid unknown-device barrier hangs."""
        if torch.distributed.is_initialized():
            import torch.distributed as dist

            if dist.get_backend() == "nccl":
                device_index = None
                if accelerator.device.type == "cuda":
                    device_index = accelerator.device.index
                if device_index is None:
                    local_rank = getattr(accelerator, "local_process_index", None)
                    if local_rank is not None:
                        device_index = int(local_rank)
                if device_index is not None:
                    dist.barrier(device_ids=[device_index])
                    return
        accelerator.wait_for_everyone()
    
    # Get distributed info
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    is_main = accelerator.is_main_process
    
    if is_main:
        print(f"[InMemoryRULER] Starting evaluation with {world_size} GPUs (sample-based distribution)")
        print(f"[InMemoryRULER] seq_len={seq_len}, samples_per_task={num_samples_per_task}")
        print(f"[InMemoryRULER] kernel_size={kernel_size}, kernel_stride={kernel_stride}")
    
    # RULER paths — use NOSA/benchmarks/RULER which has the same data generation
    # scripts (synthetic.yaml, data/prepare.py, eval/) as the root RULER/ folder.
    ruler_scripts_dir = os.path.join(PROJECT_ROOT, "NOSA", "benchmarks", "RULER", "scripts")
    synthetic_yaml = os.path.join(ruler_scripts_dir, "synthetic.yaml")
    
    if not os.path.exists(synthetic_yaml):
        if is_main:
            print(f"[InMemoryRULER] Error: synthetic.yaml not found at {synthetic_yaml}")
        return 0.0
    
    # Load task configurations from synthetic.yaml (matching run_minicpm.sh)
    with open(synthetic_yaml, 'r') as f:
        tasks_customized = yaml.safe_load(f)
    
    # Task list from config_tasks.sh - EXACT match
    task_list = [
        "niah_single_1", "niah_single_2", "niah_single_3",
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multivalue", "niah_multiquery",
        "vt", "cwe", "fwe", "qa_1", "qa_2"
    ]
    
    # tokens_to_generate from data/synthetic/constants.py - EXACT match
    # These values are used by call_api_distributed.py via config.get('tokens_to_generate', 50)
    TASKS_BASE = {
        'niah': {'tokens_to_generate': 128},
        'variable_tracking': {'tokens_to_generate': 30},
        'common_words_extraction': {'tokens_to_generate': 120},
        'freq_words_extraction': {'tokens_to_generate': 50},
        'qa': {'tokens_to_generate': 32},
    }
    
    # Setup data directory (matching run_minicpm.sh ROOT_DIR/RULER/data path)
    # In run_minicpm.sh: ROOT_DIR="${RULER_OUTPUT_DIR:-benchmark_root}"
    # ROOT_DIR is relative to ruler_scripts_dir
    # 
    # IMPORTANT: Data is SHARED across all evaluations (same seq_len and num_samples)
    # This is intentional - data generation is expensive and deterministic.
    # We always use the default benchmark_root path for data.
    # The output_dir parameter is ignored for data path to ensure sharing.
    root_dir = os.path.join(ruler_scripts_dir, "benchmark_root")
    
    # Data path format: {ROOT_DIR}/RULER/data/{MAX_SEQ_LENGTH}_n{NUM_SAMPLES}
    data_dir = os.path.join(root_dir, "RULER", "data", f"{seq_len}_n{num_samples_per_task}")
    
    # IMPORTANT: Use a shared directory for temp predictions.
    # Also avoid collisions between overlapping evaluations.
    #
    # Fix: let rank 0 create a unique directory under a shared parent, then broadcast the chosen
    # path to all ranks so everyone writes into the same (collision-proof) temp directory.
    temp_parent = os.path.join(root_dir, ".ruler_temp")
    pred_dir_base = None
    if is_main:
        os.makedirs(temp_parent, exist_ok=True)

        # Cleanup policy:
        # - Do NOT delete the current eval temp dir at the end of evaluation (keeps artifacts for debugging).
        # - Instead, delete the *previous* eval temp dir right before starting a new eval.
        # This mirrors "clean up before next in-memory eval" behavior and avoids leaving piles of dirs.
        run_id = os.environ.get("SPARDA_RUN_ID", f"pid{os.getpid()}")
        host = socket.gethostname().split(".")[0]
        last_dir_marker = os.path.join(temp_parent, f".ruler_inmem_last_dir_{run_id}_{host}")

        # Force fresh run (like run_minicpm.sh FORCE_FRESH_RUN):
        # If enabled, remove any previous in-memory eval temp dirs for this run+host before starting.
        # This helps avoid confusion from lingering old/empty artifacts.
        force_fresh = bool(int(os.environ.get("RULER_FORCE_FRESH_RUN", "0")))
        if force_fresh:
            prefix = f"ruler_inmem_{run_id}_{host}_"
            removed = 0
            try:
                for name in os.listdir(temp_parent):
                    if name.startswith(prefix):
                        shutil.rmtree(os.path.join(temp_parent, name), ignore_errors=True)
                        removed += 1
            except Exception:
                pass
            try:
                if os.path.exists(last_dir_marker):
                    os.remove(last_dir_marker)
            except Exception:
                pass
            print(f"[InMemoryRULER] Force fresh run enabled: removed {removed} old temp dir(s) with prefix {prefix!r}")

        try:
            if os.path.exists(last_dir_marker):
                with open(last_dir_marker, "r") as f:
                    last_dir = f.read().strip()
                # Only delete if it looks like a ruler temp dir under our parent.
                if last_dir and os.path.isdir(last_dir) and os.path.dirname(last_dir) == temp_parent:
                    # Extra safety for concurrent runs: only delete if _meta.txt matches our run+host.
                    meta_path = os.path.join(last_dir, "_meta.txt")
                    ok = False
                    try:
                        if os.path.exists(meta_path):
                            with open(meta_path, "r") as mf:
                                meta = mf.read()
                            ok = (f"run_id={run_id}\n" in meta) and (f"host={host}\n" in meta)
                    except Exception:
                        ok = False
                    if ok:
                        shutil.rmtree(last_dir, ignore_errors=True)
        except Exception:
            pass

        # NOTE: Old-dir cleanup (>1h) removed intentionally to avoid race with concurrent long-running evals.
        # Manual cleanup is safe; run-scoped cleanup of *previous* eval is done above.

        pred_dir_base = tempfile.mkdtemp(
            prefix=f"ruler_inmem_{run_id}_{host}_",
            dir=temp_parent,
        )
        # Record this eval directory so the *next* eval can clean it up.
        try:
            tmp_marker = last_dir_marker + ".tmp"
            with open(tmp_marker, "w") as f:
                f.write(pred_dir_base)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_marker, last_dir_marker)
        except Exception:
            pass
        # Write a tiny metadata file for postmortems/debugging.
        try:
            meta_path = os.path.join(pred_dir_base, "_meta.txt")
            with open(meta_path, "w") as f:
                f.write(f"run_id={run_id}\n")
                f.write(f"host={host}\n")
                f.write(f"time_ns={time.time_ns()}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    # Multi-node safe broadcast:
    # `broadcast_object_list` can be backend-sensitive; broadcasting a byte buffer works reliably
    # for NCCL/Gloo and avoids object-collective pitfalls.
    if torch.distributed.is_initialized():
        import torch.distributed as dist
        max_bytes = 1024  # plenty for absolute paths
        path_len = torch.zeros(1, dtype=torch.int64, device=accelerator.device)
        path_buf = torch.zeros(max_bytes, dtype=torch.uint8, device=accelerator.device)
        if is_main:
            b = pred_dir_base.encode("utf-8")
            if len(b) >= max_bytes:
                raise RuntimeError(f"[InMemoryRULER] pred_dir_base path too long ({len(b)} bytes): {pred_dir_base}")
            path_len[0] = len(b)
            path_buf[: len(b)] = torch.tensor(list(b), dtype=torch.uint8, device=accelerator.device)
        dist.broadcast(path_len, src=0)
        dist.broadcast(path_buf, src=0)
        if not is_main:
            n = int(path_len.item())
            pred_dir_base = bytes(path_buf[:n].cpu().tolist()).decode("utf-8")
    else:
        # Single-process fallback (rank0 only).
        if pred_dir_base is None:
            os.makedirs(temp_parent, exist_ok=True)
            pred_dir_base = tempfile.mkdtemp(prefix="ruler_inmem_", dir=temp_parent)

    # Ensure the directory is visible before ranks start creating rank subdirs.
    _distributed_barrier("pred_dir_ready")
    try:
        import time as _time_mod
        _time_mod.sleep(1)
    except Exception:
        pass
    
    # Step 1: Generate data (only on main process, others wait)
    # Matches run_minicpm.sh data generation logic (lines 285-308)
    #
    # Race-safe for concurrent runs on a shared filesystem:
    # - Use atomic mkdir() lockdir instead of fcntl.flock (more robust across distributed FS configs).
    # - If lock holder crashes, allow stale-lock recovery (age threshold).
    # - Create .data_ready atomically (write tmp + rename).
    data_ready_file = os.path.join(data_dir, ".data_ready")
    data_lock_dir = os.path.join(data_dir, ".data_lockdir")

    def _is_data_complete() -> bool:
        """Best-effort check that required files exist and are non-empty."""
        if not os.path.exists(data_ready_file):
            return False
        for task in task_list:
            task_file = os.path.join(data_dir, task, "validation.jsonl")
            if not os.path.exists(task_file):
                return False
            try:
                if os.path.getsize(task_file) <= 0:
                    return False
            except Exception:
                return False
        return True

    if is_main:
        os.makedirs(data_dir, exist_ok=True)

        if _is_data_complete():
            print(f"[InMemoryRULER] Using existing data at {data_dir}")
        else:
            import time as time_module

            stale_seconds = int(os.environ.get("RULER_DATA_LOCK_STALE_SECONDS", str(2 * 3600)))  # 2h default
            lock_acquired = False

            # Loop: try to acquire lock, else wait; recover stale locks.
            for _attempt in range(999999):
                if _is_data_complete():
                    break
                try:
                    os.mkdir(data_lock_dir)  # atomic across nodes
                    lock_acquired = True
                    break
                except FileExistsError:
                    # Someone else is generating. If lock looks stale, try to remove it.
                    try:
                        age = time_module.time() - os.path.getmtime(data_lock_dir)
                        if age > stale_seconds:
                            print(f"[InMemoryRULER] Detected stale data lock (age={int(age)}s), removing: {data_lock_dir}")
                            shutil.rmtree(data_lock_dir, ignore_errors=True)
                            continue
                    except Exception:
                        pass
                    # Wait before retry
                    time_module.sleep(5)
                    continue

            if lock_acquired:
                try:
                    # Re-check after acquiring lock (another process could have finished between attempts)
                    if not _is_data_complete():
                        print(f"[InMemoryRULER] Generating data for {len(task_list)} tasks...")
                        prepare_script = os.path.join(ruler_scripts_dir, "data", "prepare.py")

                        for task in task_list:
                            cmd = [
                                sys.executable, prepare_script,
                                "--save_dir", data_dir,
                                "--benchmark", "synthetic",
                                "--task", task,
                                "--tokenizer_path", "openbmb/MiniCPM4.1-8B",  # TOKENIZER_PATH in run_minicpm.sh
                                "--tokenizer_type", "hf",                      # TOKENIZER_TYPE in run_minicpm.sh
                                "--max_seq_length", str(seq_len),
                                "--model_template_type", "minicpm4",
                                "--num_samples", str(num_samples_per_task),
                            ]
                            result = subprocess.run(cmd, cwd=ruler_scripts_dir, capture_output=True, text=True)
                            if result.returncode != 0:
                                print(f"[InMemoryRULER] Warning: Failed to generate data for {task}")
                                if result.stderr:
                                    print(f"  stderr: {result.stderr[-200:]}")

                        # Mark data as ready atomically
                        try:
                            tmp_ready = data_ready_file + ".tmp"
                            with open(tmp_ready, "w") as f:
                                f.write("ready\n")
                                f.flush()
                                os.fsync(f.fileno())
                            os.rename(tmp_ready, data_ready_file)
                        except Exception:
                            # Fall back to non-atomic marker if needed
                            with open(data_ready_file, "w") as f:
                                f.write("ready\n")
                        print(f"[InMemoryRULER] Data generation complete")
                finally:
                    shutil.rmtree(data_lock_dir, ignore_errors=True)

            if _is_data_complete():
                print(f"[InMemoryRULER] Using existing data at {data_dir}")
            else:
                # If we couldn't complete data (e.g., repeated generator failures), let ranks hit the usual timeout.
                print(f"[InMemoryRULER] WARNING: Data still incomplete after generation/locking attempts: {data_dir}")
    else:
        # Non-main processes wait for data to be ready (matching run_minicpm.sh lines 286-291)
        # This is important for multi-node scenarios
        import time as time_module
        wait_count = 0
        # First data generation can take 15-20 minutes, so use 30 minute timeout
        while not os.path.exists(data_ready_file):
            time_module.sleep(5)
            wait_count += 1
            if wait_count > 360:  # 30 minute timeout (360 * 5s)
                print(f"[Rank {rank}] Timeout waiting for data generation (30 min)")
                break
    
    # Synchronize all processes (additional barrier for safety)
    _distributed_barrier("data_ready")
    
    # Verify data is actually ready before proceeding
    if not os.path.exists(data_ready_file):
        raise RuntimeError(f"[Rank {rank}] Data ready file not found after barrier: {data_ready_file}")
    
    # Step 2: Load all samples and distribute across GPUs
    # Matches call_api_distributed.py sample loading logic
    def read_manifest(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    # Build list of all work units (matching call_api_distributed.py)
    # Format: (task_name, sample_idx, sample_data, tokens_to_generate)
    all_work_units = []
    
    for task_name in task_list:
        task_file = os.path.join(data_dir, task_name, "validation.jsonl")
        if not os.path.exists(task_file):
            if is_main:
                print(f"[InMemoryRULER] Warning: Task file not found: {task_file}")
            continue
        
        # Get task config (matching call_api_distributed.py config merging)
        config = tasks_customized.get(task_name, {}).copy()
        base_task = config.get('task', 'niah')
        if base_task in TASKS_BASE:
            config.update(TASKS_BASE[base_task])
        tokens_to_generate = config.get('tokens_to_generate', 50)
        
        # Load samples
        samples = read_manifest(task_file)
        for sample_idx, sample in enumerate(samples):
            all_work_units.append((task_name, sample_idx, sample, tokens_to_generate))
    
    total_samples = len(all_work_units)
    
    if is_main:
        print(f"[InMemoryRULER] Total samples: {total_samples} across {len(task_list)} tasks")
        print(f"[InMemoryRULER] Distribution: ~{(total_samples + world_size - 1) // world_size} samples per GPU")
    
    # Distribute samples across ranks (round-robin, matching call_api_distributed.py)
    my_work_units = [all_work_units[i] for i in range(rank, total_samples, world_size)]
    
    # Step 3: Run inference
    # Unwrap model and set to eval mode
    unwrapped_model = accelerator.unwrap_model(model)
    # Some distributed wrappers can wrap the real module; always run inference on the underlying module if present.
    eval_model = getattr(unwrapped_model, "module", unwrapped_model)
    was_training = unwrapped_model.training
    unwrapped_model.eval()
    try:
        eval_model.eval()
    except Exception:
        pass

    # CRITICAL: generation should use KV-cache.
    # In training we enable gradient checkpointing, which in many HF models implicitly disables cache
    # (config.use_cache=False). run_minicpm.sh loads the model for inference with cache enabled.
    # If cache is disabled, generate() effectively re-prefills every token, and our sparse-attn decode
    # path (which depends on past_key_values / InfLLMv2Cache) will behave very differently.
    orig_use_cache = None
    orig_gen_use_cache = None
    gc_was_enabled = bool(getattr(eval_model, "gradient_checkpointing", False))
    try:
        if hasattr(eval_model, "config") and hasattr(eval_model.config, "use_cache"):
            orig_use_cache = eval_model.config.use_cache
            eval_model.config.use_cache = True
        if hasattr(eval_model, "generation_config") and eval_model.generation_config is not None:
            orig_gen_use_cache = getattr(eval_model.generation_config, "use_cache", None)
            eval_model.generation_config.use_cache = True
        # Best-effort: disable gradient checkpointing during eval (not needed and can interact with caching)
        if gc_was_enabled and hasattr(eval_model, "gradient_checkpointing_disable"):
            eval_model.gradient_checkpointing_disable()
    except Exception:
        pass
    # Match pred/model_wrappers.py: ensure pad token exists
    # (important when batching; harmless for batch_size=1)
    try:
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    # Save and update sparse config for evaluation (matching run_minicpm.sh sparda mode)
    # sparda mode: --enable_future_indexer --topk_mode future
    #
    # CRITICAL: Must update BOTH sparse_config dict AND layer instance attributes.
    # The attention layers read from instance attributes, not the config dict at runtime.
    # This matches model_wrappers.py lines 183-190 which explicitly updates layer attrs.
    orig_sparse_config = {}
    
    if hasattr(eval_model, 'config') and hasattr(eval_model.config, 'sparse_config'):
        sparse_config = eval_model.config.sparse_config
        # Save original values for restoration after eval
        orig_sparse_config = {
            'kernel_size': sparse_config.get('kernel_size'),
            'kernel_stride': sparse_config.get('kernel_stride'),
            'use_q_future_for_topk': sparse_config.get('use_q_future_for_topk'),
            'use_q_future_decode_only': sparse_config.get('use_q_future_decode_only'),
            'dense_len': sparse_config.get('dense_len'),
        }
        # Set evaluation values (matching sparda mode in run_minicpm.sh)
        sparse_config['kernel_size'] = kernel_size
        sparse_config['kernel_stride'] = kernel_stride
        sparse_config['use_q_future_for_topk'] = True  # topk_mode=future
        sparse_config['use_q_future_decode_only'] = False
        # Force pure sparse for benchmark eval: disable dense flash-attn fallback for all sequence lengths.
        # (The attention module uses `kv_seq_len < dense_len` to choose dense vs sparse.)
        eval_dense_len = -1
        sparse_config['dense_len'] = eval_dense_len
        
        if is_main:
            print(f"[InMemoryRULER] Sparse config for eval: kernel_size={kernel_size}, kernel_stride={kernel_stride}, "
                  f"dense_len={eval_dense_len}, use_q_future_for_topk=True")
        
        # Update attention layer instance attributes (critical for correct inference)
        # Must update ALL attributes that the attention forward() reads from self.*
        # This matches model_wrappers.py lines 183-190
        for layer in eval_model.model.layers:
            if hasattr(layer.self_attn, 'kernel_size'):
                layer.self_attn.kernel_size = kernel_size
            if hasattr(layer.self_attn, 'kernel_stride'):
                layer.self_attn.kernel_stride = kernel_stride
            if hasattr(layer.self_attn, 'use_q_future_for_topk'):
                layer.self_attn.use_q_future_for_topk = True
            if hasattr(layer.self_attn, 'use_q_future_decode_only'):
                layer.self_attn.use_q_future_decode_only = False
            # CRITICAL: dense_len determines sparse vs dense path (model_wrappers.py line 189)
            if hasattr(layer.self_attn, 'dense_len'):
                layer.self_attn.dense_len = eval_dense_len
            # Update CompressK modules which store their own kernel_size/kernel_stride
            if hasattr(layer.self_attn, 'compress_k'):
                layer.self_attn.compress_k.kernel_size = kernel_size
                layer.self_attn.compress_k.kernel_stride = kernel_stride
            if hasattr(layer.self_attn, 'compress_k2'):
                layer.self_attn.compress_k2.kernel_size = kernel_size * 4
                layer.self_attn.compress_k2.kernel_stride = kernel_stride * 4
    
    # Group by tokens_to_generate for efficiency (matching call_api_distributed.py)
    work_by_tokens = {}
    for task_name, sample_idx, sample, tokens in my_work_units:
        if tokens not in work_by_tokens:
            work_by_tokens[tokens] = []
        work_by_tokens[tokens].append((task_name, sample_idx, sample))
    
    def postprocess_pred(predict_str):
        """Strip whitespace and remove non-printable characters.
        Matches model_wrappers.py postprocess_pred exactly."""
        predict_str = predict_str.strip()
        np_pattern = re.compile(r'[\x00-\x1f]')
        predict_str = np_pattern.sub('\n', predict_str).strip()
        return predict_str
    
    # Process samples (matching call_api_distributed.py and model_wrappers.py)
    my_results = {}  # task_name -> [(sample_idx, result_dict)]
    device = next(eval_model.parameters()).device
    
    with torch.no_grad():
        for tokens_to_generate, work_units in work_by_tokens.items():
            for task_name, sample_idx, sample in work_units:
                try:
                    # Build prompt (matching call_api_distributed.py)
                    prompt = sample['input'] + sample.get('answer_prefix', '')
                    
                    # Tokenize with truncation (match pred/model_wrappers.py)
                    # Default behavior: do NOT force add_special_tokens either way.
                    tok_kwargs = dict(
                        return_tensors="pt",
                        truncation=True,
                        max_length=seq_len,
                    )
                    inputs = tokenizer(prompt, **tok_kwargs)
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    input_len = input_ids.shape[1]
                    
                    # Generate (matching model_wrappers.py exactly)
                    # do_sample=False for greedy (TEMPERATURE=0.0 in config_models.sh)
                    # run_minicpm.sh/model_wrappers.py do not pass use_cache here (it comes from model defaults).
                    outputs = eval_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=tokens_to_generate,
                        do_sample=False,  # Greedy decoding
                        # Explicitly force KV-cache on. Relying on config can be brittle because
                        # training enables gradient checkpointing which often flips config.use_cache=False.
                        # run_minicpm.sh loads the model for inference with cache enabled.
                        use_cache=True,
                        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
                    )
                    
                    # Extract output (matching model_wrappers.py)
                    pred = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                    pred = postprocess_pred(pred)
                    
                    if task_name not in my_results:
                        my_results[task_name] = []
                    
                    # Result format matches call_api_distributed.py
                    my_results[task_name].append((sample_idx, {
                        'index': sample.get('index', sample_idx),
                        'input': sample['input'],
                        'outputs': sample.get('outputs', []),
                        'pred': pred,
                        'prompt_len': int(input_len),
                        'others': sample.get('others', {}),
                        'truncation': sample.get('truncation', -1),
                        'length': sample.get('length', -1),
                    }))
                    
                except Exception as e:
                    print(f"[Rank {rank}] Error processing {task_name}/{sample_idx}: {e}")
                    if task_name not in my_results:
                        my_results[task_name] = []
                    my_results[task_name].append((sample_idx, {
                        'index': sample.get('index', sample_idx),
                        'input': sample['input'],
                        'outputs': sample.get('outputs', []),
                        'pred': '',
                        'prompt_len': -1,
                        'error': str(e),
                        'others': sample.get('others', {}),
                        'truncation': sample.get('truncation', -1),
                        'length': sample.get('length', -1),
                    }))
    
    # Restore model state
    if was_training:
        unwrapped_model.train()
        try:
            eval_model.train()
        except Exception:
            pass

    # Restore cache / grad-checkpointing state
    try:
        if orig_use_cache is not None and hasattr(eval_model, "config") and hasattr(eval_model.config, "use_cache"):
            eval_model.config.use_cache = orig_use_cache
        if hasattr(eval_model, "generation_config") and eval_model.generation_config is not None and orig_gen_use_cache is not None:
            eval_model.generation_config.use_cache = orig_gen_use_cache
        if gc_was_enabled and hasattr(eval_model, "gradient_checkpointing_enable"):
            eval_model.gradient_checkpointing_enable()
    except Exception:
        pass
    
    # Restore original sparse config
    if orig_sparse_config and hasattr(eval_model, 'config') and hasattr(eval_model.config, 'sparse_config'):
        sparse_config = eval_model.config.sparse_config
        for key, value in orig_sparse_config.items():
            if value is None:
                # Key was not originally present; remove if we added it during eval.
                try:
                    sparse_config.pop(key, None)
                except Exception:
                    pass
            else:
                sparse_config[key] = value
        # Restore attention layer attributes (including CompressK modules)
        orig_ks = orig_sparse_config.get('kernel_size')
        orig_kst = orig_sparse_config.get('kernel_stride')
        orig_dense_len = orig_sparse_config.get('dense_len')
        for layer in eval_model.model.layers:
            if hasattr(layer.self_attn, 'kernel_size') and orig_ks is not None:
                layer.self_attn.kernel_size = orig_ks
            if hasattr(layer.self_attn, 'kernel_stride') and orig_kst is not None:
                layer.self_attn.kernel_stride = orig_kst
            if hasattr(layer.self_attn, 'use_q_future_for_topk') and orig_sparse_config.get('use_q_future_for_topk') is not None:
                layer.self_attn.use_q_future_for_topk = orig_sparse_config['use_q_future_for_topk']
            if hasattr(layer.self_attn, 'use_q_future_decode_only') and orig_sparse_config.get('use_q_future_decode_only') is not None:
                layer.self_attn.use_q_future_decode_only = orig_sparse_config['use_q_future_decode_only']
            # Restore dense_len
            if hasattr(layer.self_attn, 'dense_len') and orig_dense_len is not None:
                layer.self_attn.dense_len = orig_dense_len
            # Restore CompressK modules
            if hasattr(layer.self_attn, 'compress_k') and orig_ks is not None and orig_kst is not None:
                layer.self_attn.compress_k.kernel_size = orig_ks
                layer.self_attn.compress_k.kernel_stride = orig_kst
            if hasattr(layer.self_attn, 'compress_k2') and orig_ks is not None and orig_kst is not None:
                layer.self_attn.compress_k2.kernel_size = orig_ks * 4
                layer.self_attn.compress_k2.kernel_stride = orig_kst * 4

    # Step 4: Save partial results to temp files (matching call_api_distributed.py)
    # Each rank saves to pred_dir_base/rank{N}/task.jsonl
    #
    # IMPORTANT: Use atomic writes (write to .tmp, fsync, rename) to avoid filesystem races
    # where rank 0 sees the file but content hasn't fully propagated.
    pred_dir = os.path.join(pred_dir_base, f"rank{rank}")
    os.makedirs(pred_dir, exist_ok=True)
    
    # Track what we write for verification in .done marker
    written_files = {}  # task_name -> (num_samples, file_size)
    
    for task_name, results in my_results.items():
        # Avoid creating empty files (e.g., if a task key was created but no samples were recorded).
        if not results:
            continue
        final_file = os.path.join(pred_dir, f'{task_name}.jsonl')
        tmp_file = final_file + '.tmp'
        
        # Write to temp file first
        with open(tmp_file, 'w') as f:
            for _, result_dict in sorted(results, key=lambda x: x[0]):
                f.write(json.dumps(result_dict) + '\n')
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic rename (POSIX guarantees atomicity)
        os.rename(tmp_file, final_file)
        
        # Record for verification
        written_files[task_name] = (len(results), os.path.getsize(final_file))
    
    # Sync directory metadata after renames
    try:
        dir_fd = os.open(pred_dir, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(dir_fd)
        os.close(dir_fd)
    except (OSError, AttributeError):
        pass
    
    # Write completion marker with verification info
    # Include task counts so reader can verify completeness
    done_marker = os.path.join(pred_dir, '.done')
    done_tmp = done_marker + '.tmp'
    with open(done_tmp, 'w') as f:
        f.write(f'rank={rank}\n')
        f.write(f'num_tasks={len(written_files)}\n')
        for task_name, (count, size) in written_files.items():
            f.write(f'{task_name}={count},{size}\n')
        f.flush()
        os.fsync(f.fileno())
    os.rename(done_tmp, done_marker)
    
    # CRITICAL: Synchronize all processes BEFORE aggregation
    # This ensures all ranks have written their files before main tries to read them
    _distributed_barrier("predictions_written")
    
    # Shared filesystems can have significant metadata latency across processes.
    import time as time_mod
    time_mod.sleep(5)
    
    # Step 5: Main process aggregates and evaluates results
    # Matches eval/evaluate.py logic exactly
    overall_accuracy = 0.0
    
    if is_main:
        print(f"[InMemoryRULER] Aggregating results from {world_size} ranks...")
        print(f"[InMemoryRULER] Temp directory: {pred_dir_base}")
        
        # Parse .done marker to get expected file info for verification
        def parse_done_marker(done_path):
            """Parse .done marker to get expected task counts and file sizes."""
            info = {}
            try:
                with open(done_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line:
                            key, val = line.split('=', 1)
                            if key not in ('rank', 'num_tasks'):
                                # task_name=count,size
                                parts = val.split(',')
                                if len(parts) == 2:
                                    info[key] = (int(parts[0]), int(parts[1]))
            except Exception:
                pass
            return info
        
        # Wait for all rank .done markers (more reliable than directory existence)
        # Each rank writes a .done file AFTER all task files are written and fsync'd
        max_retries = 30
        retry_delay = 2
        for retry in range(max_retries):
            missing_ranks = []
            for r in range(world_size):
                done_marker = os.path.join(pred_dir_base, f"rank{r}", '.done')
                if not os.path.exists(done_marker):
                    missing_ranks.append(r)
            
            if not missing_ranks:
                break  # All ranks done
            
            if retry < max_retries - 1:
                if retry % 5 == 0:  # Print every 5th retry to reduce spam
                    print(f"[InMemoryRULER] Waiting for {len(missing_ranks)} rank(s) to complete (retry {retry+1}/{max_retries})...")
                time_mod.sleep(retry_delay)
        
        # Verify files match expected sizes to catch propagation issues.
        verification_failures = []
        for r in range(world_size):
            done_marker = os.path.join(pred_dir_base, f"rank{r}", '.done')
            rank_dir = os.path.join(pred_dir_base, f"rank{r}")
            if os.path.exists(done_marker):
                expected = parse_done_marker(done_marker)
                for task_name, (exp_count, exp_size) in expected.items():
                    fpath = os.path.join(rank_dir, f'{task_name}.jsonl')
                    if os.path.exists(fpath):
                        actual_size = os.path.getsize(fpath)
                        if actual_size != exp_size:
                            verification_failures.append(
                                f"rank{r}/{task_name}.jsonl: expected {exp_size}B, got {actual_size}B"
                            )
                    else:
                        verification_failures.append(f"rank{r}/{task_name}.jsonl: missing")
        
        if verification_failures:
            print(f"[InMemoryRULER] WARNING: {len(verification_failures)} file verification failures")
            for vf in verification_failures[:5]:
                print(f"  {vf}")
            if len(verification_failures) > 5:
                print(f"  ... and {len(verification_failures) - 5} more")
            # Extra wait and retry verification
            time_mod.sleep(5)
        
        # Report final state
        found_count = 0
        total_task_files = 0
        for r in range(world_size):
            done_marker = os.path.join(pred_dir_base, f"rank{r}", '.done')
            rank_dir = os.path.join(pred_dir_base, f"rank{r}")
            if os.path.exists(done_marker):
                files = [f for f in os.listdir(rank_dir) if f.endswith('.jsonl')]
                found_count += 1
                total_task_files += len(files)
                if found_count <= 8 or r >= world_size - 4:  # Show first 8 and last 4
                    print(f"[InMemoryRULER] Rank {r}: {len(files)} task files")
                elif found_count == 9:
                    print(f"[InMemoryRULER] ... ({world_size - 12} more ranks) ...")
            else:
                print(f"[InMemoryRULER] WARNING: Rank {r} not completed after {max_retries * retry_delay}s")
        
        print(f"[InMemoryRULER] Total: {found_count}/{world_size} ranks completed, {total_task_files} task files")
        
        # Metric functions from eval/synthetic/constants.py - EXACT match
        def string_match_all(preds, refs):
            """For each prediction, check if ALL references are found (average across refs).
            Used for: niah, variable_tracking, common_words_extraction, freq_words_extraction"""
            score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
            return round(score, 2)
        
        def string_match_part(preds, refs):
            """For each prediction, check if ANY reference is found (max across refs).
            Used for: qa"""
            score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
            return round(score, 2)
        
        # Task type to metric mapping from eval/synthetic/constants.py
        EVAL_TASKS = {
            'niah': {'metric_fn': string_match_all},
            'variable_tracking': {'metric_fn': string_match_all},
            'common_words_extraction': {'metric_fn': string_match_all},
            'freq_words_extraction': {'metric_fn': string_match_all},
            'qa': {'metric_fn': string_match_part},
        }
        
        task_scores = {}
        task_nulls = {}
        
        for task_name in task_list:
            # Collect results from all ranks (matching call_api_distributed.py aggregation)
            all_results = []
            parse_errors = 0
            files_read = 0
            for r in range(world_size):
                temp_file = os.path.join(pred_dir_base, f"rank{r}", f'{task_name}.jsonl')
                if os.path.exists(temp_file):
                    files_read += 1
                    try:
                        with open(temp_file, 'r') as f:
                            for line_num, line in enumerate(f, 1):
                                if line.strip():
                                    try:
                                        all_results.append(json.loads(line))
                                    except json.JSONDecodeError as e:
                                        parse_errors += 1
                                        if parse_errors <= 3:
                                            print(f"[InMemoryRULER] JSON parse error rank{r}/{task_name}.jsonl line {line_num}: {e}")
                    except Exception as e:
                        print(f"[InMemoryRULER] Error reading rank{r}/{task_name}.jsonl: {e}")
            
            if not all_results:
                print(f"[InMemoryRULER] No results for task {task_name} (files_read={files_read}, parse_errors={parse_errors})")
                continue
            
            if parse_errors > 0:
                print(f"[InMemoryRULER] WARNING: {parse_errors} JSON parse errors for {task_name}")
            
            # Sort by index (matching call_api_distributed.py)
            all_results.sort(key=lambda x: x.get('index', 0))
            
            # Extract predictions and references (matching eval/evaluate.py)
            preds = [r['pred'] for r in all_results]
            refs = [r['outputs'] for r in all_results]
            
            # Count nulls (matching eval/evaluate.py)
            nulls = sum([1 for p in preds if len(p) == 0])
            task_nulls[task_name] = f'{nulls}/{len(preds)}'
            
            # Get metric function based on task type (matching eval/evaluate.py)
            config = tasks_customized.get(task_name, {})
            base_task = config.get('task', 'niah')
            task_eval_config = EVAL_TASKS.get(base_task, EVAL_TASKS['niah'])
            metric_fn = task_eval_config['metric_fn']
            
            # Compute score (matching eval/evaluate.py)
            if refs and refs[0] and refs[0][0] is not None:
                score = metric_fn(preds, refs)
            else:
                score = 0.0
            
            task_scores[task_name] = score
            
            print(f"[InMemoryRULER] Task {task_name}: {score:.2f}% (nulls: {task_nulls[task_name]}, samples: {len(preds)})")
        
        # Calculate overall accuracy (average across all tasks)
        if task_scores:
            overall_accuracy = sum(task_scores.values()) / len(task_scores)
        
        eval_elapsed = time.time() - eval_start_time
        print(f"[InMemoryRULER] Evaluation complete (took {eval_elapsed:.1f}s)")
        print(f"[InMemoryRULER] OVERALL AVERAGE: {overall_accuracy:.2f}%")
        
        # NOTE: We intentionally do NOT delete pred_dir_base here.
        # It will be cleaned up at the start of the next in-memory eval (rank 0),
        # and very old stale dirs are cleaned up proactively at start as well.
    
    # Synchronize before returning
    _distributed_barrier("cleanup_done")
    
    # Broadcast result to all ranks (so all ranks have the accuracy)
    if torch.distributed.is_initialized():
        result_tensor = torch.tensor([overall_accuracy], device=device)
        torch.distributed.broadcast(result_tensor, src=0)
        overall_accuracy = result_tensor.item()
    
    return overall_accuracy
