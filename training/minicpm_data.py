# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset and DataLoader helpers shared by training scripts."""

from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ProLong-64K dataset from Princeton NLP (https://huggingface.co/datasets/princeton-nlp/prolong-data-64K)
# NOTE: The original ProLong dataset is tokenized for Llama-3. MiniCPM training expects a MiniCPM-tokenized copy.
PROLONG_DATASET_ID = "princeton-nlp/prolong-data-64K"
PROLONG_MINICPM_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "prolong-64k-minicpm")  # MiniCPM-tokenized
# Default dataset shortcut. Prepare the MiniCPM-tokenized copy before training.
DEFAULT_DATA_PATH = "prolong-64k"


def has_mds_index(data_path):
    """Return True when a local Mosaic MDS dataset has at least one indexed stream."""
    if not os.path.isdir(data_path):
        return False
    try:
        entries = os.listdir(data_path)
    except OSError:
        return False
    for entry in entries:
        split_dir = os.path.join(data_path, entry)
        if os.path.isdir(split_dir) and os.path.exists(os.path.join(split_dir, "index.json")):
            return True
    return False


def _infer_local_world_size() -> int:
    """Best-effort inference of processes-per-node for launchers that don't set LOCAL_WORLD_SIZE.

    MosaicML Streaming uses LOCAL_WORLD_SIZE to elect a single local leader per node for /dev/shm
    coordination. If it defaults to 1 under some multi-GPU launchers,
    every rank may think it's the local leader and collide on shared memory names.
    """
    # Prefer CUDA_VISIBLE_DEVICES when present (works even if CUDA init is constrained).
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        # Handle formats like "0,1,2,3" (ignore empty items).
        ids = [x for x in cvd.split(",") if x.strip() != ""]
        if len(ids) > 0:
            return len(ids)

    # Fallback to device_count (may initialize CUDA).
    try:
        n = torch.cuda.device_count()
        if n > 0:
            return int(n)
    except Exception:
        pass

    return 1


def _normalize_dist_env_for_streaming() -> None:
    """Ensure env vars needed by Mosaic Streaming's local-leader election are set sanely."""
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")

    # LOCAL_WORLD_SIZE is frequently missing under some launchers.
    if world_size > 1:
        lws_raw = os.environ.get("LOCAL_WORLD_SIZE")
        try:
            lws = int(lws_raw) if lws_raw is not None and str(lws_raw).strip() != "" else None
        except ValueError:
            lws = None
        if lws is None or lws <= 1:
            os.environ["LOCAL_WORLD_SIZE"] = str(_infer_local_world_size())


# Must run before importing `streaming` so it picks up corrected LOCAL_WORLD_SIZE.
_normalize_dist_env_for_streaming()

try:
    # Installed by the release Docker image.
    from streaming import StreamingDataset, Stream
except ImportError:
    StreamingDataset = None
    Stream = None

if Stream is not None:
    class SafeStream(Stream):
        """Safe if multiple processes try to decompress the same shard."""

        def _decompress_shard_part(self, zip_info, zip_filename, raw_filename, compression):
            unique_extension = "." + str(os.getenv("SPARDA_RUN_ID", "local")) + "-" + str(os.getpid())
            super()._decompress_shard_part(zip_info, zip_filename, raw_filename + unique_extension, compression)
            os.rename(raw_filename + unique_extension, raw_filename)
else:
    SafeStream = None

def is_prolong_dataset(data_path):
    """Check if data_path refers to the ProLong-64K dataset."""
    # Check if it's the HuggingFace dataset ID or shortcut
    if data_path == PROLONG_DATASET_ID or data_path in ("prolong-64k", "prolong"):
        return True
    # Check if it's the local path for ProLong data
    if "prolong" in data_path.lower() and "64k" in data_path.lower():
        return True
    return False


def get_prolong_data_path(data_path):
    """Get the actual local path for ProLong dataset.
    
    For MiniCPM training, 'prolong-64k' always refers to the prepared
    MiniCPM-tokenized version.
    """
    # For prolong-64k shortcuts, always use MiniCPM-tokenized version
    if data_path == PROLONG_DATASET_ID or data_path in ("prolong-64k", "prolong"):
        return PROLONG_MINICPM_DATA_PATH
    return data_path


def download_prolong_if_needed(data_path, verbose=False):
    """
    Download ProLong-64K dataset from HuggingFace if not available locally.
    
    The dataset uses MDS (Mosaic Data Streaming) format with multiple domain subdirectories.
    Dataset size: ~50GB+ (multiple domains).
    
    Args:
        data_path: Local directory for the dataset
        verbose: Print progress info
    
    Returns:
        Path to the dataset directory
    """
    if os.path.isdir(data_path):
        if has_mds_index(data_path):
            if verbose:
                print(f"ProLong-64K dataset found at {data_path} (MDS format)")
            return data_path
        if os.listdir(data_path):
            if verbose:
                print(
                    f"Data directory {data_path} exists but is missing MDS indices required by StreamingDataset. "
                    "Refreshing from Hugging Face..."
                )
        else:
            if verbose:
                print(f"Data directory {data_path} is empty. Downloading ProLong-64K...")
    else:
        if verbose:
            print(f"Data directory {data_path} not found. Downloading ProLong-64K...")
            print(f"Dataset: {PROLONG_DATASET_ID}")
            print("Note: This is a large dataset (~50GB+). First download may take a while.")
    
    try:
        from huggingface_hub import snapshot_download
        if verbose:
            print("Using huggingface_hub to download...")
        os.makedirs(data_path, exist_ok=True)
        snapshot_download(
            repo_id=PROLONG_DATASET_ID,
            repo_type="dataset",
            local_dir=data_path,
            local_dir_use_symlinks=False  # Ensure we get actual files
        )
        if verbose:
            print(f"Download completed: {data_path}")
    except ImportError:
        if verbose:
            print("huggingface_hub not found. Falling back to git clone...")
        try:
            import subprocess
            repo_url = f"https://huggingface.co/datasets/{PROLONG_DATASET_ID}"
            subprocess.run(["git", "clone", repo_url, data_path], check=True)
            if verbose:
                print("Git download completed.")
        except Exception as e:
            print(f"Failed to download data: {e}")
            return None
    except Exception as e:
        print(f"Failed to download data: {e}")
        print(f"Please manually download: git clone https://huggingface.co/datasets/{PROLONG_DATASET_ID} {data_path}")
        return None
    
    return data_path


def cleanup_streaming_shm(verbose=False):
    """Clean up stale shared memory from previous streaming dataset runs.
    
    Should be called by local rank 0 on each node BEFORE creating StreamingDataset,
    with a proper barrier after to ensure cleanup completes before other ranks proceed.
    """
    import glob as glob_module
    shm_patterns = ['/dev/shm/*_locals*', '/dev/shm/*_barrier*', '/dev/shm/*_cache*']
    cleaned = 0
    for pattern in shm_patterns:
        for f in glob_module.glob(pattern):
            try:
                os.remove(f)
                cleaned += 1
            except (OSError, PermissionError):
                pass
    if verbose and cleaned > 0:
        print(f"Cleaned up {cleaned} stale shared memory files")


def get_prolong_dataloader(
    data_path,
    batch_size,
    seq_len,
    epoch_samples,
    verbose=False,
    vocab_size=None,
    rank=0,
    world_size=1,
):
    """
    Get dataloader for ProLong-64K dataset using StreamingDataset.
    
    IMPORTANT: Before calling this function, ensure:
    1. Local rank 0 on each node calls cleanup_streaming_shm()
    2. A barrier (accelerator.wait_for_everyone()) is called
    Then all ranks can safely call this function together.
    
    Args:
        vocab_size: If provided, clamp all token IDs to [0, vocab_size-1] to prevent
                   embedding lookup errors.
        rank: Current process rank (unused, StreamingDataset handles internally)
        world_size: Total number of processes (unused, StreamingDataset handles internally)
    """
    if StreamingDataset is None or Stream is None:
        raise RuntimeError(
            "mosaicml-streaming is not installed. Rebuild the SparDA Docker "
            "image from docker/Dockerfile or install the matching dependency "
            "stack in your local environment."
        )
    
    # ProLong 64K data mixing recipe
    domain_proportions = {
        "thestackv1_concat_by_repo-65536": 0.3,
        "book-65536": 0.3,
        "fineweb-edu": 0.1,
        "fineweb-2023-50": 0.1,
        "stackexchange": 0.04,
        "dolmawiki": 0.04,
        "tuluv2": 0.03,
        "arxiv": 0.03,
        "openwebmath": 0.03,
        "textbooks": 0.03,
    }
    
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"ProLong data path {data_path} does not exist.")
    
    # Build streams with proportions matching ProLong's recipe
    streams = []
    for domain, proportion in domain_proportions.items():
        domain_path = os.path.join(data_path, domain)
        if os.path.isdir(domain_path) and os.path.exists(os.path.join(domain_path, "index.json")):
            streams.append(SafeStream(local=domain_path, remote=None, split=None, proportion=proportion))
        elif verbose:
            print(f"Warning: Domain {domain} not found at {domain_path}, skipping")
    
    if not streams:
        raise RuntimeError(f"No valid MDS streams found in {data_path}")
    
    if verbose:
        print(f"Creating StreamingDataset with {len(streams)} streams, epoch_size={epoch_samples}")
    
    dataset = StreamingDataset(
        streams=streams,
        shuffle=True,
        shuffle_algo='py1e',
        shuffle_seed=42,
        batch_size=batch_size,
        epoch_size=epoch_samples,
        predownload=None,
        cache_limit=None,
    )
    
    # Track if we've already warned about out-of-range tokens
    warned_oob = [False]
    
    def collate_fn(batch):
        input_ids_list = []
        for item in batch:
            ids = torch.tensor(item['input_ids'][:seq_len]).long()
            if vocab_size is not None:
                if not warned_oob[0] and ((ids >= vocab_size).any() or (ids < 0).any()):
                    print(f"WARNING: Out-of-range token IDs found, clamping to vocab_size={vocab_size}")
                    warned_oob[0] = True
                ids = ids.clamp(0, vocab_size - 1)
            input_ids_list.append(ids)
        return {"input_ids": torch.stack(input_ids_list)}
    
    # StreamingDataset generally works best with num_workers=0 due to shared-memory coordination.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )


def get_dataloader(
    data_path,
    batch_size,
    seq_len,
    epoch_samples,
    verbose=False,
    rank=0,
    world_size=1,
):
    """
    Dataloader factory for ProLong-format training data.
    
    Args:
        data_path: Path to dataset or dataset identifier
            - "prolong-64k" or "princeton-nlp/prolong-data-64K": Use ProLong dataset
            - Local path: Use as-is (ProLong MDS format expected)
        batch_size: Batch size per GPU
        seq_len: Sequence length
        epoch_samples: Total samples per epoch
        verbose: Print debug info
        rank: Current process rank
        world_size: Total number of processes
    
    Returns:
        DataLoader
    """
    actual_path = get_prolong_data_path(data_path)
    if verbose:
        print(f"Using ProLong dataset from {actual_path}")
    return get_prolong_dataloader(
        actual_path, batch_size, seq_len, epoch_samples,
        verbose=verbose, rank=rank, world_size=world_size,
    )


def infinite_dataloader(dataloader, accelerator=None, start_batch=0):
    """
    Infinite iterator over a dataloader that cycles through epochs.
    
    Properly handles distributed samplers by calling set_epoch() for each new epoch.
    """
    epoch = 0
    batches_to_skip = max(int(start_batch or 0), 0)
    skip_announced = False
    while True:
        # Set epoch for sampler to ensure proper shuffling (if supported)
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            if accelerator is not None and accelerator.is_main_process and epoch > 0:
                print(f"  Starting epoch {epoch + 1} (cycling through data)")
        
        for batch in dataloader:
            if batches_to_skip > 0:
                if accelerator is not None and accelerator.is_main_process and not skip_announced:
                    print(f"  Skipping {batches_to_skip} already-consumed microbatches for exact resume")
                    skip_announced = True
                batches_to_skip -= 1
                if accelerator is not None and accelerator.is_main_process and skip_announced and batches_to_skip == 0:
                    print("  Resume data stream aligned to checkpoint")
                continue
            yield batch
        
        epoch += 1
