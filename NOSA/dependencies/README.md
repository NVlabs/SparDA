# Setting up Environment from Scratch

These instructions describe the environment flow we use for SparDA training and
NOSA/NOSI evaluation. We do not publish or update a prebuilt runtime image for
users. Build your own image from this repository.

The checked-in Dockerfile at `docker/Dockerfile` is the recommended unified
container recipe. It compiles local CUDA extensions from the repo during
`docker build`, but expects the repo checkout to be bind-mounted at
`/workspace/sparda` when you run the container.

```bash
cd <sparda-repo>
git submodule update --init --recursive infllmv2_cuda_impl/csrc/cutlass
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --build-arg CUDA_ARCH_LIST="8.0;9.0" \
  --build-arg FLASH_ATTN_CUDA_ARCHS="80;90" \
  -t sparda:local \
  .
```

`CUDA_ARCH_LIST` sets `TORCH_CUDA_ARCH_LIST` for local PyTorch extension builds.
The default covers A100 and H100/H200. `FLASH_ATTN_CUDA_ARCHS` controls the
upstream FlashAttention source build, using FlashAttention's integer format
(`80`, `90`, `100`, `120`). Override both for your hardware, for example
`--build-arg CUDA_ARCH_LIST="8.9"` and
`--build-arg FLASH_ATTN_CUDA_ARCHS="89"` for Ada cards.

```bash
docker run --rm --gpus all --ipc=host --network host \
  -v "$(pwd)":/workspace/sparda \
  -v /tmp:/tmp \
  -w /workspace/sparda \
  sparda:local \
  bash
```

If Docker images are cached only on the worker where they are built, save the
image to shared storage and load it on later workers:

```bash
mkdir -p <shared-path>/docker-images
docker save sparda:local -o <shared-path>/docker-images/sparda-local.tar
docker load -i <shared-path>/docker-images/sparda-local.tar
```

## Included Runtime Components

The Dockerfile installs one `/venv/sparda` environment for SparDA training,
NOSA, FullAttn, InfLLMv2, DMA, and NOSI. For vLLM, SGLang, ShadowKV, InfLLM,
and ArkVale baselines, use their official repositories or Docker images; this
release does not vendor those third-party source trees.

If you specifically need Nemo-backed upstream RULER ablations, install
`nemo-toolkit[all]` separately. The release benchmark flow uses Hugging Face
tokenizers and does not require Nemo.

## Finishing Environment Setup

After building the image, the SparDA virtual environment is already active by
default. If you open an interactive shell and need to reactivate it manually:

```bash
source /venv/sparda/bin/activate
```
