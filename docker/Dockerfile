# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/.local/bin:${PATH}
ARG CUDA_ARCH_LIST=8.0\;9.0
ARG FLASH_ATTN_CUDA_ARCHS=80\;90
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}
ENV FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS}
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV MAX_JOBS=8
ENV PYTHONPATH=
ENV PYTHONNOUSERSITE=1
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV LD_LIBRARY_PATH=/venv/sparda/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv python install 3.10
RUN "$(uv python find 3.10)" -m venv --copies /venv/sparda

RUN --mount=type=bind,source=.,target=/src,readonly \
    mkdir -p /tmp/buildsrc/NOSA && \
    cp -a /src/infllmv2_cuda_impl /tmp/buildsrc/infllmv2_cuda_impl && \
    cp -a /src/NOSA/dependencies /tmp/buildsrc/NOSA/dependencies && \
    cp -a /src/NOSA/nosi /tmp/buildsrc/NOSA/nosi && \
    unset PYTHONPATH && \
    . /venv/sparda/bin/activate && \
    export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True && \
    uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0 && \
    uv pip install -r /tmp/buildsrc/NOSA/dependencies/requirements_nosa.txt && \
    python -c "import torch; print(torch.__version__, torch.__file__)" && \
    uv pip install flashinfer-python==0.5.3 --no-build-isolation && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install flash_attn==2.7.4.post1 --no-build-isolation --no-cache && \
    uv pip install /tmp/buildsrc/infllmv2_cuda_impl --no-build-isolation --no-deps && \
    uv pip install /tmp/buildsrc/NOSA/dependencies/flash-attention-nosa --no-build-isolation --no-deps && \
    uv pip install /tmp/buildsrc/NOSA/nosi --no-deps && \
    rm -rf /tmp/buildsrc

RUN . /venv/sparda/bin/activate && \
    uv pip install "wandb>=0.16.0"

RUN . /venv/sparda/bin/activate && \
    python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab words

RUN mkdir -p /workspace/sparda

ENV VIRTUAL_ENV=/venv/sparda
ENV PATH=/venv/sparda/bin:${PATH}
WORKDIR /workspace/sparda
