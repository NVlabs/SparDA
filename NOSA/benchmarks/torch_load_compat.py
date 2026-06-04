# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for loading local checkpoints across PyTorch versions."""

import warnings

import torch


def torch_load_compat(path, map_location="cpu", *, allow_unsafe_fallback: bool = True):
    """Load a checkpoint across torch versions with a safe fallback.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    Some older SparDA checkpoints include objects that are rejected by the new
    safe unpickler, so we first try the safe path and then fall back to
    ``weights_only=False`` for trusted local checkpoints.
    """

    def _try_weights_only_true():
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            # Older torch versions do not support the weights_only kwarg.
            return torch.load(path, map_location=map_location)

    try:
        return _try_weights_only_true()
    except Exception as exc:
        msg = str(exc)
        looks_like_weights_only_issue = (
            "Weights only load failed" in msg
            or "weights_only" in msg
            or "WeightsUnpickler" in msg
            or "Unsupported global" in msg
        )
        if allow_unsafe_fallback and looks_like_weights_only_issue:
            warnings.warn(
                f"torch.load(weights_only=True) failed for checkpoint {path}. "
                "Falling back to weights_only=False. Only do this for trusted checkpoints.",
                RuntimeWarning,
            )
            return torch.load(path, map_location=map_location, weights_only=False)
        raise
