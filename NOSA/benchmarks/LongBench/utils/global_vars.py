# This file is copied/adapted from LongBench (https://github.com/THUDM/LongBench).
# Copyright (c) 2023 THU-KEG & Zhipu AI.
# SPDX-License-Identifier: MIT


_GLOBAL_ARGS = None

def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

def get_args():
    """Return arguments."""
    return _GLOBAL_ARGS