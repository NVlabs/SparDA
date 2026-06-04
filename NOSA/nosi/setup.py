# This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
# Copyright (c) 2026 THUNLP.
# SPDX-License-Identifier: MIT

from setuptools import setup, find_packages

setup(
    name="nosi",
    version="0.0.1",
    package_data={
        "nosi": [
            "flash_cache_engine/*.cpp",
            "flash_cache_engine/*.cu",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
)
