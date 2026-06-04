# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer parsing utilities for reasoning benchmarks.

Extracted from root LongReasoning/lr_utils.py to keep NOSA Reasoning
self-contained.
"""
from __future__ import annotations

import re
from typing import List, Optional


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} / \\fbox{...} content (simple brace matching)."""
    if not text:
        return None
    for token in ("\\boxed{", "\\fbox{"):
        idx = text.rfind(token)
        if idx == -1:
            continue
        i = idx + len(token)
        depth = 1
        out_chars: List[str] = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            out_chars.append(ch)
            i += 1
        if out_chars:
            return "".join(out_chars).strip()
    return None


_RE_FINAL_ANSWER = re.compile(
    r"(final\s*answer\s*[:：]\s*)(?P<ans>.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def extract_final_answer_line(text: str) -> Optional[str]:
    if not text:
        return None
    matches = list(_RE_FINAL_ANSWER.finditer(text))
    if not matches:
        return None
    return matches[-1].group("ans").strip()


def normalize_math_answer(ans: str) -> str:
    if ans is None:
        return ""
    s = str(ans)
    s = s.strip()
    s = s.strip("$")
    s = s.replace("\\,", "")
    s = s.replace(" ", "")
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        s = s[1:-1]
    return s


def parse_math_prediction(text: str) -> str:
    """Try boxed first, then Final Answer line, else fallback to last non-empty line."""
    if not text:
        return ""
    boxed = extract_last_boxed(text)
    if boxed:
        return normalize_math_answer(boxed)
    fa = extract_final_answer_line(text)
    if fa:
        return normalize_math_answer(fa)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    return normalize_math_answer(lines[-1])


_RE_INT = re.compile(r"(?<![-\\w])([0-9]{1,4})(?![-\\w])")


def parse_aime_prediction(text: str) -> str:
    """Extract last integer token as AIME answer (0..999 typical)."""
    if not text:
        return ""
    boxed = extract_last_boxed(text)
    if boxed:
        boxed = normalize_math_answer(boxed)
        m = _RE_INT.findall(boxed)
        if m:
            return str(int(m[-1]))
    fa = extract_final_answer_line(text)
    if fa:
        m = _RE_INT.findall(fa)
        if m:
            return str(int(m[-1]))
    m = _RE_INT.findall(text)
    if not m:
        return ""
    try:
        v = int(m[-1])
        return str(v)
    except Exception:
        return m[-1]


def strip_thinking_blocks(text: str) -> str:
    """Strip model thinking content, keeping the final answer after the last closing tag."""
    if not text:
        return ""
    s = str(text)
    s_low = s.lower()
    for close_tag in ("</think>", "</thinking>"):
        pos = s_low.rfind(close_tag)
        if pos != -1:
            tail = s[pos + len(close_tag):].strip()
            if tail:
                return tail
    for pattern in (
        re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    ):
        s = pattern.sub("", s)
    return s.strip()
