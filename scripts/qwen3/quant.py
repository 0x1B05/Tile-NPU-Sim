#!/usr/bin/env python3
"""Shared weight quantization helpers for the Qwen3 tools."""

import numpy as np


def quantize_rows(w):
    """Quantize each output row symmetrically to int8.

    Returns the quantized matrix and one f32 scale for each output row.
    """
    absmax = np.abs(w).max(axis=1)
    scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype("<f4")
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale
