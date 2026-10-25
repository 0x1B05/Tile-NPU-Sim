"""Q3P1 runtime prompt image writer and atomic output helpers."""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from blob import check_output

MAGIC = b"Q3P1"
HEADER_SIZE = 16
OUTPUT_MODE = 0o644


def prompt_bytes(tokens):
    tokens = list(tokens)
    if not tokens:
        raise ValueError("prompt is empty")
    if any(not 0 <= int(token) <= 0xFFFFFFFF for token in tokens):
        raise ValueError("prompt token id out of u32 range")
    header = struct.pack("<4sII4x", MAGIC, len(tokens), 0)
    assert len(header) == HEADER_SIZE
    return header + np.asarray(tokens, "<u4").tobytes()


def write_prompt(path, tokens):
    check_output(path)
    data = prompt_bytes(tokens)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            f.write(data)
        os.chmod(tmp_path, OUTPUT_MODE)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return len(data)
