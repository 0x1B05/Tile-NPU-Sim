"""Q3B2 blob writer and atomic output helpers."""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np

MAGIC = b"Q3B2"
OUTPUT_MODE = 0o644


def check_output(path):
    if path.exists() and path.is_dir():
        raise ValueError(f"output path is a directory: {path}")
    if not path.parent.is_dir():
        raise ValueError(f"output directory does not exist: {path.parent}")


class Blob:
    def __init__(self, arch):
        self.arch = arch  # dict of u32 arch fields
        self.sections = []  # (name, bytes)

    def add(self, name, data):
        assert len(name) <= 8, name
        self.sections.append((name, data))

    def write(self, path):
        check_output(path)
        header_size = 12 * 4
        directory_size = len(self.sections) * 16
        offset = header_size + directory_size
        entries = []
        payload = bytearray()
        for name, data in self.sections:
            pad = (-offset) % 16
            payload += b"\0" * pad
            offset += pad
            entries.append((name, offset, len(data)))
            payload += data
            offset += len(data)
        a = self.arch
        header = struct.pack(
            "<4s11I",
            MAGIC,
            a["n_layers"],
            a["hidden"],
            a["inter"],
            a["n_q_heads"],
            a["n_kv_heads"],
            a["head_dim"],
            a["vocab"],
            a["max_pos"],
            np.frombuffer(np.float32(a["eps"]).tobytes(), "<u4")[0],
            len(self.sections),
            0,
        )
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
                f.write(header)
                for name, off, size in entries:
                    f.write(struct.pack("<8sII", name.encode(), off, size))
                f.write(payload)
            os.chmod(tmp_path, OUTPUT_MODE)
            os.replace(tmp_path, path)
        except BaseException:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
        return offset


def write_npz(path, arrays):
    check_output(path)
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
            np.savez(f, **arrays)
        os.chmod(tmp_path, OUTPUT_MODE)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
