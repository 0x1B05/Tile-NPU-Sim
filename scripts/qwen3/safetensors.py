"""Minimal BF16 safetensors reader used by the Qwen3 converter."""

import json
import struct

import numpy as np


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4")


class SafeTensors:
    """Lazily materializes tensors; peak memory stays near one tensor."""

    def __init__(self, path):
        self.path = path
        with path.open("rb") as f:
            raw_size = f.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"{path}: truncated safetensors header")
            header_size = struct.unpack("<Q", raw_size)[0]
            self.header = json.loads(f.read(header_size))
            self.data_start = 8 + header_size

    def get(self, name):
        info = self.header.get(name)
        if info is None:
            raise ValueError(f"{self.path}: tensor not found: {name}")
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: unexpected dtype {info['dtype']}")
        begin, end = info["data_offsets"]
        with self.path.open("rb") as f:
            f.seek(self.data_start + begin)
            raw = f.read(end - begin)
        return bf16_to_f32(raw).reshape(info["shape"])
