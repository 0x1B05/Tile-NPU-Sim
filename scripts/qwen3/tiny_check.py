#!/usr/bin/env python3
"""Tiny Qwen3 regression: guest snapshot and generated tokens vs NumPy.

The snapshot guest writes named f32 records into a fixed RAM region. sim_run
dumps that region after the run, and this script compares the records and
generated token ids with ref.py.

Usage: tiny_check.py SIM_RUN APP_BIN TINY_Q3B TINY_PROMPT TINY_REF_NPZ SNAPSHOT_BIN
"""

import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REF_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref.py")

MEMORY_MIB = "1024"
PROMPT_BASE = "0x80F00000"
BLOB_BASE = "0x81000000"
SNAPSHOT_BASE = 0x80B00000
SNAPSHOT_SIZE = 4 << 20
SNAPSHOT_MAGIC = 0x00504E53
SNAPSHOT_KINDS = {
    0: ("<f4", "f32"),
    1: ("i1", "i8"),
    2: ("<i4", "i32"),
}
FLOAT_RTOL = 1e-4
FLOAT_ATOL = 1e-6


def parse_snapshot(path):
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError(f"{path}: truncated snapshot header")
    magic, version, used, records = struct.unpack_from("<IIII", data, 0)
    if magic != SNAPSHOT_MAGIC:
        raise ValueError(f"{path}: bad snapshot magic")
    if version != 1:
        raise ValueError(f"{path}: unsupported snapshot version {version}")
    if used > len(data):
        raise ValueError(f"{path}: snapshot used {used} > file {len(data)}")

    out = []
    offset = 16
    for _ in range(records):
        offset = (offset + 15) & ~15
        if offset + 32 > used:
            raise ValueError(f"{path}: truncated snapshot record")
        name, kind, count, nbytes, _, _, _ = struct.unpack_from(
            "<8sIIIIII", data, offset
        )
        if kind not in SNAPSHOT_KINDS:
            raise ValueError(f"{path}: bad snapshot kind {kind}")
        dtype, kind_name = SNAPSHOT_KINDS[kind]
        payload = data[offset + 32 : offset + 32 + nbytes]
        if len(payload) != nbytes:
            raise ValueError(f"{path}: truncated snapshot payload")
        values = np.frombuffer(payload, dtype=dtype, count=count).copy()
        if values.nbytes != nbytes:
            raise ValueError(f"{path}: snapshot size mismatch for {name!r}")
        label = name.split(b"\0", 1)[0].decode()
        out.append((label, kind_name, values))
        offset += 32 + nbytes
    return out


def generated_ids(text, prefix):
    for line in text.splitlines():
        if line.startswith(prefix):
            return [int(tok) for tok in line[len(prefix) :].split()]
    return None


def run(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        sys.exit(f"command timed out: {' '.join(cmd)}")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def main():
    sim_run, app_bin, blob, prompt, ref_npz, snapshot_bin = sys.argv[1:7]
    snapshot_path = Path(snapshot_bin)
    ref_snapshot = str(snapshot_path.with_suffix(".ref.npz"))

    guest = run([
        sim_run,
        "-m",
        MEMORY_MIB,
        "--dump-range",
        f"{snapshot_path}@{SNAPSHOT_BASE:#x}:{SNAPSHOT_SIZE:#x}",
        app_bin,
        f"{blob}@{BLOB_BASE}",
        f"{prompt}@{PROMPT_BASE}",
    ])
    ref_text = run([sys.executable, REF_PY, ref_npz, "32", ref_snapshot])

    failures = []
    try:
        guest_records = parse_snapshot(snapshot_path)
    except (OSError, ValueError) as exc:
        sys.exit(f"tiny_check: {exc}")
    ref_arrays = np.load(ref_snapshot)
    ref_names = list(ref_arrays.keys())
    guest_names = [name for name, _, _ in guest_records]

    if guest_names != ref_names:
        failures.append(f"snapshot records: guest {guest_names} vs ref {ref_names}")

    worst_rel, worst_where = 0.0, ""
    for name, kind, values in guest_records:
        if name not in ref_arrays:
            continue
        expected = np.asarray(ref_arrays[name], dtype=np.float32).reshape(-1)
        if values.size != expected.size:
            failures.append(
                f"{name}: {values.size} guest elements vs {expected.size} ref"
            )
            continue
        if kind != "f32":
            if not np.array_equal(values, expected):
                failures.append(f"{name}: integer values differ")
            continue
        for i, (a, b) in enumerate(zip(values, expected)):
            if not math.isclose(float(a), float(b), rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
                failures.append(f"{name}[{i}]: guest {a!r} vs ref {b!r}")
            rel = abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)
            if rel > worst_rel:
                worst_rel, worst_where = rel, f"{name}[{i}]"

    g_ids = generated_ids(guest, "generated: ")
    r_ids = generated_ids(ref_text, "ref generated: ")
    if g_ids is None or r_ids is None:
        failures.append(f"generated ids missing: guest {g_ids} vs ref {r_ids}")
        r_ids = r_ids or []
    elif g_ids != r_ids:
        failures.append(f"generated ids differ: guest {g_ids} vs ref {r_ids}")

    if failures:
        for failure in failures:
            print(f"MISMATCH: {failure}")
        sys.exit(f"tiny_check: FAIL ({len(failures)} mismatch(es))")
    print(
        f"tiny_check: PASS ({len(ref_names)} snapshots, worst float rel err "
        f"{worst_rel:.3g} at {worst_where}, {len(r_ids)} generated tokens)"
    )


if __name__ == "__main__":
    main()
