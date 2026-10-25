#!/usr/bin/env python3
"""Tiny Qwen3 regression: guest forward under sim_run vs the numpy reference.

convert.py tiny emits a fixed-seed random model (a q3b blob plus an f32
reference npz and a Q3P1 prompt image). This script runs the guest app on the
simulator and compares ordered records emitted by guest and ref.py:

  PROBE <name> [index] <kind> <count> <word>...

  - f32 records are compared by value with a relative tolerance; summation
    order and the approx instructions diverge in the last ulps;
  - i8/i32 records (packed q8 activations, i32 accumulators) must match
    bit-exactly;
  - order, duplicate records, kinds, and word counts must match;
  - generated token ids must match exactly.

Usage: tiny_check.py SIM_RUN APP_BIN TINY_Q3B TINY_PROMPT TINY_REF_NPZ
"""

import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass

REF_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref.py")

# Guest memory map (apps/qwen3/include/qwen3.h): prompt at 0x80F00000, blob at
# 0x81000000, KV cache and stack top below 0xC0000000, so 1 GiB of RAM covers
# the tiny model.
MEMORY_MIB = "1024"
PROMPT_BASE = "0x80F00000"
BLOB_BASE = "0x81000000"

FLOAT_RTOL = 1e-4
FLOAT_ATOL = 1e-6
PROBE_KINDS = {"f32", "i8", "i32"}


@dataclass(frozen=True)
class ProbeRecord:
    name: str
    index: int | None
    kind: str
    words: list[int]


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def parse_probes(text):
    """Parse the explicit PROBE record protocol."""
    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith("PROBE "):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"line {lineno}: malformed PROBE record")
        name = fields[1]
        pos = 2
        index = None
        if fields[pos] not in PROBE_KINDS:
            try:
                index = int(fields[pos])
            except ValueError as exc:
                raise ValueError(f"line {lineno}: invalid probe index") from exc
            pos += 1
        if pos + 2 > len(fields):
            raise ValueError(f"line {lineno}: missing probe kind or count")
        kind = fields[pos]
        pos += 1
        if kind not in PROBE_KINDS:
            raise ValueError(f"line {lineno}: invalid probe kind {kind!r}")
        try:
            count = int(fields[pos])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: invalid probe count") from exc
        pos += 1
        raw_words = fields[pos:]
        if len(raw_words) != count:
            raise ValueError(
                f"line {lineno}: PROBE count {count} but {len(raw_words)} words"
            )
        words = []
        for raw in raw_words:
            if len(raw) != 8:
                raise ValueError(f"line {lineno}: invalid probe word {raw!r}")
            try:
                words.append(int(raw, 16))
            except ValueError as exc:
                raise ValueError(f"line {lineno}: invalid probe word {raw!r}") from exc
        records.append(ProbeRecord(name, index, kind, words))
    return records


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
    sim_run, app_bin, blob, prompt, ref_npz = sys.argv[1:6]
    guest = run([
        sim_run,
        "-m",
        MEMORY_MIB,
        app_bin,
        f"{blob}@{BLOB_BASE}",
        f"{prompt}@{PROMPT_BASE}",
    ])
    ref = run([sys.executable, REF_PY, ref_npz])

    failures = []
    try:
        guest_probes = parse_probes(guest)
        ref_probes = parse_probes(ref)
    except ValueError as exc:
        sys.exit(f"tiny_check: {exc}")

    for label, records in (("guest", guest_probes), ("ref", ref_probes)):
        seen = set()
        for record in records:
            key = (record.name, record.index)
            if key in seen:
                failures.append(f"{label}: duplicate probe {key}")
            seen.add(key)

    if len(guest_probes) != len(ref_probes):
        failures.append(
            f"probe count: guest {len(guest_probes)} vs ref {len(ref_probes)}"
        )
    if guest_probes and ref_probes:
        guest_names = [(r.name, r.index) for r in guest_probes]
        ref_names = [(r.name, r.index) for r in ref_probes]
        if guest_names != ref_names:
            failures.append(
                f"probe order: guest {guest_names} vs ref {ref_names}"
            )

    worst_rel, worst_where = 0.0, ""
    for i, (g, r) in enumerate(zip(guest_probes, ref_probes)):
        label = f"{g.name}[{g.index}]" if g.index is not None else g.name
        if g.name != r.name or g.index != r.index:
            failures.append(
                f"record {i}: guest {g.name}/{g.index} vs ref {r.name}/{r.index}"
            )
            continue
        if g.kind != r.kind:
            failures.append(f"{label}: kind {g.kind} vs {r.kind}")
            continue
        if len(g.words) != len(r.words):
            failures.append(
                f"{label}: {len(g.words)} guest words vs {len(r.words)} ref"
            )
            continue
        if g.kind != "f32":
            if g.words != r.words:
                diff = [j for j, (a, b) in enumerate(zip(g.words, r.words)) if a != b]
                failures.append(f"{label}: integer words differ at {diff[:4]}")
            continue
        for j, (gw, rw) in enumerate(zip(g.words, r.words)):
            a, b = f32(gw), f32(rw)
            if not math.isclose(a, b, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
                failures.append(f"{label}[{j}]: guest {a!r} vs ref {b!r}")
            rel = abs(a - b) / max(abs(b), 1e-30)
            if rel > worst_rel:
                worst_rel, worst_where = rel, f"{label}[{j}]"

    g_ids = generated_ids(guest, "generated: ")
    r_ids = generated_ids(ref, "ref generated: ")
    if g_ids is None or r_ids is None:
        failures.append(f"generated ids missing: guest {g_ids} vs ref {r_ids}")
        r_ids = r_ids or []
    elif g_ids != r_ids:
        failures.append(f"generated ids differ: guest {g_ids} vs ref {r_ids}")

    if failures:
        for f in failures:
            print(f"MISMATCH: {f}")
        sys.exit(f"tiny_check: FAIL ({len(failures)} mismatch(es))")
    print(
        f"tiny_check: PASS ({len(ref_probes)} probes, worst float rel err "
        f"{worst_rel:.3g} at {worst_where}, {len(r_ids)} generated tokens)"
    )


if __name__ == "__main__":
    main()
