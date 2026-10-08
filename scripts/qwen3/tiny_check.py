#!/usr/bin/env python3
"""Tiny Qwen3 regression: guest forward under sim_run vs the numpy reference.

convert.py --tiny emits a fixed-seed random model (a q3b blob plus an f32
reference npz). This script runs the guest app on the simulator and compares
its probe stream against ref.py line by line:

  - qx0/ac0/qxa/aca carry packed integers (q8 activations, i32 accumulators)
    and must match bit-exactly;
  - the remaining probes are f32 bit patterns compared with a relative
    tolerance: summation order and the approx instructions diverge in the
    last ulps (observed drift is single-digit ulp);
  - the logits top-5 and the generated token ids must match exactly.

Usage: tiny_check.py SIM_RUN APP_BIN TINY_Q3B TINY_REF_NPZ
"""

import ast
import math
import os
import re
import struct
import subprocess
import sys

REF_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref.py")

# Guest memory map (apps/qwen3/qwen3.h): blob at 0x81000000, KV cache and
# stack top below 0xC0000000, so 1 GiB of RAM covers the tiny model.
MEMORY_MIB = "1024"
BLOB_BASE = "0x81000000"

INT_TAGS = {"qx0", "ac0", "qxa", "aca"}
FLOAT_RTOL = 1e-4
FLOAT_ATOL = 1e-6

WORD = re.compile(r"[0-9a-f]{8}")


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def probe_lines(text):
    """Parse "tag w0 w1 ..." lines into tag -> list of u32 words."""
    probes = {}
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) > 1 and WORD.fullmatch(tokens[1]):
            words = []
            for tok in tokens[1:]:
                if not WORD.fullmatch(tok):
                    break  # ref.py appends a parenthesized float rendering
                words.append(int(tok, 16))
            probes[tokens[0]] = words
    return probes


def generated_ids(text, prefix):
    for line in text.splitlines():
        if line.startswith(prefix):
            return [int(tok) for tok in line[len(prefix):].split()]
    return None


def run(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=600)
    except subprocess.TimeoutExpired:
        sys.exit(f"command timed out: {' '.join(cmd)}")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def main():
    sim_run, app_bin, blob, ref_npz = sys.argv[1:5]
    guest = run([sim_run, "--run", "-m", MEMORY_MIB, app_bin,
                 f"{blob}@{BLOB_BASE}"])
    ref = run([sys.executable, REF_PY, ref_npz])

    failures = []
    guest_probes = probe_lines(guest)
    ref_probes = probe_lines(ref)
    if set(guest_probes) != set(ref_probes):
        failures.append(
            f"probe tags differ: guest-only "
            f"{sorted(set(guest_probes) - set(ref_probes))}, ref-only "
            f"{sorted(set(ref_probes) - set(guest_probes))}")

    worst_rel, worst_where = 0.0, ""
    for tag in sorted(ref_probes):
        if tag not in guest_probes:
            continue
        g, r = guest_probes[tag], ref_probes[tag]
        if len(g) != len(r):
            failures.append(f"{tag}: {len(g)} guest words vs {len(r)} ref")
            continue
        if tag in INT_TAGS:
            if g != r:
                diff = [i for i, (a, b) in enumerate(zip(g, r)) if a != b]
                failures.append(f"{tag}: integer words differ at {diff[:4]}")
            continue
        for i, (gw, rw) in enumerate(zip(g, r)):
            a, b = f32(gw), f32(rw)
            if not math.isclose(a, b, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
                failures.append(f"{tag}[{i}]: guest {a!r} vs ref {b!r}")
            rel = abs(a - b) / max(abs(b), 1e-30)
            if rel > worst_rel:
                worst_rel, worst_where = rel, f"{tag}[{i}]"

    # the guest dumps logits at fixed ids; ref.py prints its own top5
    logit_line = next((l for l in guest.splitlines()
                       if l.startswith("logit probe: ")), None)
    top5_line = next((l for l in ref.splitlines()
                      if l.startswith("ref logits top5: ")), None)
    if logit_line is None or top5_line is None:
        failures.append("missing logit probe output")
    else:
        words = [int(w, 16) for w in logit_line[len("logit probe: "):].split()]
        top5 = ast.literal_eval(top5_line[len("ref logits top5: "):])
        if len(words) != len(top5):
            failures.append(f"logit probe: {len(words)} guest values vs "
                            f"{len(top5)} ref")
        else:
            for rank, (w, (idx, val)) in enumerate(zip(words, top5)):
                if not math.isclose(f32(w), val, rel_tol=FLOAT_RTOL,
                                    abs_tol=FLOAT_ATOL):
                    failures.append(f"logit top{rank + 1} (id {idx}): guest "
                                    f"{f32(w)!r} vs ref {val!r}")

    g_ids = generated_ids(guest, "generated: ")
    r_ids = generated_ids(ref, "ref generated: ")
    if g_ids != r_ids:
        failures.append(f"generated ids differ: guest {g_ids} vs ref {r_ids}")

    if failures:
        for f in failures:
            print(f"MISMATCH: {f}")
        sys.exit(f"tiny_check: FAIL ({len(failures)} mismatch(es))")
    print(f"tiny_check: PASS ({len(ref_probes)} probes, worst float rel err "
          f"{worst_rel:.3g} at {worst_where}, {len(r_ids)} generated tokens)")


if __name__ == "__main__":
    main()
