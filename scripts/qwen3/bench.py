#!/usr/bin/env python3
"""Benchmark Qwen3 prefill and decode phases from the guest serial stream.

The guest prints one prompt line, runs prefill, then prints "generated:" and
one space-separated token per decode step. This script timestamps the serial
bytes as they arrive, so no simulator instrumentation is needed.

Run with a no-probe build for accurate numbers:
  xmake f --qwen3probes=n && xmake build qwen3_app
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from prompt import encode_prompt, write_prompt
from tokenizer import Tokenizer

BLOB_BASE = "0x81000000"
PROMPT_BASE = "0x80F00000"
PROBES_WARNING = (
    "warning: PROBE records detected; prefill/TTFT include "
    "qwen3_probe_last_prompt. Rebuild with `xmake f --qwen3probes=n`."
)


@dataclass(frozen=True)
class RunMetrics:
    load_s: float
    prefill_s: float
    ttft_s: float
    ttft_compute_s: float
    decode_s: float
    decode_steps: int
    decode_ms_per_token: float
    decode_tokens_per_s: float
    total_s: float
    prompt_tokens: int
    output_tokens: int
    probes: bool

    def as_dict(self):
        return asdict(self)


def run_once(cmd):
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert proc.stdout is not None
    fd = proc.stdout.fileno()

    line = bytearray()
    token_bytes = bytearray()
    token_end_times = []
    tokens = []
    mode = "lines"
    in_token = False
    probes = False
    prompt_count = None
    t_prompt = None
    t_generated = None
    t_first_token = None

    while True:
        chunk = os.read(fd, 1)
        if not chunk:
            break
        now = time.perf_counter()
        byte = chunk[0]

        if mode == "lines":
            line.extend(chunk)
            if line.startswith(b"generated: "):
                mode = "generated"
                t_generated = now
                continue
            if byte == 0x0A:
                if line.startswith(b"PROBE "):
                    probes = True
                elif line.startswith(b"prompt tokens:"):
                    t_prompt = now
                    prompt_count = len(line.split()) - 2
                line.clear()
            continue

        if 0x30 <= byte <= 0x39:
            if not in_token and t_first_token is None:
                t_first_token = now
            token_bytes.append(byte)
            in_token = True
        elif byte in (0x20, 0x09, 0x0A, 0x0D) and in_token:
            tokens.append(int(token_bytes))
            token_end_times.append(now)
            token_bytes.clear()
            in_token = False

    end = time.perf_counter()
    if in_token:
        tokens.append(int(token_bytes))
        token_end_times.append(end)

    returncode = proc.wait()
    stderr = proc.stderr.read().decode(errors="replace") if proc.stderr is not None else ""
    if returncode != 0:
        raise SystemExit(f"bench: sim_run exited with {returncode}: {stderr.strip()}")
    if t_prompt is None or t_generated is None or t_first_token is None or not tokens:
        raise SystemExit("bench: incomplete serial output")

    decode_steps = len(tokens) - 1
    decode_s = token_end_times[-1] - token_end_times[0] if decode_steps else 0.0
    decode_ms = decode_s * 1000.0 / decode_steps if decode_steps else 0.0
    decode_rate = decode_steps / decode_s if decode_s > 0 else 0.0

    return RunMetrics(
        load_s=t_prompt - start,
        prefill_s=t_generated - t_prompt,
        ttft_s=t_first_token - start,
        ttft_compute_s=t_first_token - t_prompt,
        decode_s=decode_s,
        decode_steps=decode_steps,
        decode_ms_per_token=decode_ms,
        decode_tokens_per_s=decode_rate,
        total_s=end - start,
        prompt_tokens=prompt_count,
        output_tokens=len(tokens),
        probes=probes,
    )


def print_report(runs):
    print(
        "run  load_s  prefill_s  ttft_s  ttft_compute_s  decode_s  "
        "ms/token  tok/s  total_s"
    )
    for i, r in enumerate(runs):
        print(
            f"{i:>3}  {r.load_s:6.3f}  {r.prefill_s:9.3f}  {r.ttft_s:6.3f}  "
            f"{r.ttft_compute_s:14.3f}  {r.decode_s:8.3f}  "
            f"{r.decode_ms_per_token:8.1f}  {r.decode_tokens_per_s:5.2f}  "
            f"{r.total_s:7.3f}"
        )

    keys = [
        "load_s",
        "prefill_s",
        "ttft_s",
        "ttft_compute_s",
        "decode_s",
        "decode_ms_per_token",
        "decode_tokens_per_s",
        "total_s",
    ]
    med = {key: statistics.median(getattr(r, key) for r in runs) for key in keys}
    print(
        f"med  {med['load_s']:6.3f}  {med['prefill_s']:9.3f}  {med['ttft_s']:6.3f}  "
        f"{med['ttft_compute_s']:14.3f}  {med['decode_s']:8.3f}  "
        f"{med['decode_ms_per_token']:8.1f}  {med['decode_tokens_per_s']:5.2f}  "
        f"{med['total_s']:7.3f}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure Qwen3 prefill, TTFT, and decode time from the guest serial stream.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Use a no-probe build for accurate timings:\n"
            "  xmake f --qwen3probes=n && xmake build qwen3_app"
        ),
    )
    parser.add_argument("text", nargs="?", help="prompt text (or use --prompt-image)")
    parser.add_argument("--prompt-image", type=Path, help="existing Q3P prompt image")
    parser.add_argument("--prompt-out", type=Path, default=Path("build/apps/qwen3/bench_prompt.bin"))
    parser.add_argument("--tokenizer", type=Path, default=Path("external/qwen3/tokenizer.json"))
    parser.add_argument("--max-pos", type=int)
    parser.add_argument("--sim-run", type=Path, required=True, help="sim_run executable")
    parser.add_argument("--app", type=Path, required=True, help="qwen3 guest binary")
    parser.add_argument(
        "--model", type=Path, default=Path("build/apps/qwen3/model.q3b"), help="Q3B model blob"
    )
    parser.add_argument("--memory-mib", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu", type=int, help="pin sim_run with taskset -c CPU")
    parser.add_argument("--json", type=Path, help="write the median and per-run metrics as JSON")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.text) == bool(args.prompt_image):
        parser.error("provide either prompt text or --prompt-image")
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup must be >= 0")

    if args.prompt_image is not None:
        prompt = args.prompt_image
    else:
        tokenizer = Tokenizer(str(args.tokenizer))
        tokens = encode_prompt(args.text, args.tokenizer, args.max_pos, tokenizer=tokenizer)
        prompt = args.prompt_out
        write_prompt(prompt, tokens)

    cmd = [
        str(args.sim_run),
        "-m",
        str(args.memory_mib),
        str(args.app),
        f"{args.model}@{BLOB_BASE}",
        f"{prompt}@{PROMPT_BASE}",
    ]
    if args.cpu is not None:
        cmd = ["taskset", "-c", str(args.cpu), *cmd]

    for _ in range(args.warmup):
        run_once(cmd)

    runs = [run_once(cmd) for _ in range(args.runs)]
    if any(run.probes for run in runs):
        print(PROBES_WARNING, file=sys.stderr)

    print(f"prompt tokens: {runs[0].prompt_tokens}")
    print(f"runs: {args.runs} (warmup {args.warmup})")
    print_report(runs)

    if args.json is not None:
        keys = [
            "load_s",
            "prefill_s",
            "ttft_s",
            "ttft_compute_s",
            "decode_s",
            "decode_ms_per_token",
            "decode_tokens_per_s",
            "total_s",
        ]
        median = {key: statistics.median(getattr(r, key) for r in runs) for key in keys}
        median["prompt_tokens"] = runs[0].prompt_tokens
        median["output_tokens"] = runs[0].output_tokens
        args.json.write_text(
            json.dumps({"runs": [run.as_dict() for run in runs], "median": median}, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        sys.exit(1)
