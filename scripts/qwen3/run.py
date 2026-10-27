#!/usr/bin/env python3
"""Tokenize a prompt and run the Qwen3 guest under the simulator."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from prompt import encode_prompt, write_prompt
from tokenizer import Tokenizer

EOS_IDS = {151643, 151645}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tokenize a prompt and run apps/qwen3 under the simulator.",
    )
    parser.add_argument("text", metavar="TEXT", help="prompt text")
    parser.add_argument(
        "--sim-run",
        type=Path,
        required=True,
        metavar="SIM_RUN",
        help="simulator executable",
    )
    parser.add_argument(
        "--app",
        type=Path,
        required=True,
        metavar="APP_BIN",
        help="qwen3 guest binary",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("build/apps/qwen3/model.q3b"),
        metavar="MODEL_Q3B",
        help="Q3B model blob (default: build/apps/qwen3/model.q3b)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("external/qwen3/tokenizer.json"),
        metavar="TOKENIZER_JSON",
        help="tokenizer.json (default: external/qwen3/tokenizer.json)",
    )
    parser.add_argument(
        "--prompt-out",
        type=Path,
        default=Path("build/apps/qwen3/prompt.bin"),
        metavar="PROMPT",
        help="Q3P image path (default: build/apps/qwen3/prompt.bin)",
    )
    parser.add_argument(
        "--max-pos",
        type=int,
        metavar="N",
        help="reject prompts longer than N tokens",
    )
    parser.add_argument(
        "--memory-mib",
        default="1024",
        metavar="MIB",
        help="simulator RAM size (default: 1024)",
    )
    return parser


def generated_ids(output):
    match = re.search(r"^generated:\s*(.*)$", output, re.MULTILINE)
    if not match:
        return None
    try:
        return [int(token) for token in match.group(1).split()]
    except ValueError:
        return None


def run_guest(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    assert proc.stdout is not None
    chunks = []
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        chunks.append(chunk)
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    return proc.wait(), b"".join(chunks).decode(errors="replace")


def main(argv=None):
    args = build_parser().parse_args(argv)
    tokenizer = Tokenizer(str(args.tokenizer))
    tokens = encode_prompt(args.text, args.tokenizer, args.max_pos, tokenizer=tokenizer)
    write_prompt(args.prompt_out, tokens)
    cmd = [
        str(args.sim_run),
        "-m",
        str(args.memory_mib),
        str(args.app),
        f"{args.model}@0x81000000",
        f"{args.prompt_out}@0x80F00000",
    ]
    rc, output = run_guest(cmd)
    if rc != 0:
        return rc
    ids = generated_ids(output)
    if ids is not None:
        text = tokenizer.decode([token for token in ids if token not in EOS_IDS])
        print(f"generated text: {text}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
