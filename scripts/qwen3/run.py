#!/usr/bin/env python3
"""Tokenize a prompt and run the Qwen3 guest under sim_run."""

import argparse
import subprocess
import sys
from pathlib import Path

from prompt import encode_prompt, write_prompt


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tokenize a prompt and run apps/qwen3 under sim_run.",
    )
    parser.add_argument("text", metavar="TEXT", help="prompt text")
    parser.add_argument(
        "--sim-run",
        type=Path,
        required=True,
        metavar="SIM_RUN",
        help="sim_run executable",
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    tokens = encode_prompt(args.text, args.tokenizer, args.max_pos)
    write_prompt(args.prompt_out, tokens)
    cmd = [
        str(args.sim_run),
        "-m",
        str(args.memory_mib),
        str(args.app),
        f"{args.model}@0x81000000",
        f"{args.prompt_out}@0x80F00000",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
