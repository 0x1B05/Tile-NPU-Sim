#!/usr/bin/env python3
"""Tokenize a prompt and run the Qwen3 guest under sim_run."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from prompt import write_prompt
from tokenizer import Tokenizer


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
        metavar="PROMPT",
        help="write the Q3P image here instead of a temporary file",
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
    if not args.text:
        raise ValueError("prompt is empty")
    if args.max_pos is not None and args.max_pos <= 0:
        raise ValueError("--max-pos must be positive")
    tokenizer = Tokenizer(str(args.tokenizer))
    tokens = tokenizer.encode(args.text)
    if args.max_pos is not None and len(tokens) > args.max_pos:
        raise ValueError(
            f"prompt has {len(tokens)} tokens, context capacity is {args.max_pos}"
        )

    prompt_file = None
    remove_prompt = False
    try:
        if args.prompt_out is not None:
            prompt_file = args.prompt_out
        else:
            with tempfile.NamedTemporaryFile(
                prefix="qwen3-prompt-", suffix=".bin", delete=False
            ) as f:
                prompt_file = Path(f.name)
            remove_prompt = True
        write_prompt(prompt_file, tokens)
        cmd = [
            str(args.sim_run),
            "-m",
            str(args.memory_mib),
            str(args.app),
            f"{args.model}@0x81000000",
            f"{prompt_file}@0x80F00000",
        ]
        return subprocess.call(cmd)
    finally:
        if remove_prompt and prompt_file is not None:
            prompt_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
