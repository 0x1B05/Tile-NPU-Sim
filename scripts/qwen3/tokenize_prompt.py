#!/usr/bin/env python3
"""Tokenize prompt text into a Q3P1 runtime prompt image.

Usage:
  tokenize_prompt.py "prompt text" --out prompt.bin
"""

import argparse
import sys
from pathlib import Path

from prompt import write_prompt
from tokenizer import Tokenizer


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tokenize a prompt into a Q3P1 image loaded at 0x80F00000.",
    )
    parser.add_argument("text", metavar="TEXT", help="prompt text to tokenize")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("external/qwen3/tokenizer.json"),
        metavar="TOKENIZER_JSON",
        help="tokenizer.json (default: external/qwen3/tokenizer.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="OUT",
        help="output Q3P1 prompt image",
    )
    parser.add_argument(
        "--max-pos",
        type=int,
        metavar="N",
        help="reject prompts longer than N tokens",
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
    size = write_prompt(args.out, tokens)
    print(f"wrote {args.out} ({len(tokens)} tokens, {size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
