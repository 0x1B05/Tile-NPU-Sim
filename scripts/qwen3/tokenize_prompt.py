#!/usr/bin/env python3
"""Tokenize prompt text into a Q3P runtime prompt image.

Usage:
  tokenize_prompt.py "prompt text" --out prompt.bin
"""

import argparse
import sys
from pathlib import Path

from prompt import encode_prompt, write_prompt

ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tokenize a prompt into a Q3P image loaded at 0x80F00000.",
    )
    parser.add_argument("text", metavar="TEXT", help="prompt text to tokenize")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "external/qwen3/tokenizer.json",
        metavar="TOKENIZER_JSON",
        help="tokenizer.json (default: external/qwen3/tokenizer.json under the project root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="OUT",
        help="output Q3P prompt image",
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
    tokens = encode_prompt(args.text, args.tokenizer, args.max_pos)
    size = write_prompt(args.out, tokens)
    print(f"wrote {args.out} ({len(tokens)} tokens, {size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
