#!/usr/bin/env python3
"""Optional zstd wrapper for UARCH traces.

The simulator reads the compact raw trace directly. This helper is for
archiving or transferring large traces and for reporting compression ratios:

    python3 scripts/uarch/trace_zstd.py stats run.utr
    python3 scripts/uarch/trace_zstd.py compress run.utr
    python3 scripts/uarch/trace_zstd.py decompress run.utr.zst

Compression uses the system `zstd` executable; it is intentionally optional so
the simulator has no mandatory zstd dependency.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def zstd_path() -> str:
    path = shutil.which("zstd")
    if path is None:
        raise SystemExit("zstd is not installed; install it or use the raw trace")
    return path


def compressed_path(source: Path) -> Path:
    return source.with_suffix(source.suffix + ".zst")


def compress(source: Path, output: Path | None = None) -> Path:
    target = output or compressed_path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [zstd_path(), "-q", "-f", "-19", str(source), "-o", str(target)],
        cwd=ROOT,
        check=True,
    )
    return target


def decompress(source: Path, output: Path | None = None) -> Path:
    target = output or source.with_suffix("")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [zstd_path(), "-q", "-f", "-d", str(source), "-o", str(target)],
        cwd=ROOT,
        check=True,
    )
    return target


def stats(source: Path) -> int:
    if not source.is_file():
        raise SystemExit(f"missing trace: {source}")
    raw_size = source.stat().st_size
    with tempfile.TemporaryDirectory(prefix="tile-npu-zstd-") as tmp:
        target = Path(tmp) / compressed_path(source).name
        compress(source, target)
        compressed_size = target.stat().st_size
    ratio = raw_size / compressed_size if compressed_size else 0.0
    print(
        f"raw={raw_size / (1024 * 1024):.2f}MiB "
        f"zstd={compressed_size / (1024 * 1024):.2f}MiB "
        f"ratio={ratio:.2f}x"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stats", "compress", "decompress"))
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, help="output path")
    args = parser.parse_args()

    source = args.trace.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"missing trace: {source}")

    if args.command == "stats":
        return stats(source)
    if args.command == "compress":
        print(compress(source, args.out))
        return 0
    print(decompress(source, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
