#!/usr/bin/env python3
"""Run a design-space sweep over the Tile-NPU uarch timing model.

The sweep file is a JSON object mapping a --uarch-set key to a list of values:

    {
      "vector.lanes": [4, 8, 16],
      "sram.tile.banks": [1, 2, 4],
      "memory.bytes_per_cycle": [16, 32, 64]
    }

Every image is run for the Cartesian product of those values. The base
configuration is a normal uarch JSON file and is applied before the sweep
overrides. Results are written as CSV (one row per image and point) and JSON
(with the full nested run record) for plotting. CSV column names are flattened
with underscores and match the columns written by --stats-csv.

Examples:

    python3 scripts/uarch/sweep.py \
        --sweep scripts/uarch/example-sweep.json \
        --csv /tmp/uarch-sweep.csv --json /tmp/uarch-sweep.json

    python3 scripts/uarch/sweep.py --images mma ecmp \
        --base /tmp/base.json --set memory.latency=8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GUEST_NPU = ROOT / "tests" / "guest" / "npu"
GUEST_RV64IM = ROOT / "tests" / "guest" / "rv64im"
DEFAULT_BUILD_DIR = ROOT / "build"


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def find_simulator(build_dir: Path) -> Path:
    candidates = sorted(build_dir.glob("*/**/sim_uarch"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit(
        f"cannot find sim_uarch under {build_dir}; build it with "
        "'xmake build sim_uarch' or pass --sim"
    )


def default_images(image_dir: Path) -> list[str]:
    names = {path.stem for path in GUEST_NPU.glob("*.S")}
    names |= {path.stem for path in GUEST_RV64IM.glob("*.c")}
    names |= {path.stem for path in GUEST_RV64IM.glob("*.S")}
    images = sorted(name for name in names if (image_dir / f"{name}.bin").is_file())
    if not images:
        raise SystemExit(f"no guest images found in {image_dir}")
    return images


def load_json_object(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"cannot read {what} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {what} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{what} must contain a JSON object")
    return value


def sweep_points(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    if not sweep:
        return [{}]
    keys: list[str] = []
    values: list[list[Any]] = []
    for key, choices in sweep.items():
        if not isinstance(choices, list) or not choices:
            raise SystemExit(f"sweep key {key!r} must map to a non-empty JSON array")
        keys.append(key)
        values.append(choices)
    if any(isinstance(choice, (dict, list)) for choices in values for choice in choices):
        raise SystemExit("sweep values must be scalars")
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}_{key}" if prefix else str(key), child, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, separators=(",", ":"))
    else:
        out[prefix] = value


def setting_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_one(
    simulator: Path,
    image: Path,
    base: Path | None,
    overrides: list[str],
    stats_dir: Path | None,
    tag: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tile-npu-sweep-") as tmp:
        stats_path = Path(tmp) / "stats.json"
        cmd = [
            str(simulator),
            "--mode=uarch",
            "--stats-json",
            str(stats_path),
        ]
        if base is not None:
            cmd += ["--uarch-config", str(base)]
        for override in overrides:
            cmd += ["--uarch-set", override]
        cmd.append(str(image))

        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if not stats_path.is_file():
            raise RuntimeError(
                f"sim_uarch produced no stats for {image.name}: {result.stderr.strip()}"
            )
        stats = json.loads(stats_path.read_text())
        if stats_dir is not None:
            stats_dir.mkdir(parents=True, exist_ok=True)
            (stats_dir / f"{tag}.json").write_text(json.dumps(stats, indent=2) + "\n")

    return {
        "image": image.stem,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        **stats,
    }


def record_flat(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if key in ("sweep", "stderr"):
            continue
        # Merge the effective configuration into the top level so its columns
        # match the config fields written by --stats-csv.
        if key == "uarch_config":
            flatten("", value, flat)
        else:
            flatten("", {key: value}, flat)
    for key, value in record.get("sweep", {}).items():
        flat[key.replace(".", "_")] = value
    return flat


def csv_fields(records: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for record in records:
        fields.update(record_flat(record))
    preferred = ["image", "returncode", "status", "exit_code", "cycles", "retired"]
    ordered = [field for field in preferred if field in fields]
    ordered += sorted(fields.difference(ordered))
    return ordered


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    import csv

    fields = csv_fields(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for record in records:
            flat = record_flat(record)
            writer.writerow({field: flat.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", help="sim_uarch executable (default: auto-detect in build/)")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR), help="build directory")
    parser.add_argument("--base", help="base uarch config JSON")
    parser.add_argument("--sweep", help="sweep JSON: key -> list of values")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="base override applied before every sweep point (repeatable)",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="guest image names without .bin (default: uarch regression images)",
    )
    parser.add_argument("--image-dir", default=str(DEFAULT_BUILD_DIR / "guest"))
    parser.add_argument("--csv", help="CSV output path")
    parser.add_argument("--json", dest="json_path", help="JSON output path")
    parser.add_argument("--stats-dir", help="keep every raw stats JSON in this directory")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel simulator runs (default: min(8, CPU count))",
    )
    parser.add_argument("--verbose", action="store_true", help="print every run")
    args = parser.parse_args()

    if not args.csv and not args.json_path:
        parser.error("at least one of --csv or --json is required")

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    simulator = (
        Path(args.sim).expanduser().resolve() if args.sim else find_simulator(build_dir)
    )
    if not simulator.is_file():
        raise SystemExit(f"simulator not found: {simulator}")

    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = ROOT / image_dir
    image_names = args.images or default_images(image_dir)
    images = []
    for name in image_names:
        path = image_dir / (name if name.endswith(".bin") else f"{name}.bin")
        if not path.is_file():
            raise SystemExit(f"missing guest image: {path}")
        images.append(path)

    base = Path(args.base).expanduser().resolve() if args.base else None
    if base is not None and not base.is_file():
        raise SystemExit(f"missing base config: {base}")

    sweep = load_json_object(Path(args.sweep), "sweep file") if args.sweep else {}
    points = sweep_points(sweep)
    stats_dir = Path(args.stats_dir) if args.stats_dir else None

    jobs: list[tuple[str, Path, list[str], str, dict[str, Any]]] = []
    for index, point in enumerate(points):
        overrides = list(args.set)
        overrides += [f"{key}={setting_text(value)}" for key, value in point.items()]
        label = ",".join(f"{key}={setting_text(value)}" for key, value in point.items()) or "base"
        for image in images:
            tag = f"p{index:04d}-{image.stem}"
            jobs.append((tag, image, overrides, label, point))

    eprint(
        f"sweeping {len(points)} point(s) x {len(images)} image(s) = {len(jobs)} run(s) "
        f"with {simulator}"
    )

    records: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_map = {
            pool.submit(run_one, simulator, image, base, overrides, stats_dir, tag): (
                tag,
                point,
                label,
            )
            for tag, image, overrides, label, point in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            tag, point, label = future_map[future]
            try:
                record = future.result()
            except Exception as exc:
                eprint(f"[FAIL] {tag}: {exc}")
                failures += 1
                continue
            record["sweep"] = point
            records.append(record)
            if record["returncode"] != 0:
                failures += 1
                eprint(f"[FAIL] {tag} ({label}) exit {record['returncode']}")
            elif args.verbose:
                eprint(
                    f"[ok] {tag} ({label}) cycles={record.get('cycles')} "
                    f"retired={record.get('retired')}"
                )

    records.sort(key=lambda record: (record["image"], json.dumps(record["sweep"], sort_keys=True)))

    if args.csv:
        path = Path(args.csv)
        write_csv(path, records)
        eprint(f"wrote {path} ({len(records)} row(s))")
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "simulator": str(simulator),
            "base_config": str(base) if base is not None else None,
            "sweep": sweep,
            "records": records,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        eprint(f"wrote {path} ({len(records)} record(s))")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
