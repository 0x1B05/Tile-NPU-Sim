#!/usr/bin/env python3
"""Fit Uarch timing parameters against measured workload cycles.

Measurements are a JSON object with a ``workloads`` array. Each workload
contains a name, an image name (or path), and the measured/RTL cycle count:

    {
      "schema": 1,
      "workloads": [
        {"name": "mma_k8", "image": "mma", "cycles": 649},
        {"name": "dma_stride", "image": "mem", "cycles": 138}
      ]
    }

The parameter file maps each ``--uarch-set`` key to candidate values, either a
list or a {"min": ..., "max": ..., "step": ...} object. The fit uses a
deterministic random search followed by coordinate refinement, reports MAPE,
maximum relative error, and Spearman rank correlation on train and hold-out
sets. It does not require RTL at this stage; replace the example measurements
with RTL counters when they are available.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = ROOT / "scripts" / "uarch" / "calibration-params.json"


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def find_simulator(build_dir: Path) -> Path:
    candidates = sorted(build_dir.glob("*/**/sim_uarch"))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    raise SystemExit(
        f"cannot find sim_uarch under {build_dir}; build it with 'xmake build sim_uarch'"
    )


def load_object(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"cannot read {what} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {what} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{what} must contain a JSON object")
    return value


def parameter_values(spec: Any, key: str) -> list[int]:
    if isinstance(spec, list):
        values = spec
    elif isinstance(spec, dict):
        if "values" in spec:
            values = spec["values"]
        else:
            if "min" not in spec or "max" not in spec:
                raise SystemExit(f"parameter {key!r} needs values or min/max")
            step = int(spec.get("step", 1))
            if step <= 0:
                raise SystemExit(f"parameter {key!r} has non-positive step")
            values = list(range(int(spec["min"]), int(spec["max"]) + 1, step))
    else:
        raise SystemExit(f"invalid specification for parameter {key!r}")

    parsed = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"parameter {key!r} values must be integers")
        parsed.append(value)
    parsed = sorted(set(parsed))
    if not parsed:
        raise SystemExit(f"parameter {key!r} has no candidate values")
    return parsed


def load_measurements(path: Path, image_dir: Path) -> list[dict[str, Any]]:
    data = load_object(path, "measurements")
    raw = data.get("workloads")
    if not isinstance(raw, list) or not raw:
        raise SystemExit("measurements must contain a non-empty workloads array")
    workloads = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SystemExit(f"measurement {index} is not an object")
        name = item.get("name")
        image = item.get("image")
        cycles = item.get("cycles")
        if not isinstance(name, str) or not isinstance(image, str):
            raise SystemExit(f"measurement {index} needs string name and image")
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise SystemExit(f"measurement {name!r} needs a positive integer cycles")
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = image_dir / (image if image.endswith(".bin") else image + ".bin")
        if not image_path.is_file():
            raise SystemExit(f"missing image for {name!r}: {image_path}")
        workloads.append(
            {
                "name": name,
                "image": image,
                "image_path": image_path,
                "cycles": cycles,
            }
        )
    return workloads


def run_one(
    simulator: Path,
    workload: dict[str, Any],
    base: Path | None,
    overrides: list[str],
    timeout: int,
) -> tuple[int | None, str | None]:
    with tempfile.TemporaryDirectory(prefix="tile-npu-cal-") as tmp:
        stats_path = Path(tmp) / "stats.json"
        cmd = [str(simulator), "--mode=uarch"]
        if base is not None:
            cmd += ["--uarch-config", str(base)]
        for override in overrides:
            cmd += ["--uarch-set", override]
        # Keep --stats-json after the vector --uarch-set option so CLI11 does
        # not consume the positional image as another override.
        cmd += ["--stats-json", str(stats_path), str(workload["image_path"])]
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout after {timeout}s"
        if result.returncode != 0 or not stats_path.is_file():
            return None, result.stderr.strip() or f"exit {result.returncode}"
        try:
            stats = json.loads(stats_path.read_text())
        except json.JSONDecodeError as exc:
            return None, f"invalid stats JSON: {exc}"
        return int(stats["cycles"]), None


def metrics(measured: list[int], predicted: list[int]) -> dict[str, Any]:
    if not measured:
        return {"count": 0, "mape": None, "max_error": None, "rank_correlation": None}
    errors = [
        abs(p - m) / m
        for m, p in zip(measured, predicted, strict=True)
    ]
    return {
        "count": len(measured),
        "mape": 100.0 * sum(errors) / len(errors),
        "max_error": 100.0 * max(errors),
        "rank_correlation": spearman(measured, predicted),
    }


def rank_values(values: list[int]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    return ranks


def spearman(left: list[int], right: list[int]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks = rank_values(left)
    right_ranks = rank_values(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def candidate_points(
    keys: list[str], values: list[list[int]], limit: int, rng: random.Random
) -> list[tuple[int, ...]]:
    total = math.prod(len(options) for options in values)
    if total <= limit:
        return list(itertools.product(*values))
    seen: set[tuple[int, ...]] = set()
    points: list[tuple[int, ...]] = []
    while len(points) < limit:
        point = tuple(rng.choice(options) for options in values)
        if point not in seen:
            seen.add(point)
            points.append(point)
    return points


def evaluate(
    simulator: Path,
    workloads: list[dict[str, Any]],
    base: Path | None,
    keys: list[str],
    candidate: tuple[int, ...],
    image_dir: Path,
    jobs: int,
    timeout: int,
    cache: dict[tuple[int, ...], dict[str, Any]],
) -> dict[str, Any]:
    del image_dir  # Images are resolved before the search.
    if candidate in cache:
        return cache[candidate]
    overrides = [f"{key}={value}" for key, value in zip(keys, candidate, strict=True)]
    measured: list[int] = []
    predicted: list[int] = []
    per_workload: list[dict[str, Any]] = []

    def task(workload: dict[str, Any]) -> tuple[dict[str, Any], int | None, str | None]:
        cycles, error = run_one(simulator, workload, base, overrides, timeout)
        return workload, cycles, error

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(task, workload) for workload in workloads]
        for future in futures:
            workload, cycles, error = future.result()
            if cycles is None:
                result = {"error": f"{workload['name']}: {error}"}
                cache[candidate] = result
                return result
            measured.append(workload["cycles"])
            predicted.append(cycles)
            per_workload.append(
                {
                    "name": workload["name"],
                    "image": workload["image"],
                    "measured": workload["cycles"],
                    "predicted": cycles,
                    "error_pct": 100.0 * abs(cycles - workload["cycles"]) / workload["cycles"],
                }
            )

    result = {
        "metrics": metrics(measured, predicted),
        "workloads": per_workload,
    }
    cache[candidate] = result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, help="JSON workload measurements")
    parser.add_argument("--parameters", default=str(DEFAULT_PARAMS), help="parameter ranges JSON")
    parser.add_argument("--base", help="base uarch config JSON")
    parser.add_argument("--sim", help="sim_uarch executable (default: auto-detect in build/)")
    parser.add_argument("--build-dir", default=str(ROOT / "build"), help="build directory")
    parser.add_argument("--image-dir", default=str(ROOT / "build" / "guest"), help="guest image directory")
    parser.add_argument("--samples", type=int, default=24, help="random search points")
    parser.add_argument("--passes", type=int, default=2, help="coordinate refinement passes")
    parser.add_argument("--seed", type=int, default=0, help="deterministic search seed")
    parser.add_argument("--holdout", type=float, default=0.25, help="hold-out fraction in [0, 0.9]")
    parser.add_argument("--jobs", type=int, default=min(8, __import__("os").cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=120, help="per-run timeout in seconds")
    parser.add_argument("--output", help="write the full result as JSON")
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.passes < 0:
        raise SystemExit("--passes must be non-negative")
    if not 0.0 <= args.holdout <= 0.9:
        raise SystemExit("--holdout must be in [0, 0.9]")

    build_dir = Path(args.build_dir).expanduser().resolve()
    simulator = (
        Path(args.sim).expanduser().resolve()
        if args.sim
        else find_simulator(build_dir)
    )
    if not simulator.is_file():
        raise SystemExit(f"simulator not found: {simulator}")

    image_dir = Path(args.image_dir).expanduser().resolve()
    measurements = load_measurements(Path(args.measurements).expanduser().resolve(), image_dir)
    parameter_data = load_object(Path(args.parameters).expanduser().resolve(), "parameters")
    keys = list(parameter_data)
    if not keys:
        raise SystemExit("parameter file is empty")
    values = [parameter_values(parameter_data[key], key) for key in keys]
    base = Path(args.base).expanduser().resolve() if args.base else None
    if base is not None and not base.is_file():
        raise SystemExit(f"missing base config: {base}")

    rng = random.Random(args.seed)
    indices = list(range(len(measurements)))
    rng.shuffle(indices)
    test_count = (
        max(1, min(len(indices) - 1, round(len(indices) * args.holdout)))
        if args.holdout > 0 and len(indices) > 1
        else 0
    )
    test_indices = set(indices[:test_count])
    train = [workload for index, workload in enumerate(measurements) if index not in test_indices]
    test = [workload for index, workload in enumerate(measurements) if index in test_indices]
    if not train:
        raise SystemExit("no training workloads after hold-out split")

    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    candidates = candidate_points(keys, values, args.samples, rng)
    best: tuple[int, ...] | None = None
    best_result: dict[str, Any] | None = None

    def consider(candidate: tuple[int, ...]) -> bool:
        nonlocal best, best_result
        result = evaluate(
            simulator, train, base, keys, candidate, image_dir, args.jobs, args.timeout, cache
        )
        if "error" in result:
            eprint(f"[skip] {' '.join(f'{k}={v}' for k, v in zip(keys, candidate))}: {result['error']}")
            return False
        current = (result["metrics"]["mape"], result["metrics"]["max_error"])
        if best is None or current < (best_result["metrics"]["mape"], best_result["metrics"]["max_error"]):
            best = candidate
            best_result = result
            return True
        return False

    for index, candidate in enumerate(candidates, start=1):
        if consider(candidate):
            eprint(
                f"[search {index}/{len(candidates)}] mape={best_result['metrics']['mape']:.3f}% "
                f"max={best_result['metrics']['max_error']:.3f}%"
            )
    if best is None or best_result is None:
        raise SystemExit("no candidate produced valid timing stats")

    # Coordinate refinement around the best random-search point.
    for pass_index in range(args.passes):
        improved = False
        for position, key in enumerate(keys):
            for value in values[position]:
                candidate = list(best)
                candidate[position] = value
                candidate_tuple = tuple(candidate)
                if candidate_tuple == best:
                    continue
                if consider(candidate_tuple):
                    improved = True
                    eprint(
                        f"[refine {pass_index + 1}] {key}={value} "
                        f"mape={best_result['metrics']['mape']:.3f}%"
                    )
        if not improved:
            break

    evaluated = len(cache)
    cache.clear()  # The training cache covers train workloads only.
    all_result = evaluate(
        simulator, measurements, base, keys, best, image_dir, args.jobs, args.timeout, cache
    )
    if "error" in all_result:
        raise SystemExit(f"final evaluation failed: {all_result['error']}")

    fitted = {key: value for key, value in zip(keys, best, strict=True)}
    split_by_name = {workload["name"]: "train" for workload in train}
    split_by_name.update({workload["name"]: "test" for workload in test})
    records = [
        {**record, "split": split_by_name[record["name"]]}
        for record in all_result["workloads"]
    ]
    train_records = [record for record in records if record["split"] == "train"]
    test_records = [record for record in records if record["split"] == "test"]
    result = {
        "schema": 1,
        "fitted": fitted,
        "fitted_overrides": [f"{key}={value}" for key, value in fitted.items()],
        "train": metrics(
            [record["measured"] for record in train_records],
            [record["predicted"] for record in train_records],
        ),
        "holdout": metrics(
            [record["measured"] for record in test_records],
            [record["predicted"] for record in test_records],
        ),
        "workloads": records,
        "search": {
            "seed": args.seed,
            "samples": args.samples,
            "passes": args.passes,
            "evaluated": evaluated,
        },
    }

    def format_metric(label: str, value: dict[str, Any]) -> str:
        if value["count"] == 0:
            return f"{label}: n=0"
        rank = "n/a" if value["rank_correlation"] is None else f"{value['rank_correlation']:.3f}"
        return (
            f"{label}: n={value['count']} mape={value['mape']:.3f}% "
            f"max={value['max_error']:.3f}% rank={rank}"
        )

    eprint("fitted: " + " ".join(result["fitted_overrides"]))
    eprint(format_metric("train", result["train"]))
    eprint(format_metric("holdout", result["holdout"]))
    for record in result["workloads"]:
        eprint(
            f"  {record['split']:5s} {record['name']:12s} "
            f"measured={record['measured']} predicted={record['predicted']} "
            f"error={record['error_pct']:.2f}%"
        )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        eprint(f"wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
