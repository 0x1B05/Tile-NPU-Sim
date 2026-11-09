#!/usr/bin/env python3
"""Run host, guest, uarch, trace, calibration, riscv-tests, and Qwen3 suites.

The build system produces test artifacts; this runner decides which artifacts
to execute. It deliberately does not register individual guest images with
xmake.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RV64IM_SRCDIR = ROOT / "tests" / "guest" / "rv64im"
NPU_SRCDIR = ROOT / "tests" / "guest" / "npu"
UARCH_BASELINE = ROOT / "tests" / "uarch" / "baseline.json"
SUITES = ("host", "guest", "uarch", "trace", "calibration", "riscv-tests", "qwen3")


class MissingPrerequisite(RuntimeError):
    pass


def path_arg(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, text=True)
    if check and result.returncode != 0:
        raise MissingPrerequisite("command failed: " + " ".join(cmd))
    return result


def xmake_build(targets: list[str]) -> None:
    try:
        run(["xmake", "build", *targets], check=True)
    except FileNotFoundError as error:
        raise MissingPrerequisite("xmake is not installed") from error


def find_artifact(name: str, explicit: str | None, build_dir: Path) -> Path:
    if explicit:
        path = path_arg(explicit, build_dir / name)
        if not path.is_file():
            raise MissingPrerequisite(f"missing artifact: {path}")
        return path

    if not build_dir.is_dir():
        raise MissingPrerequisite(f"missing build directory: {build_dir}")
    candidates = [
        path
        for path in build_dir.rglob(name)
        if path.is_file() and path.name == name and path.parent.name != ".objs"
    ]
    if not candidates:
        raise MissingPrerequisite(f"cannot find {name} under {build_dir}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def build_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    build_dir = path_arg(args.build_dir, ROOT / "build")
    guest_dir = path_arg(args.guest_dir, build_dir / "guest")
    qwen3_dir = path_arg(args.qwen3_dir, build_dir / "apps" / "qwen3")
    return build_dir, guest_dir, qwen3_dir


def rv64im_names() -> list[str]:
    names = {path.stem for path in RV64IM_SRCDIR.glob("*.c")}
    names |= {path.stem for path in RV64IM_SRCDIR.glob("*.S")}
    return sorted(names)


def uarch_image_names() -> list[str]:
    names = {path.stem for path in NPU_SRCDIR.glob("*.S")}
    names |= set(rv64im_names())
    return sorted(names)


def suite_host(args: argparse.Namespace) -> int:
    build_dir, _, _ = build_dirs(args)
    if args.build:
        xmake_build(["sim_tests"])
    sim_tests = find_artifact("sim_tests", args.sim_tests, build_dir)
    cmd = [str(sim_tests)]
    if args.filter:
        cmd += ["--filter", args.filter]
    return run(cmd).returncode


def suite_guest(args: argparse.Namespace) -> int:
    build_dir, guest_dir, _ = build_dirs(args)
    if args.build:
        xmake_build(["guest_images", "sim_guest_tests", "sim"])

    sim_guest_tests = find_artifact(
        "sim_guest_tests", args.sim_guest_tests, build_dir
    )
    simulator = find_artifact("sim", args.simulator, build_dir)

    listed = subprocess.run(
        [str(sim_guest_tests), "--list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if listed.returncode != 0:
        raise MissingPrerequisite("sim_guest_tests --list failed")
    driver_names = [
        name
        for name in listed.stdout.splitlines()
        if not args.filter or args.filter in name
    ]

    failures = 0
    if driver_names:
        if not guest_dir.is_dir():
            raise MissingPrerequisite(f"missing guest image directory: {guest_dir}")
        cmd = [str(sim_guest_tests), "--guest-dir", str(guest_dir)]
        if args.filter:
            cmd += ["--filter", args.filter]
        failures += run(cmd).returncode != 0

    image_names = [
        name for name in rv64im_names() if not args.filter or args.filter in name
    ]
    for name in image_names:
        image = guest_dir / f"{name}.bin"
        if not image.is_file():
            raise MissingPrerequisite(f"missing guest image: {image}")
        result = run([str(simulator), str(image)])
        status = "PASS" if result.returncode == 0 else f"FAIL ({result.returncode})"
        print(f"[{status}] guest/{name}")
        failures += result.returncode != 0

    if not driver_names and not image_names:
        raise MissingPrerequisite(f"no guest tests match filter: {args.filter!r}")
    return 1 if failures else 0


def uarch_stats_errors(stats: dict) -> list[str]:
    """Check that the P4 counter schema is present and internally consistent."""
    errors: list[str] = []
    required = (
        "cycles",
        "retired",
        "macs",
        "bytes",
        "unit_count",
        "unit_busy_cycles",
        "unit_utilization",
        "unit_in_flight_peak",
        "dispatch_stall_cycles",
        "dependency_stall_cycles",
        "queue_occupancy_cycles",
        "queue_peak",
        "sram",
        "memory",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        return ["stats JSON missing " + ", ".join(missing)]

    sram = stats["sram"]
    missing = [
        key
        for key in ("tile", "acc", "vec32", "bandwidth_bytes_per_cycle")
        if key not in sram
    ]
    if missing:
        return ["sram stats missing " + ", ".join(missing)]
    if "bandwidth_bytes_per_cycle" not in stats["memory"]:
        return ["memory stats missing bandwidth_bytes_per_cycle"]

    units = set(stats["unit_count"])
    for key in (
        "unit_busy_cycles",
        "unit_utilization",
        "unit_in_flight_peak",
        "dispatch_stall_cycles",
        "dependency_stall_cycles",
        "queue_occupancy_cycles",
        "queue_peak",
    ):
        if set(stats[key]) != units:
            errors.append(f"{key} does not cover the same units as unit_count")

    if sum(stats["unit_count"].values()) != stats["retired"]:
        errors.append("unit_count sum does not match retired")
    cycles = stats["cycles"]
    for unit, busy in stats["unit_busy_cycles"].items():
        if busy > cycles:
            errors.append(f"unit_busy_cycles[{unit}] exceeds cycles")
        utilization = stats["unit_utilization"].get(unit)
        if utilization is None or utilization < 0 or utilization > 1:
            errors.append(f"unit_utilization[{unit}] is outside [0, 1]")

    domains = [sram["tile"], sram["acc"], sram["vec32"]]
    if sram["accesses"] != sum(domain["accesses"] for domain in domains):
        errors.append("sram accesses do not match the domain sum")
    if sram["conflicts"] != sum(domain["conflicts"] for domain in domains):
        errors.append("sram conflicts do not match the domain sum")
    sram_bytes = sram["read_bytes"] + sram["write_bytes"]
    if sram_bytes != sum(domain["read_bytes"] + domain["write_bytes"] for domain in domains):
        errors.append("sram bytes do not match the domain sum")
    if sram_bytes > 0 and cycles > 0 and sram["bandwidth_bytes_per_cycle"] <= 0:
        errors.append("SRAM traffic has no reported bandwidth")
    memory_bytes = stats["memory"]["read_bytes"] + stats["memory"]["write_bytes"]
    if memory_bytes > 0 and stats["memory"]["bandwidth_bytes_per_cycle"] <= 0:
        errors.append("memory traffic has no reported bandwidth")
    return errors


def suite_uarch(args: argparse.Namespace) -> int:
    build_dir, guest_dir, _ = build_dirs(args)
    if args.build:
        xmake_build(["guest_images", "sim", "sim_uarch"])

    simulator = find_artifact("sim_uarch", args.simulator_uarch, build_dir)
    if args.update_uarch_baseline and args.filter:
        raise MissingPrerequisite("--update-uarch-baseline cannot be combined with --filter")

    baseline: dict = {}
    if UARCH_BASELINE.is_file():
        baseline = json.loads(UARCH_BASELINE.read_text())
    elif not args.update_uarch_baseline:
        raise MissingPrerequisite(
            f"missing uarch baseline: {UARCH_BASELINE} "
            "(rerun with --update-uarch-baseline)"
        )

    image_names = [
        name for name in uarch_image_names() if not args.filter or args.filter in name
    ]
    if not image_names:
        raise MissingPrerequisite(f"no uarch guest images match filter: {args.filter!r}")

    updates: dict[str, dict] = {}
    failures = 0
    with tempfile.TemporaryDirectory(prefix="tile-npu-uarch-") as tmp:
        tmpdir = Path(tmp)
        for name in image_names:
            image = guest_dir / f"{name}.bin"
            if not image.is_file():
                raise MissingPrerequisite(f"missing guest image: {image}")

            runs: dict[str, dict] = {}
            for mode in ("functional", "uarch"):
                json_path = tmpdir / f"{name}-{mode}.json"
                cmd = [
                    str(simulator),
                    f"--mode={mode}",
                    "--stats-json",
                    str(json_path),
                    str(image),
                ]
                result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
                if not json_path.is_file():
                    print(
                        f"[FAIL] uarch/{name} {mode}: no stats JSON\n{result.stderr}",
                        file=sys.stderr,
                    )
                    failures += 1
                    break
                runs[mode] = json.loads(json_path.read_text())
                runs[mode]["_returncode"] = result.returncode
                if result.returncode != 0:
                    print(
                        f"[FAIL] uarch/{name} {mode}: exit {result.returncode}\n"
                        f"{result.stderr}",
                        file=sys.stderr,
                    )
                    failures += 1
                    break

            if "uarch" not in runs or "functional" not in runs:
                continue

            functional = runs["functional"]
            uarch = runs["uarch"]
            errors: list[str] = []
            if functional["_returncode"] != uarch["_returncode"]:
                errors.append(
                    f"exit code differs: functional={functional['_returncode']} "
                    f"uarch={uarch['_returncode']}"
                )
            if functional["exit_code"] != uarch["exit_code"]:
                errors.append(
                    f"a0 differs: functional={functional['exit_code']} "
                    f"uarch={uarch['exit_code']}"
                )
            if functional["status"] != uarch["status"]:
                errors.append(
                    f"stop status differs: functional={functional['status']} "
                    f"uarch={uarch['status']}"
                )
            if functional["arch_digest"] != uarch["arch_digest"]:
                errors.append(
                    f"architectural state differs: functional={functional['arch_digest']} "
                    f"uarch={uarch['arch_digest']}"
                )
            errors.extend(uarch_stats_errors(uarch))

            expected = baseline.get("images", {}).get(name)
            if expected is not None and not args.update_uarch_baseline:
                for key in ("cycles", "retired", "arch_digest"):
                    if uarch.get(key) != expected.get(key):
                        errors.append(
                            f"{key} {uarch.get(key)!r} != baseline {expected.get(key)!r}"
                        )
                if uarch.get("unit_count") != expected.get("unit_count"):
                    errors.append(
                        f"unit_count {uarch.get('unit_count')!r} != "
                        f"baseline {expected.get('unit_count')!r}"
                    )
            elif not args.update_uarch_baseline:
                errors.append("no baseline entry (rerun with --update-uarch-baseline)")

            updates[name] = {
                "cycles": uarch.get("cycles"),
                "retired": uarch.get("retired"),
                "arch_digest": uarch.get("arch_digest"),
                "unit_count": uarch.get("unit_count"),
            }
            if errors:
                print(f"[FAIL] uarch/{name}: " + "; ".join(errors), file=sys.stderr)
                failures += 1
            else:
                print(
                    f"[PASS] uarch/{name} cycles={uarch.get('cycles')} "
                    f"retired={uarch.get('retired')} digest={uarch.get('arch_digest')}"
                )

    if args.update_uarch_baseline:
        UARCH_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": 1, "images": updates}
        UARCH_BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"updated uarch baseline: {UARCH_BASELINE}")

    return 1 if failures else 0


def suite_trace(args: argparse.Namespace) -> int:
    """Check that a recorded DynInst trace replays to identical timing stats."""
    build_dir, guest_dir, _ = build_dirs(args)
    if args.build:
        xmake_build(["guest_images", "sim", "sim_uarch"])

    simulator = find_artifact("sim_uarch", args.simulator_uarch, build_dir)
    image_names = [
        name for name in uarch_image_names() if not args.filter or args.filter in name
    ]
    if not image_names:
        raise MissingPrerequisite(f"no trace images match filter: {args.filter!r}")

    failures = 0
    compared = (
        "cycles",
        "retired",
        "macs",
        "bytes",
        "unit_count",
        "sram",
        "memory",
    )
    with tempfile.TemporaryDirectory(prefix="tile-npu-trace-") as tmp:
        tmpdir = Path(tmp)
        for name in image_names:
            image = guest_dir / f"{name}.bin"
            if not image.is_file():
                raise MissingPrerequisite(f"missing guest image: {image}")
            trace = tmpdir / f"{name}.utr"
            online_path = tmpdir / f"{name}-online.json"
            replay_path = tmpdir / f"{name}-replay.json"
            online_start = time.perf_counter()
            online = subprocess.run(
                [
                    str(simulator),
                    "--mode=uarch",
                    "--trace-out",
                    str(trace),
                    "--stats-json",
                    str(online_path),
                    str(image),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            online_elapsed = time.perf_counter() - online_start
            replay_start = time.perf_counter()
            replay = subprocess.run(
                [
                    str(simulator),
                    "--mode=replay",
                    "--trace-in",
                    str(trace),
                    "--stats-json",
                    str(replay_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            replay_elapsed = time.perf_counter() - replay_start
            if online.returncode != 0 or replay.returncode != 0:
                print(
                    f"[FAIL] trace/{name}: online={online.returncode} "
                    f"replay={replay.returncode}\n{online.stderr}{replay.stderr}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            if not online_path.is_file() or not replay_path.is_file():
                print(f"[FAIL] trace/{name}: missing stats JSON", file=sys.stderr)
                failures += 1
                continue
            left = json.loads(online_path.read_text())
            right = json.loads(replay_path.read_text())
            errors = [key for key in compared if left.get(key) != right.get(key)]
            if errors:
                print(
                    f"[FAIL] trace/{name}: differs in {', '.join(errors)}",
                    file=sys.stderr,
                )
                failures += 1
            else:
                trace_bytes = trace.stat().st_size
                size_text = (
                    f"{trace_bytes / 1024:.1f}KiB"
                    if trace_bytes < 1024 * 1024
                    else f"{trace_bytes / (1024 * 1024):.2f}MiB"
                )
                speedup = online_elapsed / replay_elapsed if replay_elapsed > 0 else 0.0
                compressed = ""
                zstd = shutil.which("zstd")
                if zstd is not None:
                    zst = tmpdir / f"{name}.utr.zst"
                    subprocess.run(
                        [zstd, "-q", "-f", "-19", str(trace), "-o", str(zst)],
                        cwd=ROOT,
                        check=True,
                    )
                    zstd_bytes = zst.stat().st_size
                    zstd_text = (
                        f"{zstd_bytes / 1024:.1f}KiB"
                        if zstd_bytes < 1024 * 1024
                        else f"{zstd_bytes / (1024 * 1024):.2f}MiB"
                    )
                    compressed = f" zstd={zstd_text}"
                print(
                    f"[PASS] trace/{name} cycles={left.get('cycles')} "
                    f"retired={left.get('retired')} size={size_text}{compressed} "
                    f"online={online_elapsed:.3f}s replay={replay_elapsed:.3f}s "
                    f"speedup={speedup:.2f}x"
                )
    return 1 if failures else 0


def suite_calibration(args: argparse.Namespace) -> int:
    """Smoke-test the calibration harness on the synthetic example dataset."""
    build_dir, guest_dir, _ = build_dirs(args)
    if args.build:
        xmake_build(["guest_images", "sim_uarch"])

    simulator = find_artifact("sim_uarch", args.simulator_uarch, build_dir)
    measurements = ROOT / "tests" / "uarch" / "calibration-example.json"
    parameters = ROOT / "scripts" / "uarch" / "calibration-params.json"
    if not measurements.is_file() or not parameters.is_file():
        raise MissingPrerequisite("calibration example dataset is missing")

    with tempfile.TemporaryDirectory(prefix="tile-npu-cal-suite-") as tmp:
        output = Path(tmp) / "result.json"
        result = run(
            [
                sys.executable,
                str(ROOT / "scripts" / "uarch" / "calibrate.py"),
                "--measurements",
                str(measurements),
                "--parameters",
                str(parameters),
                "--sim",
                str(simulator),
                "--image-dir",
                str(guest_dir),
                "--samples",
                "12",
                "--passes",
                "1",
                "--holdout",
                "0.25",
                "--output",
                str(output),
            ]
        )
        if result.returncode != 0 or not output.is_file():
            return 1
        data = json.loads(output.read_text())
        for section in ("train", "holdout"):
            metrics = data.get(section, {})
            if metrics.get("count", 0) < 1 or metrics.get("mape") is None:
                print(f"[FAIL] calibration/{section}: missing metrics", file=sys.stderr)
                return 1
            if metrics["mape"] > 15.0:
                print(
                    f"[FAIL] calibration/{section}: mape={metrics['mape']:.2f}%",
                    file=sys.stderr,
                )
                return 1
        print(
            "[PASS] calibration "
            f"train_mape={data['train']['mape']:.3f}% "
            f"holdout_mape={data['holdout']['mape']:.3f}% "
            f"fitted={data['fitted_overrides']}"
        )
    return 0


def suite_riscv_tests(args: argparse.Namespace) -> int:
    build_dir, guest_dir, _ = build_dirs(args)
    if args.build:
        if not (ROOT / "external" / "riscv-tests").is_dir():
            raise MissingPrerequisite(
                "external/riscv-tests is missing; run scripts/fetch_riscv_tests.sh"
            )
        xmake_build(["riscv_tests"])

    simulator = find_artifact("sim", args.simulator, build_dir)
    image_dir = guest_dir / "riscv-tests"
    images = sorted(image_dir.glob("*.bin"))
    images = [path for path in images if not args.filter or args.filter in path.name]
    if not images:
        raise MissingPrerequisite(
            f"no riscv-tests images under {image_dir}; build riscv_tests first"
        )

    failures = 0
    for image in images:
        result = run([str(simulator), str(image)])
        status = "PASS" if result.returncode == 0 else f"FAIL ({result.returncode})"
        print(f"[{status}] riscv-tests/{image.stem}")
        failures += result.returncode != 0
    return 1 if failures else 0


def suite_qwen3(args: argparse.Namespace) -> int:
    if args.filter and not any(
        part in args.filter for part in ("qwen3", "tiny", "forward")
    ):
        raise MissingPrerequisite(f"no Qwen3 test matches filter: {args.filter!r}")

    build_dir, _, app_dir = build_dirs(args)
    if args.build:
        xmake_build(["qwen3_snapshot_app", "sim"])

    simulator = find_artifact("sim", args.simulator, build_dir)
    snapshot_app = app_dir / "qwen3_snapshot.bin"
    if not snapshot_app.is_file():
        raise MissingPrerequisite(f"missing snapshot app: {snapshot_app}")

    probe = subprocess.run(
        [sys.executable, "-c", "import numpy"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise MissingPrerequisite("Qwen3 tiny regression needs Python numpy")

    blob = app_dir / "tiny.q3b"
    prompt = app_dir / "tiny_prompt.bin"
    reference = app_dir / "tiny_ref.npz"
    snapshot = app_dir / "tiny_snapshot.bin"
    convert = ROOT / "scripts" / "qwen3" / "convert.py"
    check = ROOT / "scripts" / "qwen3" / "tiny_check.py"
    app_dir.mkdir(parents=True, exist_ok=True)

    result = run(
        [
            sys.executable,
            str(convert),
            "tiny",
            "--out",
            str(blob),
            "--prompt-out",
            str(prompt),
            "--ref",
            str(reference),
        ]
    )
    if result.returncode != 0:
        return result.returncode
    return run(
        [
            sys.executable,
            str(check),
            str(simulator),
            str(snapshot_app),
            str(blob),
            str(prompt),
            str(reference),
            str(snapshot),
        ]
    ).returncode


def run_suite(args: argparse.Namespace, suite: str) -> int:
    if suite == "host":
        return suite_host(args)
    if suite == "guest":
        return suite_guest(args)
    if suite == "uarch":
        return suite_uarch(args)
    if suite == "trace":
        return suite_trace(args)
    if suite == "calibration":
        return suite_calibration(args)
    if suite == "riscv-tests":
        return suite_riscv_tests(args)
    if suite == "qwen3":
        return suite_qwen3(args)
    raise AssertionError(f"unknown suite: {suite}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite",
        nargs="?",
        choices=(*SUITES, "all"),
        default="all",
        help="suite to run (default: all)",
    )
    parser.add_argument("--filter", default="", help="substring test filter")
    parser.add_argument(
        "--build",
        action="store_true",
        help="build the artifacts needed by the selected suite first",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="for 'all', fail when an optional suite is unavailable",
    )
    parser.add_argument(
        "--update-uarch-baseline",
        action="store_true",
        help="rewrite tests/uarch/baseline.json from the current uarch runs",
    )
    parser.add_argument("--build-dir", help="xmake build directory (default: build)")
    parser.add_argument("--sim-run", dest="simulator", help="sim executable")
    parser.add_argument("--sim-uarch", dest="simulator_uarch", help="sim_uarch executable")
    parser.add_argument("--sim-tests", help="sim_tests executable")
    parser.add_argument("--sim-guest-tests", help="sim_guest_tests executable")
    parser.add_argument("--guest-dir", help="guest image directory")
    parser.add_argument("--qwen3-dir", help="Qwen3 app artifact directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = list(SUITES) if args.suite == "all" else [args.suite]
    failed = False

    for suite in selected:
        try:
            result = run_suite(args, suite)
        except MissingPrerequisite as error:
            if args.suite == "all" and not args.strict and suite not in ("host", "guest", "uarch"):
                print(f"[SKIP] {suite}: {error}", file=sys.stderr)
                continue
            print(f"error: {suite}: {error}", file=sys.stderr)
            failed = True
            continue
        if result != 0:
            print(f"[FAIL] suite {suite} (exit {result})", file=sys.stderr)
            failed = True
        else:
            print(f"[PASS] suite {suite}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
