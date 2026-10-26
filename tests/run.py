#!/usr/bin/env python3
"""Run host, guest, riscv-tests, and Qwen3 test suites.

The build system produces test artifacts; this runner decides which artifacts
to execute. It deliberately does not register individual guest images with
xmake.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RV64IM_SRCDIR = ROOT / "tests" / "guest" / "rv64im"
SUITES = ("host", "guest", "riscv-tests", "qwen3")


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
        xmake_build(["guest_images", "sim_guest_tests", "sim_run"])

    sim_guest_tests = find_artifact(
        "sim_guest_tests", args.sim_guest_tests, build_dir
    )
    sim_run = find_artifact("sim_run", args.sim_run, build_dir)

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
        result = run([str(sim_run), str(image)])
        status = "PASS" if result.returncode == 0 else f"FAIL ({result.returncode})"
        print(f"[{status}] guest/{name}")
        failures += result.returncode != 0

    if not driver_names and not image_names:
        raise MissingPrerequisite(f"no guest tests match filter: {args.filter!r}")
    return 1 if failures else 0


def suite_riscv_tests(args: argparse.Namespace) -> int:
    build_dir, guest_dir, _ = build_dirs(args)
    if args.build:
        if not (ROOT / "external" / "riscv-tests").is_dir():
            raise MissingPrerequisite(
                "external/riscv-tests is missing; run scripts/fetch_riscv_tests.sh"
            )
        xmake_build(["riscv_tests"])

    sim_run = find_artifact("sim_run", args.sim_run, build_dir)
    image_dir = guest_dir / "riscv-tests"
    images = sorted(image_dir.glob("*.bin"))
    images = [path for path in images if not args.filter or args.filter in path.name]
    if not images:
        raise MissingPrerequisite(
            f"no riscv-tests images under {image_dir}; build riscv_tests first"
        )

    failures = 0
    for image in images:
        result = run([str(sim_run), str(image)])
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
        xmake_build(["qwen3_snapshot_app", "sim_run"])

    sim_run = find_artifact("sim_run", args.sim_run, build_dir)
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
            str(sim_run),
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
    parser.add_argument("--build-dir", help="xmake build directory (default: build)")
    parser.add_argument("--sim-run", help="sim_run executable")
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
            if args.suite == "all" and not args.strict and suite not in ("host", "guest"):
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
