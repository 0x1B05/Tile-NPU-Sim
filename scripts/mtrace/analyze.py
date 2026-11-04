#!/usr/bin/env python3
"""Summarize and query an MTRB binary MTRACE file.

Examples:
    python3 scripts/mtrace/analyze.py mtrace.mtrb
    python3 scripts/mtrace/analyze.py mtrace.mtrb --by op --top 20
    python3 scripts/mtrace/analyze.py mtrace.mtrb --filter-op 'mma|vmma' --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

try:
    from .reader import Endpoint, Event, Memory, TraceReader
except ImportError:  # running as a script
    from reader import Endpoint, Event, Memory, TraceReader


def _event_bytes(event: Event) -> int:
    return sum(ep.bytes for ep in event.src) + sum(ep.bytes for ep in event.dst)


def _endpoint_key(endpoint: Endpoint) -> str:
    return f"{endpoint.kind}:{endpoint.part}" if endpoint.part else endpoint.kind


def _memory_key(memory: Memory) -> str:
    return f"{memory.pattern}:{memory.region}"


def _group_key(event: Event, by: str) -> str:
    if by == "op":
        return event.op
    if by == "type":
        return event.type
    if by == "pc":
        return f"0x{event.pc:016x}"
    return str(event.instr)


def _filter_events(events: Iterable[Event], args: argparse.Namespace) -> Iterator[Event]:
    op_re = re.compile(args.filter_op) if args.filter_op else None
    emitted = 0
    for event in events:
        if op_re is not None and not op_re.search(event.op):
            continue
        if args.filter_type and event.type != args.filter_type:
            continue
        if event.pc < args.pc_start:
            continue
        if args.pc_end is not None and event.pc >= args.pc_end:
            continue
        if event.instr < args.instr_start:
            continue
        if args.instr_end is not None and event.instr >= args.instr_end:
            continue
        if args.limit is not None and emitted >= args.limit:
            return
        yield event
        emitted += 1


def _print_table(rows: list[tuple[str, int, int, int]], top: int) -> None:
    selected = rows if top <= 0 else rows[:top]
    print(f"{'key':<30} {'count':>10} {'bytes':>14} {'macs':>14}")
    for key, count, size, macs in selected:
        print(f"{key:<30} {count:>10} {size:>14} {macs:>14}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze an MTRB binary MTRACE file.")
    parser.add_argument("path", type=Path, help="mtrace.mtrb")
    parser.add_argument(
        "--by",
        choices=("op", "type", "pc", "instr", "endpoint", "memory"),
        help="group events instead of printing the default summary",
    )
    parser.add_argument("--top", type=int, default=20, help="rows to print (0 = all)")
    parser.add_argument(
        "--sort", choices=("count", "bytes", "macs"), default="count", help="row order"
    )
    parser.add_argument("--csv", action="store_true", help="emit the selected table as CSV")
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    parser.add_argument("--filter-op", help="regular expression matched against op")
    parser.add_argument("--filter-type", choices=("move", "compute"))
    parser.add_argument("--pc-start", type=lambda text: int(text, 0), default=0)
    parser.add_argument("--pc-end", type=lambda text: int(text, 0), default=None)
    parser.add_argument("--instr-start", type=int, default=0)
    parser.add_argument("--instr-end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="stop after N matched events")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.csv and args.by is None:
        args.by = "op"

    try:
        with TraceReader(args.path) as trace:
            type_counts: Counter[str] = Counter()
            op_counts: Counter[str] = Counter()
            op_bytes: Counter[str] = Counter()
            op_macs: Counter[str] = Counter()
            kind_counts: Counter[str] = Counter()
            kind_bytes: Counter[str] = Counter()
            pattern_counts: Counter[str] = Counter()
            region_counts: Counter[str] = Counter()
            groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
            instructions: set[int] = set()
            pcs: set[int] = set()
            total_events = 0
            total_bytes = 0
            total_macs = 0

            for event in _filter_events(trace, args):
                total_events += 1
                instructions.add(event.instr)
                pcs.add(event.pc)
                event_bytes = _event_bytes(event)
                total_bytes += event_bytes
                total_macs += event.macs
                type_counts[event.type] += 1
                op_counts[event.op] += 1
                op_bytes[event.op] += event_bytes
                op_macs[event.op] += event.macs

                for endpoint in event.src + event.dst:
                    kind_counts[endpoint.kind] += 1
                    kind_bytes[endpoint.kind] += endpoint.bytes
                if event.mem is not None:
                    pattern_counts[event.mem.pattern] += 1
                    region_counts[event.mem.region] += 1

                if args.by in ("op", "type", "pc", "instr"):
                    group = groups[_group_key(event, args.by)]
                    group[0] += 1
                    group[1] += event_bytes
                    group[2] += event.macs
                elif args.by == "endpoint":
                    for endpoint in event.src + event.dst:
                        group = groups[_endpoint_key(endpoint)]
                        group[0] += 1
                        group[1] += endpoint.bytes
                elif args.by == "memory" and event.mem is not None:
                    group = groups[_memory_key(event.mem)]
                    group[0] += 1
                    group[1] += event_bytes
                    group[2] += event.macs

            rows = [
                (key, values[0], values[1], values[2]) for key, values in groups.items()
            ]
            sort_index = {"count": 1, "bytes": 2, "macs": 3}[args.sort]
            rows.sort(key=lambda row: (row[sort_index], row[0]), reverse=True)

            if args.csv:
                writer = csv.writer(sys.stdout)
                writer.writerow(["key", "count", "bytes", "macs", "avg_bytes"])
                for key, count, size, macs in (rows if args.top <= 0 else rows[: args.top]):
                    writer.writerow([key, count, size, macs, f"{size / count:.2f}"])
                return 0

            if args.json:
                payload = {
                    "file": str(args.path),
                    "format": {"name": trace.header.magic, "version": trace.header.version},
                    "events": total_events,
                    "types": dict(type_counts),
                    "endpoint_bytes": total_bytes,
                    "bytes_by_kind": dict(kind_bytes),
                    "endpoint_counts": dict(kind_counts),
                    "memory_patterns": dict(pattern_counts),
                    "memory_regions": dict(region_counts),
                    "macs": total_macs,
                    "unique_instructions": len(instructions),
                    "unique_pcs": len(pcs),
                }
                if args.by:
                    payload["rows"] = [
                        {"key": key, "count": count, "bytes": size, "macs": macs}
                        for key, count, size, macs in rows
                    ]
                print(json.dumps(payload, indent=2))
                return 0

            if args.by:
                _print_table(rows, args.top)
                return 0

            print(f"file: {args.path}")
            print(f"format: {trace.header.magic} v{trace.header.version}")
            print(
                f"events: {total_events} "
                f"(move {type_counts['move']}, compute {type_counts['compute']})"
            )
            print(f"instructions: {len(instructions)}, pcs: {len(pcs)}")
            print(f"endpoint bytes: {total_bytes}")
            for kind in ("mem", "tile", "acc", "vec32", "imm"):
                if kind_counts[kind]:
                    print(f"  {kind:<6} {kind_counts[kind]:>10} endpoints  {kind_bytes[kind]:>14} bytes")
            if pattern_counts:
                patterns = ", ".join(
                    f"{name} {count}" for name, count in sorted(pattern_counts.items())
                )
                print(f"memory patterns: {patterns}")
            if region_counts:
                regions = ", ".join(
                    f"{name} {count}" for name, count in sorted(region_counts.items())
                )
                print(f"memory regions: {regions}")
            print(f"macs: {total_macs}")

            top_rows = sorted(
                (
                    (op, op_counts[op], op_bytes[op], op_macs[op])
                    for op in op_counts
                ),
                key=lambda row: (row[1], row[0]),
                reverse=True,
            )
            print()
            print("top ops:")
            _print_table(top_rows, args.top)
            return 0
    except (OSError, ValueError, re.error) as error:
        print(f"mtrace analyze: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
