#!/usr/bin/env python3
"""Generate a clangd compilation database for the Qwen3 guest app."""

import json
import pathlib
import shutil

app = pathlib.Path(__file__).resolve().parent
compiler = shutil.which("riscv64-linux-gnu-gcc") or "riscv64-linux-gnu-gcc"

sources = sorted(p.name for p in app.glob("*.c"))
sources += sorted(p.name for p in app.glob("*.S"))

entries = []
for source in sources:
    args = [compiler, "-c"]
    if source.endswith(".S"):
        args += ["-x", "assembler-with-cpp"]
    args += [
        "-march=rv64im",
        "-mabi=lp64",
        "-O2",
        "-ffreestanding",
        "-nostdlib",
        "-static",
        "-no-pie",
        "-Iinclude",
        "-I../../tests/guest/include",
        source,
    ]
    entries.append({"directory": str(app), "arguments": args, "file": source})

output = app / "compile_commands.json"
output.write_text(json.dumps(entries, indent=2) + "\n")
print(f"wrote {output} ({len(entries)} entries)")
