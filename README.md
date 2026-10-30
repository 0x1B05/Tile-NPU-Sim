# Tile-NPU functional simulator

A functional simulator for the [Tile-NPU ISA](https://github.com/0x1B05/Tile-NPU-ISA): a single-hart, in-order model that executes 32-bit RV64IM scalar instructions together with the NPU instruction set.

## Build and test

```sh
xmake                    # build the sim binary
xmake test               # host unit tests only (no cross toolchain)
xmake f --menu           # grouped menu configuration
xmake f --sdb=y          # optional interactive sdb build
xmake f --native=y       # optional host-specific -march=native code generation
```

Project options are grouped in `xmake f --menu` under `Build`, `Debug`, and `Guests`.

`sim` is the default binary target. Its internal static library, `sim_core`, is not a public build target; the host tests link it directly. SDB is off by default and only changes the `sim` build.

Integration suites are driven by `tests/run.py`:

```sh
python3 tests/run.py guest --build       # guest images + sim_guest_tests
python3 tests/run.py riscv-tests --build # official rv64ui/rv64um suite
python3 tests/run.py qwen3 --build       # tiny Qwen3 regression
python3 tests/run.py all --build
```

## Running programs

`sim` loads a flat binary at `0x80000000` into 1 GiB of RAM and starts execution there. A guest stops at `ebreak` (or `kernel.end` on the NPU side) with `a0` as the process exit code, 0 meaning success. A 16550-lite serial console is MMIO-mapped at `0x10000000`: storing a byte prints it on the host, and the status register always reads ready.

`--itrace` writes each fetched instruction as `[ITRACE] pc: encoding  disassembly` to `itrace.log` next to the first loaded image (`build/guest/itrace.log`, `build/apps/qwen3/itrace.log`, and so on). Use `--itrace-log PATH` to choose another path; it implies `--itrace`. Guest serial output stays on stdout. Build the simulator with `xmake f --itrace=y` to enable the option and its disassembler.

```sh
xmake run sim path/to/program.bin [data.bin@0xADDR ...]
```

Additional `path@0xADDR` images are loaded into the same memory before execution. `-m, --memory MiB` sets the RAM size and `-s, --steps N` caps the number of retired instructions (-1 for unlimited). The interactive `-i, --interactive` option is available only when SDB is enabled at build time:

```sh
xmake f --sdb=y
xmake
xmake run sim -i path/to/program.bin
```

```
(sdb) si 3        # step 3 instructions; an empty line repeats the previous command
(sdb) info r      # pc and scalar registers
(sdb) x 4 0x80010000
(sdb) c           # continue until the program stops
```

The runner is always built as `sim`; the internal core library is only a dependency.

## Guest tests

Guest programs under `tests/guest/` are C or assembly sources cross-compiled with a RISC-V toolchain (`riscv64-linux-gnu-` by default; override the prefix with `xmake f --riscv_cross=<prefix>`) and linked into flat binaries against a minimal `crt0.S`. `xmake build guest_images` only produces the images; it does not register tests. `python3 tests/run.py guest --build` builds the images and `sim_guest_tests`, then runs the in-process guest checks and the `rv64im` self-checking images. Use `--filter` to select a test or image by name.

## riscv-tests conformance

The official [riscv-tests](https://github.com/riscv-software-src/riscv-tests) suite (rv64ui + rv64um) is available as an opt-in check:

```sh
scripts/fetch_riscv_tests.sh      # clone the pinned commit into external/
python3 tests/run.py riscv-tests --build
```

The suite is assembled against a minimal custom env (`tests/guest/riscv-tests-env/riscv_test.h`) that reports pass/fail through the runner's `ebreak`/`a0` convention; `fence_i` and `ma_data` are excluded (Zifencei and trap delivery are not modeled). The runner executes the generated images through `sim`; missing external sources or images are reported by the runner instead of affecting `xmake f`.

## Qwen3 inference guest

`apps/qwen3/` runs Qwen3-0.6B inference on the simulator (bare-metal rv64im, NPU kernels in `kernels.S`):

```sh
xmake build qwen3_app        # build/apps/qwen3/qwen3.bin
```

Running it needs a converted model blob and a Q3P prompt image. Fetch the pinned weights and tokenizer into `external/qwen3/`, then convert the model to the flat q8s32 Q3B blob (per-row symmetric int8 weights, f32 scales, 32x32 tiled weights):

```sh
scripts/qwen3/fetch_model.sh   # set HF_BASE=... for another mirror
python3 scripts/qwen3/convert.py real external/qwen3 build/apps/qwen3/model.q3b
```

`--max-pos N` caps positions; conversion needs python3 with numpy. Run `scripts/qwen3/convert.py --help` for all options; `--verbose` reports per-tensor quantization errors and `--quiet` suppresses progress. The embedding table is q8s32 with one scale per vocabulary row and is stored as `[V/32][D/32][32][32]` tiles in `[V][D]` order; one 32x32 Tile holds one token's hidden vector for `D <= 1024`. Tokenize a prompt into a Q3P image and load it with the model; the two images must land at the addresses the guest is linked against (`BLOB_BASE` in `apps/qwen3/include/q3b.h`, `PROMPT_BASE` in `apps/qwen3/include/q3p.h`):

```sh
python3 scripts/qwen3/tokenize_prompt.py "你好" --out build/apps/qwen3/prompt.bin
./build/linux/x86_64/release/sim \
    build/apps/qwen3/qwen3.bin \
    build/apps/qwen3/model.q3b@0x81000000 \
    build/apps/qwen3/prompt.bin@0x80F00000
```

The same flow is available through the app runner, which tokenizes the prompt, writes `build/apps/qwen3/prompt.bin`, invokes `sim`, and decodes the generated token ids into text:

```sh
xmake build qwen3_run
xmake run qwen3_run "你好"
```

The app prints the prompt tokens and generated token ids over the serial console and stops after 32 new tokens or an EOS. The normal app contains no test instrumentation; the tiny regression uses a separate snapshot build.

Measure prefill, TTFT, and decode time from the serial stream:

```sh
xmake build qwen3_app
python3 scripts/qwen3/bench.py "你好" \
    --sim-run build/linux/x86_64/release/sim \
    --app build/apps/qwen3/qwen3.bin \
    --runs 3
```

`bench.py` reports load, prefill, end-to-end TTFT, compute-only TTFT, decode time, ms/token, and tokens/s. It timestamps the serial bytes directly, so it does not need simulator instrumentation. Use `--prompt-image` for an existing Q3P image and `--json` to save per-run metrics.

A fixed-seed random tiny model exercises the same path without any download, and the opt-in regression wraps the whole chain — convert, guest forward under `sim`, then compare against the numpy reference `ref.py` (snapshot records compared with the numpy reference, generated tokens exact):

```sh
python3 tests/run.py qwen3 --build
```

The runner builds `qwen3_snapshot_app`, generates the tiny model/prompt/reference, runs it under `sim`, and compares the snapshot records and generated tokens against the NumPy reference. It reports a clear error when `sim`, Python, or NumPy is unavailable.
