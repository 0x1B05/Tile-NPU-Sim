# Tile-NPU functional simulator

A functional simulator for the Tile-NPU ISA: a single-hart, in-order model that executes 32-bit RV64IM scalar instructions together with the full NPU instruction set (elementwise arithmetic and comparison, broadcast, mask, type conversion, reductions, quantization, matrix multiply and dot, configuration, block/row/gather memory access, initialization and lane movement, synchronization).

The simulator core is the `tile_npu_sim` static library with a single-step API; `sim_run` is the runner and interactive debugger built on it. The ISA manual lives in the sibling repository at `../Tile-NPU-ISA`. Internal architecture notes and development conventions are documented in `AGENTS.md`.

## Build and test

```sh
xmake          # tile_npu_sim + sim_run
xmake test     # unit tests + guest images (guest images need the cross toolchain below)
```

## Running programs

`sim_run` loads a flat binary at `0x80000000` into 1 GiB of RAM and starts execution there. A guest stops at `ebreak` (or `kernel.end` on the NPU side) with `a0` as the process exit code, 0 meaning success. A 16550-lite serial console is MMIO-mapped at `0x10000000`: storing a byte prints it on the host, and the status register always reads ready.

```sh
xmake run sim_run path/to/program.bin [data.bin@0xADDR ...]
```

Additional `path@0xADDR` images are loaded into the same memory before execution. `-m, --memory MiB` sets the RAM size, `-s, --steps N` caps the number of retired instructions (-1 for unlimited), and `-i, --interactive` opens the sdb shell:

```sh
xmake run sim_run -i path/to/program.bin
```

```
(sdb) si 3        # step 3 instructions; an empty line repeats the previous command
(sdb) info r      # pc and scalar registers
(sdb) x 4 0x80010000
(sdb) c           # continue until the program stops
```

The runner is an xmake option; `xmake f --monitor=n` builds the library only.

## Guest tests

Guest programs under `tests/guest/` are C or assembly sources cross-compiled with a RISC-V toolchain (`riscv64-linux-gnu-` by default; override the prefix with `xmake f --riscv_cross=<prefix>`) and linked into flat binaries against a minimal `crt0.S`. Self-checking programs run through `sim_run` as individual `xmake test` cases. Images attach to the `sim_tests` target, so the `sim_run` binary itself never needs the cross toolchain; before a filtered rerun like `xmake test 'sim_run/guest/factorial'`, build the images first with `xmake build guest_images` or a full `xmake test`.

## riscv-tests conformance

The official [riscv-tests](https://github.com/riscv-software-src/riscv-tests) suite (rv64ui + rv64um) is available as an opt-in check:

```sh
scripts/fetch_riscv_tests.sh      # clone the pinned commit into external/
xmake f --rvtests=y && xmake test
```

The suite is assembled against a minimal custom env (`tests/riscv-tests-env/riscv_test.h`) that reports pass/fail through the runner's `ebreak`/`a0` convention; `fence_i` and `ma_data` are excluded (Zifencei and trap delivery are not modeled).

## Qwen3 inference guest

`apps/qwen3/` runs Qwen3-0.6B inference on the simulator (bare-metal rv64im, NPU kernels in `kernels.S`):

```sh
xmake build qwen3_app        # build/apps/qwen3.bin
```

Running it needs a converted model blob and a Q3P1 prompt image. Fetch the pinned weights and tokenizer into `external/qwen3/`, then convert the model to the flat q8s32 Q3B2 blob (per-row symmetric int8 weights, f32 scales). The model no longer contains a prompt:

```sh
scripts/qwen3/fetch_model.sh   # set HF_BASE=... for another mirror
python3 scripts/qwen3/convert.py real external/qwen3 model.q3b
```

`--max-pos N` caps positions; conversion needs python3 with numpy. Run `scripts/qwen3/convert.py --help` for all options; `--verbose` reports per-tensor quantization errors and `--quiet` suppresses progress. Tokenize a prompt into a Q3P1 image and load it with the model; the two images must land at the addresses the guest is linked against (`BLOB_BASE` in `apps/qwen3/include/q3b.h`, `PROMPT_BASE` in `apps/qwen3/include/q3p.h`):

```sh
python3 scripts/qwen3/tokenize_prompt.py "你好" --out build/apps/prompt.bin
./build/linux/x86_64/release/sim_run \
    build/apps/qwen3.bin \
    model.q3b@0x81000000 \
    build/apps/prompt.bin@0x80F00000
```

The same flow is available through the app runner, which tokenizes the prompt, builds a temporary Q3P1 image, and invokes `sim_run`:

```sh
xmake build qwen3_run
xmake run qwen3_run "你好"
```

The app prints the prompt tokens and generated token ids over the serial console and stops after 32 new tokens or an EOS. Numerical alignment probes in `forward.c` are compiled in by default; the `QWEN3_PROBES` define at the top of the file controls them.

A fixed-seed random tiny model exercises the same path without any download, and the opt-in regression wraps the whole chain — convert, guest forward under `sim_run`, then compare against the numpy reference `ref.py` (integer probes bit-exact, f32 probes within tolerance, generated tokens exact):

```sh
python3 scripts/qwen3/convert.py tiny \
    --out tiny.q3b --ref tiny_ref.npz --prompt-out tiny_prompt.bin
xmake f --qwen3test=y && xmake test
```

The regression requires `--monitor=y` and is skipped with a warning when python3 with numpy is unavailable.
