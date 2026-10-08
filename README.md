# Tile-NPU functional simulator

This simulator defines the architectural register state, bound configuration and memory-descriptor slots, a byte-addressed memory interface, and a single-step entry point.

Instruction decoding produces typed scalar and NPU IR. NPU elementwise operands
carry their data domain explicitly. `MatrixView` centralizes logical matrix
coordinates, while `DescriptorView` centralizes TM/AM/BM/VM address calculation
and post-access base-address increments. Configuration fields are accessed
through the shared `read_config_field` and `write_config_field` API, including
field-number mapping and value validation.

The runner loads the raw image at `0x80000000` into 128 MiB of RAM and starts execution at that address. An EBREAK with `a0 == 0` reports success; other stop conditions report failure. These are runner conventions, not ISA-defined behavior.

Build and test from this directory:

```sh
xmake
xmake test
```

`sim_run` has two modes. Interactive, an sdb shell over a flat `.bin` image:

```sh
xmake run sim_run path/to/program.bin
```

```
(sdb) si 3        # step 3 instructions (empty line repeats the command)
(sdb) info r      # pc and scalar registers
(sdb) x 4 0x80010000
(sdb) c           # continue until the program stops
```

Batch mode reports PASS/FAIL per image:

```sh
xmake run sim_run -b path/to/program.bin [more.bin ...]
```

The runner is an xmake option; `xmake f --monitor=n` builds the library only.

Guest test programs under `tests/guest/` are C sources cross-compiled with a
RISC-V toolchain (`riscv64-linux-gnu-gcc`/`objcopy` by default; override the
prefix with `xmake f --riscv_cross=<prefix>`), linked against a minimal
`crt0.S`. Self-checking RV64IM programs run through `sim_run -b` as
individual `xmake test` cases. Images attach to the `sim_tests` target, so
the `sim_run` binary itself never needs the cross toolchain; before a
filtered rerun like `xmake test 'sim_run/guest/factorial'`, build the images
first (`xmake build guest_images` or a full `xmake test`).

The official [riscv-tests](https://github.com/riscv-software-src/riscv-tests)
suite (rv64ui + rv64um) is available as an opt-in conformance check:

```sh
scripts/fetch_riscv_tests.sh      # clone the pinned commit into external/
xmake f --rvtests=y && xmake test
```

The suite is assembled against a minimal custom env
(`tests/riscv-tests-env/riscv_test.h`) that reports pass/fail through the
runner's `ebreak`/`a0` convention; `fence_i` and `ma_data` are excluded
(Zifencei and trap delivery are not modeled).

An inference guest for Qwen3-0.6B lives in `apps/qwen3/` (bare-metal rv64im,
NPU kernels in `kernels.S`), cross-compiled with the same RISC-V toolchain:

```sh
xmake build qwen3_app        # build/apps/qwen3.bin
```

Running it needs a converted model blob. Fetch the pinned weights and
tokenizer into `external/qwen3/`, then convert to the flat q8s32 blob
(per-row symmetric int8 weights, f32 scales) with a baked-in prompt:

```sh
scripts/qwen3/fetch_model.sh   # set HF_BASE=... for another mirror
python3 scripts/qwen3/convert.py external/qwen3 model.q3b --prompt "Hi" --chat
```

`--ids "1,2,3"` supplies token ids directly; `--max-pos N` caps positions.
Conversion needs python3 with numpy. Run the app image together with the
blob, which must land at the address the guest is linked against
(`BLOB_BASE` in `apps/qwen3/qwen3.h`):

```sh
./build/linux/x86_64/release/sim_run --run -m 1024 \
    build/apps/qwen3.bin model.q3b@0x81000000
```

The default 128 MiB RAM does not fit the real blob, hence `-m 1024`. In
`--run` mode the app prints the prompt and generated token ids over
semihosting and stops after 32 new tokens or an EOS. Numerical alignment
probes in `forward.c` are compiled in by default; cross-compile with
`-DQWEN3_PROBES=0` to silence them.

A fixed-seed random tiny model exercises the same path without any download:

```sh
python3 scripts/qwen3/convert.py --tiny tiny.q3b tiny_ref.npz
```

and the opt-in regression wraps the whole chain — convert, guest forward
under `sim_run`, then compare against the numpy reference `ref.py` (integer
probes bit-exact, f32 probes within tolerance, generated tokens exact):

```sh
xmake f --qwen3test=y && xmake test
```

The test requires `--monitor=y` and is skipped with a warning when python3
with numpy is unavailable.
