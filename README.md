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
individual `xmake test` cases.
