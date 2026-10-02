# Tile-NPU functional simulator

This simulator defines the architectural register state, bound configuration and memory-descriptor slots, a byte-addressed memory interface, and a single-step entry point.

Instruction decoding produces typed scalar and NPU IR. NPU elementwise operands
carry their data domain explicitly. `MatrixView` centralizes logical matrix
coordinates, while `DescriptorView` centralizes TM/AM/BM/VM address calculation
and post-access base-address increments. Configuration fields are accessed
through the shared `read_config_field` and `write_config_field` API, including
field-number mapping and value validation.

The runner loads the raw image at `0x80000000` into 128 MiB of RAM and starts execution at that address. An EBREAK with `a0 == 0` reports success; other stop conditions report failure.

Build and test from this directory:

```sh
xmake
xmake test
```

To run a flat `.bin` image, build the runner and pass the image path:

```sh
xmake build sim_run
xmake run sim_run path/to/program.bin
```
