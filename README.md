# Tile-NPU functional simulator

This is the initial C++ simulator library. It defines the architectural register
state, bound configuration and memory-descriptor slots, a byte-addressed memory
interface, and a single-step entry point. The decoder and executor cover the
32-bit RV64IM integer instructions and Tile `tadd`. Fetch assembles four
instruction bytes in little-endian order.
Tile arrays use row-major indexing for the current canonical layout.
`decode_not_implemented` does not assert that an unrecognized encoding is legal;
`execution_not_implemented` means that decoding succeeded but execution is not
available yet.

`src/encoding.h` owns bit positions and parses the RISC-V R/I/S/B/U/J and NPU R3
layouts. `src/decode.cpp` dispatches by `q`, then NPU `major4` or the RV64IM
decoder in `src/decode_rv64im.cpp`. RV64IM decoding dispatches by opcode and then
by the instruction family's function fields. `DecodedInstr::imm`
holds a sign-extended I/S/B/U/J immediate, a shift amount for shift-immediate
instructions, or the raw 12-bit FENCE control field. `src/execute.cpp` dispatches
to the RV64I and RV64M executors or the Tile executor. The scalar executor uses
64-bit wraparound arithmetic, little-endian memory accesses, and naturally
aligned loads and stores. Misaligned accesses report a status without changing
architectural state. In this single-hart model FENCE retires as a no-op;
ECALL and EBREAK stop at the current PC for the caller to handle.

`Simulator::reset(entry_pc)` clears the simulator state and sets the PC, but does
not clear memory. This is a deterministic simulator initialization convention;
the ISA manual does not specify architectural reset values or an entry address.
Configuration fields are stored as C++ values, not packed register encodings.
The `tadd` executor validates the configuration fields it consumes. Full
configuration-write validation and bitfield packing remain future work;
`invalid_configuration` is a simulator diagnostic, not an ISA-defined trap.

Build and test from this directory:

```sh
xmake
xmake test
```

To run existing AM/NEMU `cpu-tests` raw images, build the optional runner and
pass one or more `.bin` files:

```sh
xmake build sim_cpu_tests
xmake run sim_cpu_tests path/to/test-riscv64-nemu.bin
```

The runner loads each image at `0x80000000` into 128 MiB of RAM, starts there,
and treats EBREAK with `a0 == 0` as success. This is a test harness convention,
not an architectural reset or syscall contract. It has no UART/MMIO, virtual
memory, or sign-extended address aliases. The external images are not build
dependencies and are not copied into this repository.

Xmake stores its compiler cache in the ignored local `.cache/ccache` directory.

Regenerate the compilation database for clangd from the current xmake
configuration:

```sh
xmake project -k compile_commands
```
