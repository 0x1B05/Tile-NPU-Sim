# Tile-NPU functional simulator

This is the initial C++ simulator library. It defines the architectural register
state, bound configuration and memory-descriptor slots, a byte-addressed memory
interface, and a single-step entry point. The decoder and executor are separate:
the decoder recognizes RV64I `add` and Tile `tadd`; other instruction families
are not implemented yet. Fetch assembles four instruction bytes in little-endian
order. Tile arrays use row-major indexing for the current canonical layout.
`decode_not_implemented` does not assert that an unrecognized encoding is legal.
`src/encoding.h` owns bit positions and parses the RV R-type and NPU R3 layouts.
`src/decode.cpp` dispatches by `q`, then opcode or `major4`, then the instruction
family's selector and function fields. Execution semantics remain in
`src/execute.cpp`.

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

Regenerate the compilation database for clangd from the current xmake
configuration:

```sh
xmake project -k compile_commands
```
