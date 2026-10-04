# NPU instruction encoding macros, mirroring the NpuR3 field layout in
# src/decode/encoding.h:
#   [31:28] major4  [27:25] sel3  [24:21] funct4  [20:17] reserved4
#   [16:12] src2    [11:7]  src1  [6:2]   dest    [1:0] q
# Correctness is pinned by the known-good words in tests/simulator_test.cpp
# (0x00002080 = tadd t0, t1, t2), which stay as raw constants on purpose.

# E_BIN: major4 = 0, reserved4 = 0, q = 00; sel3 = {dom2, BX}.
.macro E_BIN sel3, funct4, dest, src1, src2
    .word ((\sel3 << 25) | (\funct4 << 21) | (\src2 << 12) | (\src1 << 7) | (\dest << 2))
.endm

# E_BIN add on the tile domain with BX=0: sel3 = 0b000, funct4 = 0.
.macro TADD dest, src1, src2
    E_BIN 0, 0, \dest, \src1, \src2
.endm
