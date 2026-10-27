#include "decode_npu.h"

#include "domain.h"
#include "encoding.h"

#include <cstdint>
#include <type_traits>

namespace tile_npu {

// M_BLK block/gather (R2): major4 = 0000, q = 01.
// 31    28 27  26 25  24    22 21                12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 10 Vec8, 11 Vec32; Acc gather is reserved.
// funct3 = {form2, LS}: 000 block load, 001 block store, 010 gather load.
template <MblkOp Op> DecodeResult decode_mblk(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = sel3_domain(f.sel3);
  if (Op == MblkOp::gather && domain == NpuDomain::acc) {
    return DecodeError::illegal;
  }
  const NpuRegister reg{domain, f.dest};
  return DecodedNpuInstr{MblkInstr{.op = Op, .reg = reg, .gather_src = f.src2}};
}

DecodeResult decode_mblk_load(std::uint32_t word) { return decode_mblk<MblkOp::load>(word); }

DecodeResult decode_mblk_store(std::uint32_t word) { return decode_mblk<MblkOp::store>(word); }

DecodeResult decode_mblk_gather(std::uint32_t word) { return decode_mblk<MblkOp::gather>(word); }

// M_BLK row (MR): major4 = 0000, q = 01.
// 31    28  27  26   25  24    22 21       17 16      12 11        7 6        2 1   0
// [major4] [0] [EW] [IS] [funct3] [reserved5] [ index5 ] [reserved5] [  reg5  ] [ q ]
// EW: 0 = Tile, 1 = Acc. IS: 0 = immediate row, 1 = Scalar register in index5.
// funct3 = {form2, LS}: 100 row load, 101 row store.
template <MblkOp Op> DecodeResult decode_mblk_row(std::uint32_t word) {
  const auto f = encoding::NpuMR::from(word);
  const auto domain = paired_domains(f.sel3).matrix;
  const NpuRegister reg{domain, f.reg};
  const auto index = decode_index(f.index, (f.sel3 & 1) != 0);
  return DecodedNpuInstr{MblkInstr{.op = Op, .reg = reg, .index = index}};
}

DecodeResult decode_mblk_row_load(std::uint32_t word) {
  return decode_mblk_row<MblkOp::row_load>(word);
}

DecodeResult decode_mblk_row_store(std::uint32_t word) {
  return decode_mblk_row<MblkOp::row_store>(word);
}

// FILL_IMM (I): major4 = 0001, q = 01.
// 31    28 27  26 25    23 22                             7 6        2 1   0
// [major4] [dom2] [funct3] [            imm16             ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 10 Vec8, 11 Vec32. funct3[25:23] = 000.
DecodeResult decode_fill(std::uint32_t word) {
  const auto f = encoding::NpuI::from(word);
  const auto domain = static_cast<NpuDomain>(f.sel2);
  return DecodedNpuInstr{FillInstr{
      .dest = {.domain = domain, .index = f.dest},
      .value = FillImm16{static_cast<std::uint16_t>(f.imm)},
  }};
}

// REG_MOVE fillx/copy (R2): major4 = 0010, q = 01.
// 31    28 27  26 25  24    22 21                12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 10 Vec8, 11 Vec32.
// funct3[24:22]: 001 fillx (S5 = xS), 010 copy (S5 = same-domain source).
DecodeResult decode_fillx(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = sel3_domain(f.sel3);
  return DecodedNpuInstr{FillInstr{
      .dest = {.domain = domain, .index = f.dest},
      .value = ScalarRegister{f.src2},
  }};
}

DecodeResult decode_move_copy(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = sel3_domain(f.sel3);
  return DecodedNpuInstr{MoveInstr{
      .dest = {.domain = domain, .index = f.dest},
      .src = {.domain = domain, .index = f.src2},
  }};
}

// ROW_MOVE (R4): major4 = 0011, q = 01.
// 31    28  27  26   25  24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [0] [EW] [IS] [funct3] [reserved5] [ index5 ] [   S5   ] [   D5   ] [ q ]
// EW: must be 1 (32-bit Acc/Vec32); 0 is reserved.
// IS: 0 = immediate row, 1 = Scalar register in index5.
// funct3: 000 insert.row (vector S5 -> matrix D5),
//         001 extract.row (matrix S5 -> vector D5); 010..111 reserved.
template <RowOp Op> DecodeResult decode_row_move(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  constexpr auto MATRIX_DOMAIN = NpuDomain::acc;
  constexpr auto VECTOR_DOMAIN = NpuDomain::vec32;
  constexpr bool insert = Op == RowOp::insert;
  const auto matrix_index = insert ? f.dest : f.src1;
  return DecodedNpuInstr{RowInstr{
      .op = Op,
      .matrix = {.domain = MATRIX_DOMAIN, .index = matrix_index},
      .vector = {.domain = VECTOR_DOMAIN, .index = insert ? f.src1 : f.dest},
      .index = decode_index(f.src2, (f.sel3 & 1) != 0),
  }};
}

DecodeResult decode_row_insert(std::uint32_t word) { return decode_row_move<RowOp::insert>(word); }

DecodeResult decode_row_extract(std::uint32_t word) {
  return decode_row_move<RowOp::extract>(word);
}

// LANE_MOVE (R4): major4 = 0100, q = 01.
// 31    28  27  26   25  24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [0] [EW] [IS] [funct3] [reserved5] [ index5 ] [   S5   ] [   D5   ] [ q ]
// EW: 0 = 8-bit Vec8, 1 = 32-bit Vec32.
// IS: 0 = immediate lane, 1 = Scalar register in index5.
// funct3: 000 extract (vector S5 -> Scalar D5), 001 insert (Scalar S5 -> vector D5),
//         010 broadcast (vector S5 -> vector D5)
template <typename Instr> DecodeResult decode_lane_move(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = paired_domains(f.sel3).vector;
  const auto index = decode_index(f.src2, (f.sel3 & 1) != 0);
  if constexpr (std::is_same_v<Instr, LaneExtractInstr>) {
    return DecodedNpuInstr{LaneExtractInstr{
        .dest = {.index = f.dest},
        .src = {.domain = domain, .index = f.src1},
        .index = index,
    }};
  } else if constexpr (std::is_same_v<Instr, LaneInsertInstr>) {
    return DecodedNpuInstr{LaneInsertInstr{
        .dest = {.domain = domain, .index = f.dest},
        .src = {.index = f.src1},
        .index = index,
    }};
  } else {
    static_assert(std::is_same_v<Instr, LaneBroadcastInstr>);
    return DecodedNpuInstr{LaneBroadcastInstr{
        .dest = {.domain = domain, .index = f.dest},
        .src = {.domain = domain, .index = f.src1},
        .index = index,
    }};
  }
}

DecodeResult decode_lane_extract(std::uint32_t word) {
  return decode_lane_move<LaneExtractInstr>(word);
}

DecodeResult decode_lane_insert(std::uint32_t word) {
  return decode_lane_move<LaneInsertInstr>(word);
}

DecodeResult decode_lane_broadcast(std::uint32_t word) {
  return decode_lane_move<LaneBroadcastInstr>(word);
}

} // namespace tile_npu
