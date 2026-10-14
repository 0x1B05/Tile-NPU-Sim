#include "decode_npu.h"

#include "config.h"
#include "encoding.h"
#include "op_info.h"
#include "pattern.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <optional>

namespace tile_npu {
namespace {

// sel3 = {dom2, ...}: the two high bits name the data domain directly.
constexpr NpuDomain sel3_domain(std::uint8_t sel3) { return static_cast<NpuDomain>(sel3 >> 1); }

// sel3 = {0, EW, ...}: EW at bit 26 pairs a matrix domain with a vector
// domain, Tile with Vec8 (0) and Acc with Vec32 (1).
struct PairedDomains {
  NpuDomain matrix;
  NpuDomain vector;
};

constexpr PairedDomains paired_domains(std::uint8_t sel3) {
  return (sel3 & 0b010) != 0 ? PairedDomains{NpuDomain::acc, NpuDomain::vec32}
                             : PairedDomains{NpuDomain::tile, NpuDomain::vec8};
}

// The BX bit selects the second source: BX = 0 a same-domain register, BX = 1
// the Scalar register xS. The register form fails on an out-of-range number.
std::optional<NpuOperand> decode_bx_src2(std::uint8_t sel3, std::uint8_t src2, NpuDomain domain) {
  if ((sel3 & 1) != 0) {
    return NpuOperand{ScalarRegister{src2}};
  }
  if (src2 >= domain_register_count(domain)) {
    return std::nullopt;
  }
  return NpuOperand{NpuRegister{domain, src2}};
}

// E_BIN (R3): major4 = 0000, q = 00.
// 31    28 27  26  25  24    21 20       17 16      12 11       7 6        2 1   0
// [major4] [dom2] [BX] [funct4] [reserved4] [   B5   ] [   A5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 11 Vec32; 10 reserved.
// BX: 0 = same-domain B5, 1 = Scalar xS in B5.
// funct4 = {G, op3}, decoded by EBIN_BY_FUNCT4; Tile mul/div are reserved.
DecodeResult decode_ebin(std::uint32_t word) {
  const auto f = encoding::NpuR3::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto op = static_cast<EbinOp>(f.funct4);
  if (!ebin_legal(op, domain)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || f.src1 >= count) {
    return DecodeError::illegal;
  }
  const auto src2 = decode_bx_src2(f.sel3, f.src2, domain);
  if (!src2) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{EbinInstr{op, {domain, f.dest}, {domain, f.src1}, *src2}};
}

// E_UNARY (R2): major4 = 0001, q = 00.
// 31    28 27  26 25  24    22 21                12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 Tile (only the not operation), 01 Acc, 11 Vec32; 10 reserved.
// funct3: 000 not, 001 abs, 010 neg, 011 exp2.approx, 100 rcp.approx,
//         101 rsqrt.approx, 110 silu.approx; 111 reserved.
DecodeResult decode_eun(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto op = static_cast<EunOp>(f.funct3);
  const auto count = domain_register_count(domain);
  if (!eun_legal(op, domain) || f.dest >= count || f.src2 >= count) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{EunInstr{op, {domain, f.dest}, {domain, f.src2}}};
}

// E_CMP (R4): major4 = 0010, q = 00.
// 31    28 27  26  25  24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [dom2] [BX] [funct3] [reserved5] [   B5   ] [   A5   ] [   M5   ] [ q ]
// dom2: 01 Acc, 11 Vec32; 00/10 reserved. M5 is the u32 mask destination.
// BX: 0 = same-domain B5, 1 = Scalar xS in B5.
// funct3: 000 eq, 001 ne, 010 lt, 011 ge, 100 le, 101 unord; 110/111 reserved.
DecodeResult decode_ecmp(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto &info = CMP_BY_FUNCT3[f.funct3];
  if (!info) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.src1 >= count || f.dest >= count) {
    return DecodeError::illegal;
  }
  const auto src2 = decode_bx_src2(f.sel3, f.src2, domain);
  if (!src2) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{CmpInstr{info->op, {domain, f.dest}, {domain, f.src1}, *src2}};
}

// E_BCAST (R3): major4 = 0101, q = 00.
// 31    28  27  26   25   24    21 20       17 16      12 11       7 6        2 1   0
// [major4] [0] [EW] [DIR] [funct4] [reserved4] [   S5   ] [   A5   ] [   D5   ] [ q ]
// EW: 0 = Tile/Vec8, 1 = Acc/Vec32; A5/D5 are matrices, S5 is a vector.
// DIR: 0 = byrow, 1 = bycol. funct4 = {G, op3}, shared with E_BIN.
DecodeResult decode_ebcast(std::uint32_t word) {
  const auto f = encoding::NpuR3::from(word);
  const auto domains = paired_domains(f.sel3);
  const auto op = static_cast<EbinOp>(f.funct4);
  if (!ebin_legal(op, domains.matrix)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domains.matrix);
  if (f.dest >= count || f.src1 >= count) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{BcastInstr{op,
                                    {domains.matrix, f.dest},
                                    {domains.matrix, f.src1},
                                    {domains.vector, f.src2},
                                    (f.sel3 & 1) != 0}};
}

// E_R4 (R4): major4 = 0011, q = 00.
// 31    28 27  26 25  24    22 21      17 16      12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [  arg5  ] [   B5   ] [   A5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 11 Vec32; 10 reserved.
// funct3: 000 select (arg5 = mask M), 001 fmadd (arg5 = addend C);
//         010..111 reserved. fmadd requires Acc/Vec32 with f32 dtype.
DecodeResult decode_er4(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto op = static_cast<Er4Op>(f.funct3);
  if (!er4_legal(op, domain)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || f.arg >= count || f.src1 >= count || f.src2 >= count) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{
      Er4Instr{op, {domain, f.dest}, {domain, f.arg}, {domain, f.src1}, {domain, f.src2}}};
}

// E_MASK (R4): major4 = 0100, q = 00.
// 31    28 27  26  25  24    22 21      17 16      12 11       7 6        2 1   0
// [major4] [dom2] [FS] [funct3] [xBound5 ] [ fill5  ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 10 Vec8, 11 Vec32.
// FS: 0 = fill5 code in [16:12], 1 = Scalar xFill in [16:12].
// funct3: 000 tail, 001 tril, 010 triu; 011..111 reserved. tril/triu are matrix-only.
// xBound5 names Scalar xBounds/xLen for tail, or xDelta for tril/triu.
DecodeResult decode_emask(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const bool fill_is_scalar = (f.sel3 & 1) != 0;
  const auto op = static_cast<MaskOp>(f.funct3);
  if (!mask_legal(op, domain) || (!fill_is_scalar && f.src2 > FILL5_MAX)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || f.src1 >= count) {
    return DecodeError::illegal;
  }
  const auto fill =
      fill_is_scalar ? FillOperand{ScalarRegister{f.src2}} : FillOperand{Fill5Code{f.src2}};
  return DecodedNpuInstr{MaskInstr{op, {domain, f.dest}, {domain, f.src1}, f.arg, fill}};
}

// CVT (R2): major4 = 0011, q = 10.
// 31    28 27  26 25  24    22 21                12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// dom2: destination 01 Acc, 11 Vec32; 00/10 reserved.
// funct3: 000 i32 -> f32 (same-domain S5/D5), 001 Tile i8 -> Acc i32;
template <CvtOp Op> DecodeResult decode_cvt(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto src_domain = Op == CvtOp::widen ? NpuDomain::tile : domain;
  if (f.dest >= domain_register_count(domain) || f.src2 >= domain_register_count(src_domain)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{CvtInstr{Op, {domain, f.dest}, {src_domain, f.src2}}};
}

// AREDUCE (R2): major4 = 0001, q = 10.
// 31    28 27       25 24    22 21                12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// funct3: 000 sum, 001 max, 010 min, 011 sumsq, 100 argmax, 101 absmax;
//         110/111 reserved. S5 is Acc; D5 is Vec32.
DecodeResult decode_areduce(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  if (f.funct3 > 0b101 || f.src2 >= ACC_REGISTERS) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{ReduceInstr{static_cast<ReduceOp>(f.funct3),
                                     NpuOperand{NpuRegister{NpuDomain::vec32, f.dest}},
                                     {},
                                     {NpuDomain::acc, f.src2}}};
}

// VREDUCE sum/max/min/sumsq (R2): major4 = 0010, q = 10.
// 31    28 27       25 24    22 21                12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// funct3: 000 sum, 001 max, 010 min, 011 sumsq. S5 is Vec32; D5 is Scalar.
// funct3 = 100 uses the argmax layout below; 101..111 are reserved.
DecodeResult decode_vreduce(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  return DecodedNpuInstr{ReduceInstr{static_cast<ReduceOp>(f.funct3),
                                     NpuOperand{ScalarRegister{f.dest}},
                                     {},
                                     {NpuDomain::vec32, f.src2}}};
}

// VREDUCE argmax (R4): major4 = 0010, q = 10.
// 31    28 27       25 24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [reserved5] [ xVal5  ] [   S5   ] [   D5   ] [ q ]
// funct3 = 100. S5 is Vec32; D5/xVal5 are the Scalar xIndex/xValue destinations.
DecodeResult decode_vreduce_argmax(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  return DecodedNpuInstr{ReduceInstr{ReduceOp::argmax,
                                     NpuOperand{ScalarRegister{f.dest}},
                                     NpuOperand{ScalarRegister{f.src2}},
                                     {NpuDomain::vec32, f.src1}}};
}

// index5 with the IS bit: an immediate index, or the named Scalar register.
IndexOperand decode_index(RegisterIndex raw, bool is_scalar) {
  return is_scalar ? IndexOperand{ScalarRegister{raw}} : IndexOperand{ImmediateIndex{raw}};
}

// QNT (R4): major4 = 0100, q = 10.
// 31    28 27  26  25  24    22 21      17 16      12 11       7 6        2 1   0
// [major4] [dom2] [IS] [funct3] [vScale5 ] [ index5 ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 tquant (Tile), 01 tdequant (Acc), 10 vquant (Vec8), 11 bdequant (Vec32).
// funct3: 000 quant/dequant, 001 external-scale tquant; 010..111 reserved.
// vScale5 is an output for dynamic quant, an input for external-scale quant/dequant.
// Vector forms use IS: 0 = immediate scale lane, 1 = Scalar register in index5.
// Matrix forms require IS = 0 and index5 = 0.
DecodeResult decode_qnt(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const bool index_is_scalar = (f.sel3 & 1) != 0;
  const bool vector_form = is_vector_domain(domain);
  // funct3 = 000 quantizes into Tile/Vec8 destinations and dequantizes into
  // Acc/Vec32 destinations; the source is the paired domain.
  const auto op = f.funct3 == 0b001                 ? QntOp::quant_ext_scale
                  : qnt_domain_is_quantized(domain) ? QntOp::quant
                                                    : QntOp::dequant;
  if (f.funct3 > 0b001 || !qnt_dest_legal(op, domain) ||
      (!vector_form && (index_is_scalar || f.src2 != 0))) {
    return DecodeError::illegal;
  }
  const auto src_domain = qnt_paired_domain(domain);
  if (f.dest >= domain_register_count(domain) || f.src1 >= domain_register_count(src_domain)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{QntInstr{
      op, {domain, f.dest}, {src_domain, f.src1}, f.arg, decode_index(f.src2, index_is_scalar)}};
}

// MMA_BDOT (R4): major4 = 0000, q = 10.
// 31    28 27       25 24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [reserved5] [   B5   ] [   A5   ] [   D5   ] [ q ]
// funct3 = {mma/bdot, nn/nt, zero/acc}; bdot has only zero, so 101/111 are reserved.
DecodeResult decode_mma(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto &info = MMA_BY_FUNCT3[f.funct3];
  if (!info) {
    return DecodeError::illegal;
  }
  // src2 is always tile
  if (f.dest >= domain_register_count(info->dest_domain) ||
      f.src1 >= domain_register_count(info->lhs_domain) || f.src2 >= TILE_REGISTERS) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{MmaInstr{info->op,
                                  {info->dest_domain, f.dest},
                                  {info->lhs_domain, f.src1},
                                  {NpuDomain::tile, f.src2}}};
}

// SYS (Z): major4 = 1000, q = 10.
// 31    28 27    25 24                                           2 1   0
// [major4] [funct3] [                 reserved23                 ] [ q ]
// funct3: 000 fence.mem, 001 fence.sa, 010 fence.all, 011 kernel.end;
//         100..111 reserved.
DecodeResult decode_sys(std::uint32_t word) {
  const auto funct3 = encoding::field<27, 25>(word);
  if (funct3 > 0b011) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{SysInstr{static_cast<SysOp>(funct3)}};
}

// M_BLK block/gather (R2): major4 = 0000, q = 01.
// 31    28 27  26 25  24    22 21                12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 10 Vec8, 11 Vec32; Acc gather is reserved.
// funct3 = {form2, LS}: 000 block load, 001 block store, 010 gather load.
template <MblkOp Op> DecodeResult decode_mblk(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = sel3_domain(f.sel3);
  if ((Op == MblkOp::gather && domain == NpuDomain::acc) ||
      f.dest >= domain_register_count(domain)) {
    return DecodeError::illegal;
  }
  const NpuRegister reg{domain, f.dest};
  return DecodedNpuInstr{is_store(Op) ? MblkInstr{Op, {}, reg, ImmediateIndex{}, f.src2}
                                      : MblkInstr{Op, reg, {}, ImmediateIndex{}, f.src2}};
}

// M_BLK row (MR): major4 = 0000, q = 01.
// 31    28  27  26   25  24    22 21       17 16      12 11        7 6        2 1   0
// [major4] [0] [EW] [IS] [funct3] [reserved5] [ index5 ] [reserved5] [  reg5  ] [ q ]
// EW: 0 = Tile, 1 = Acc. IS: 0 = immediate row, 1 = Scalar register in index5.
// funct3 = {form2, LS}: 100 row load, 101 row store.
template <MblkOp Op> DecodeResult decode_mblk_row(std::uint32_t word) {
  const auto f = encoding::NpuMR::from(word);
  const auto domain = paired_domains(f.sel3).matrix;
  if (f.reg >= domain_register_count(domain)) {
    return DecodeError::illegal;
  }
  const NpuRegister reg{domain, f.reg};
  const auto index = decode_index(f.index, (f.sel3 & 1) != 0);
  return DecodedNpuInstr{is_store(Op) ? MblkInstr{Op, {}, reg, index, 0}
                                      : MblkInstr{Op, reg, {}, index, 0}};
}

// INIT_MOVE fill (I): major4 = 0010, q = 01.
// 31    28 27  26 25    23 22                             7 6        2 1   0
// [major4] [dom2] [funct3] [            imm16             ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 10 Vec8, 11 Vec32. funct3[25:23] = 000.
DecodeResult decode_init_fill(std::uint32_t word) {
  const auto f = encoding::NpuI::from(word);
  const auto domain = static_cast<NpuDomain>(f.sel2);
  if (f.dest >= domain_register_count(domain)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{InitInstr{InitOp::fill, {domain, f.dest}, 0, f.imm}};
}

// INIT_MOVE fillx/copy/ttranspose (R2): major4 = 0010, q = 01.
// 31    28 27  26 25  24    22 21                12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 10 Vec8, 11 Vec32.
// funct3[24:22]: 001 fillx (S5 = xS), 010 copy (S5 = same-domain source),
//               011 ttranspose (Tile only).
template <InitOp Op> DecodeResult decode_init_move(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = sel3_domain(f.sel3);
  if (Op == InitOp::transpose && domain != NpuDomain::tile) {
    return DecodeError::illegal; // ttranspose is Tile-only
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || (Op != InitOp::fillx && f.src2 >= count)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{InitInstr{Op, {domain, f.dest}, f.src2, 0}};
}

// ROW_MOVE (R4): major4 = 0011, q = 01.
// 31    28  27  26   25  24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [0] [EW] [IS] [funct3] [reserved5] [ index5 ] [   S5   ] [   D5   ] [ q ]
// EW: 0 = 8-bit Tile/Vec8, 1 = 32-bit Acc/Vec32.
// IS: 0 = immediate row, 1 = Scalar register in index5.
// funct3: 000 insert.row (vector S5 -> matrix D5),
//         001 extract.row (matrix S5 -> vector D5); 010..111 reserved.
template <RowOp Op> DecodeResult decode_row_move(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domains = paired_domains(f.sel3);
  constexpr bool insert = Op == RowOp::insert;
  const auto matrix_index = insert ? f.dest : f.src1;
  if (matrix_index >= domain_register_count(domains.matrix)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{RowInstr{Op,
                                  {domains.matrix, matrix_index},
                                  {domains.vector, insert ? f.src1 : f.dest},
                                  decode_index(f.src2, (f.sel3 & 1) != 0)}};
}

// LANE_MOVE (R4): major4 = 0100, q = 01.
// 31    28  27  26   25  24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [0] [EW] [IS] [funct3] [reserved5] [ index5 ] [   S5   ] [   D5   ] [ q ]
// EW: 0 = 8-bit Vec8, 1 = 32-bit Vec32.
// IS: 0 = immediate lane, 1 = Scalar register in index5.
// funct3: 000 extract (vector S5 -> Scalar D5), 001 insert (Scalar S5 -> vector D5),
//         010 broadcast (vector S5 -> vector D5)
template <LaneOp Op> DecodeResult decode_lane_move(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = paired_domains(f.sel3).vector;
  const auto dst = Op == LaneOp::extract ? NpuOperand{ScalarRegister{f.dest}}
                                         : NpuOperand{NpuRegister{domain, f.dest}};
  const auto src = Op == LaneOp::insert ? NpuOperand{ScalarRegister{f.src1}}
                                        : NpuOperand{NpuRegister{domain, f.src1}};
  return DecodedNpuInstr{LaneInstr{Op, dst, src, decode_index(f.src2, (f.sel3 & 1) != 0)}};
}

// sel2 selects the register family pair, the vector-side bit picks within it:
// 00 -> TC/BC, 01 -> AC/VC, 10 -> TM/BM, 11 -> AM/VM.
constexpr ConfigRegType cfg_reg_type(std::uint8_t sel2, bool vector_side) {
  switch (sel2) {
  case 0: return vector_side ? ConfigRegType::bc : ConfigRegType::tc;
  case 1: return vector_side ? ConfigRegType::vc : ConfigRegType::ac;
  case 2: return vector_side ? ConfigRegType::bm : ConfigRegType::tm;
  default: return vector_side ? ConfigRegType::vm : ConfigRegType::am;
  }
}

// The cfg.seti majors: 0101 is the matrix side (TC/AC/TM/AM), 0110 the
// vector side (BC/VC/BM/VM).
constexpr std::uint8_t MAJOR4_CFG_VEC_SETI = 0b0110;

// CFG_MAT_SETI / CFG_VEC_SETI (I): major4 = 0101 / 0110, q = 10.
// 31    28 27  26 25    23 22                             7 6        2 1   0
// [major4] [sel2] [field3] [            imm16             ] [   C5   ] [ q ]
// sel2: 00 TC/BC, 01 AC/VC, 10 TM/BM, 11 AM/VM (matrix/vector major).
DecodeResult decode_cfg_seti(std::uint32_t word) {
  const auto f = encoding::NpuI::from(word);
  const auto reg_type = cfg_reg_type(f.sel2, f.major4 == MAJOR4_CFG_VEC_SETI);
  if (f.dest >= config_register_count(reg_type) ||
      !std::holds_alternative<ConfigField>(config_field_from_code(reg_type, f.field3))) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{CfgInstr{CfgOp::seti, reg_type, f.dest, 0, f.field3, f.imm}};
}

// CFG_REG (R4): major4 = 0111, q = 10.
// 31    28 27  26  25  24    22 21      17 16       12 11       7 6        2 1   0
// [major4] [sel2] [MV] [funct3] [ field5 ] [reserved5] [   S5   ] [   D5   ] [ q ]
// sel2: 00 TC/BC, 01 AC/VC, 10 TM/BM, 11 AM/VM; MV: 0 = matrix, 1 = vector.
// funct3: 000 setx (xS -> C), 001 copy (CS -> CD), 010 get (C -> xD).
template <CfgOp Op> DecodeResult decode_cfg_reg(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto reg_type = cfg_reg_type(f.sel3 >> 1, (f.sel3 & 1) != 0);
  const auto count = config_register_count(reg_type);
  // Scalar operands (xS in setx, xD in get) always name a valid register;
  // config operands are bounded per register type.
  if (Op != CfgOp::get && f.dest >= count) {
    return DecodeError::illegal;
  }
  if (Op != CfgOp::setx && f.src1 >= count) {
    return DecodeError::illegal;
  }
  const auto field = static_cast<std::uint8_t>(encoding::field<2, 0>(f.arg));
  if constexpr (Op != CfgOp::copy) {
    if (!std::holds_alternative<ConfigField>(config_field_from_code(reg_type, field))) {
      return DecodeError::illegal;
    }
  }
  return DecodedNpuInstr{CfgInstr{Op, reg_type, f.dest, f.src1, field, 0}};
}

// CFG_SHAPE (R4 layout, all fields immediate): major4 = 1001, q = 10.
// 31    28 27  26  25  24    22 21      17 16      12 11       7 6        2 1   0
// [major4] [dom2] [SAT] [dtype3] [ cols5  ] [ rows5  ] [   N5   ] [   C5   ] [ q ]
// dom2: 00 TC, 01 AC, 10 BC, 11 VC. Sizes and the count are encoded minus one.
// BC/VC reserve cols5/SAT, AC reserves SAT; C5 + N5 must name a register.
DecodeResult decode_cfg_shape(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const bool sat = (f.sel3 & 1) != 0;
  if ((sat && domain != NpuDomain::tile) || (is_vector_domain(domain) && f.arg != 0) ||
      f.dest + f.src1 >= domain_register_count(domain)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{ShapeInstr{domain, f.dest, static_cast<std::uint8_t>(f.src1 + 1), f.funct3,
                                    static_cast<std::uint8_t>(f.src2 + 1),
                                    static_cast<std::uint8_t>(f.arg + 1),
                                    sat ? ArithMode::sat : ArithMode::wrap}};
}

// clang-format off

//   R3: major4 sel3 funct4 reserved4 src2 src1 dest q
//   I:  major4 sel2 field3 imm16(4x4)  C5   q
//   R4: major4 sel3 funct3 arg5    src2 src1 dest q
//   R2: major4 sel3 funct3 reserved10  S5   D5  q
//   MR: major4 sel3 funct3 reserved5 index5 reserved5 reg5 q
// E_BIN reads sel3 as {dom2, BX}: dom tile=00 acc=01 vec8=10 vec32=11.
constexpr DecodeEntry NPU_TABLE[] = {
    {"0000 00? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // tile
    {"0000 01? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // acc
    {"0000 10? ???? 0000 ????? ????? ????? 00", illegal},                             // vec8 has no E_BIN
    {"0000 11? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // vec32
    {"0000 ??? ???? ???? ????? ????? ????? 00", illegal},                             // reserved4 != 0
    // E_UNARY (R2): sel3 = {dom2, 0}; `not` on any matrix/vec32 domain, the
    // rest of the ops on Acc/Vec32 only (patterns pin funct3 = 000 for
    // dom2 x0, i.e. Tile/Vec8).
    {"0001 ??0 000 0000000000 ????? ????? 00", decode_eun},                           // not
    {"0001 ?10 ??? 0000000000 ????? ????? 00", decode_eun},                           // Acc/Vec32 ops
    {"0001 ??? ??? ????? ????? ????? ????? 00", illegal},                             // reserved E_UNARY encodings
    // E_CMP (R4): sel3 = {dom2, BX}, reserved5 pinned to 0; comparisons only
    // exist on Acc/Vec32.
    {"0010 01? ??? 00000 ????? ????? ????? 00", decode_ecmp},                           // acc
    {"0010 11? ??? 00000 ????? ????? ????? 00", decode_ecmp},                           // vec32
    {"0010 ??? ??? ????? ????? ????? ????? 00", illegal},                               // reserved E_CMP encodings
    // E_BCAST (R3): sel3 = {0, EW, DIR}, reserved4 pinned to 0.
    {"0101 0?? ???? 0000 ????? ????? ????? 00", decode_ebcast},
    {"0101 ??? ???? ???? ????? ????? ????? 00", illegal},                               // reserved E_BCAST encodings
    // E_R4 (R4): sel3 = {dom2, 0}, arg5 is the fourth operand (mask/addend).
    {"0011 ??0 ??? ????? ????? ????? ????? 00", decode_er4},
    {"0011 ??? ??? ????? ????? ????? ????? 00", illegal},                               // reserved E_R4 encodings
    // E_MASK (R4): sel3 = {dom2, FS}, [21:17] = xBound, [16:12] = fill5/xFill;
    // one pattern, the handler rejects the reserved funct3/vector-tril/fill5 encodings.
    {"0100 ??? ??? ????? ????? ????? ????? 00", decode_emask},
    // All assigned elementwise majors decode above; the rest is reserved.
    {"???? ??? ???? ???? ????? ????? ????? 00", illegal},

    // M_BLK (R2): sel3 = {dom2, 0}; S5 is the gather index vector, 0 for the
    // block forms. funct3 = {form2, LS}: 00x block, 010 gather.
    {"0000 ??0 000 0000000000 00000 ????? 01", decode_mblk<MblkOp::load>},           // block load
    {"0000 ??0 001 0000000000 00000 ????? 01", decode_mblk<MblkOp::store>},          // block store
    {"0000 ??0 010 0000000000 ????? ????? 01", decode_mblk<MblkOp::gather>},         // gather load
    // M_BLK row forms (MR): sel3 = {0, EW, IS}, funct3 = 10x.
    {"0000 0?? 100 00000 ????? 00000 ????? 01", decode_mblk_row<MblkOp::row_load>},   // row load
    {"0000 0?? 101 00000 ????? 00000 ????? 01", decode_mblk_row<MblkOp::row_store>},  // row store
    {"0000 ??? ??? ????? ????? ????? ????? 01", illegal},                             // reserved M_BLK encodings
    // INIT_MOVE R2 forms: sel3 = {dom2, 0}, funct3 = fillx/copy/ttranspose.
    // fillx precedes fill: imm16 = 0x8000..0x801F matches both patterns,
    // and the ISA assigns these encodings to fillx.
    {"0010 ??0 001 0000000000 ????? ????? 01", decode_init_move<InitOp::fillx>},       // fillx
    {"0010 ??0 010 0000000000 ????? ????? 01", decode_init_move<InitOp::copy>},        // copy
    {"0010 ??0 011 0000000000 ????? ????? 01", decode_init_move<InitOp::transpose>},   // ttranspose
    // INIT_MOVE fill (I format): dom2 + funct3 = 000 + imm16; other funct3
    // values in the I layout are reserved.
    {"0010 ?? 000 ???? ???? ???? ???? ????? 01", decode_init_fill},                   // fill
    {"0010 ??? ??? ????? ????? ????? ????? 01", illegal},                             // reserved INIT_MOVE encodings
    // ROW_MOVE (R4): sel3 = {0, EW, IS}, funct3 = insert.row/extract.row.
    {"0011 0?? 000 00000 ????? ????? ????? 01", decode_row_move<RowOp::insert>},       // insert.row
    {"0011 0?? 001 00000 ????? ????? ????? 01", decode_row_move<RowOp::extract>},      // extract.row
    {"0011 ??? ??? ????? ????? ????? ????? 01", illegal},                             // reserved ROW_MOVE encodings
    // LANE_MOVE (R4): sel3 = {0, EW, IS}, funct3 = extract/insert/broadcast.
    {"0100 0?? 000 00000 ????? ????? ????? 01", decode_lane_move<LaneOp::extract>},     // extract
    {"0100 0?? 001 00000 ????? ????? ????? 01", decode_lane_move<LaneOp::insert>},      // insert
    {"0100 0?? 010 00000 ????? ????? ????? 01", decode_lane_move<LaneOp::broadcast>},   // broadcast
    {"0100 ??? ??? ????? ????? ????? ????? 01", illegal},                             // reserved LANE_MOVE encodings
    // All assigned data movement majors decode above; the rest is reserved.
    {"???? ??? ???? ???? ????? ????? ????? 01", illegal},
    // MMA_BDOT (R4): reserved3/reserved5 pinned to 0; one pattern, the
    // handler rejects the reserved funct3 codes 101/111.
    {"0000 000 ??? 00000 ????? ????? ????? 10", decode_mma},
    {"0000 ??? ??? ????? ????? ????? ????? 10", illegal},                             // reserved MMA_BDOT encodings
    // CVT (R2): sel3 = {dom2, 0}, dom2 = Acc/Vec32 only. funct3 = 000 cvt,
    // 001 twiden (Acc destination only).
    {"0011 ?10 000 0000000000 ????? ????? 10", decode_cvt<CvtOp::cvt>},               // acvt/vcvt
    {"0011 010 001 0000000000 ????? ????? 10", decode_cvt<CvtOp::widen>},             // twiden
    {"0011 ??? ??? ????? ????? ????? ????? 10", illegal},                             // reserved CVT encodings
    // AREDUCE (R2): reserved3/reserved10 pinned to 0; funct3 = 000 sum,
    // 001 max, 010 min, 011 sumsq, 100 argmax, 101 absmax.
    {"0001 000 ??? 0000000000 ????? ????? 10", decode_areduce},
    {"0001 ??? ??? ????? ????? ????? ????? 10", illegal},                             // reserved AREDUCE encodings
    // VREDUCE (R2): funct3 = 000 sum, 001 max, 010 min, 011 sumsq; argmax
    // uses the R4 two-destination variant (xVal5 in the src2 position).
    {"0010 000 0?? 0000000000 ????? ????? 10", decode_vreduce},
    {"0010 000 100 00000 ????? ????? ????? 10", decode_vreduce_argmax},
    {"0010 ??? ??? ????? ????? ????? ????? 10", illegal},                             // reserved VREDUCE encodings
    // QNT (R4): sel3 = {dom2, IS}, vScale5 in arg5, index5 in the src2
    // position; one pattern, the handler rejects the reserved funct3 and
    // matrix-form IS/index5 encodings.
    {"0100 ??? ??? ????? ????? ????? ????? 10", decode_qnt},
    // CFG_MAT_SETI / CFG_VEC_SETI (I format): the handler checks the field
    // code against the selected config register type.
    {"0101 ?? ??? ???? ???? ???? ???? ????? 10", decode_cfg_seti},
    {"0110 ?? ??? ???? ???? ???? ???? ????? 10", decode_cfg_seti},

    // CFG_REG (R4): funct3 000/001/010 = setx/copy/get; field5 high bits and
    // reserved5 must be 0; copy requires field5 = 0. The handlers check
    // setx/get field codes against the selected config register type.
    {"0111 ??? 000 00??? 00000 ????? ????? 10", decode_cfg_reg<CfgOp::setx>},
    {"0111 ??? 001 00000 00000 ????? ????? 10", decode_cfg_reg<CfgOp::copy>},
    {"0111 ??? 010 00??? 00000 ????? ????? 10", decode_cfg_reg<CfgOp::get>},
    {"0111 ??? ??? ????? ????? ????? ????? 10", illegal},

    // SYS (Z format): one pattern pins reserved23 to 0; the handler rejects
    // the reserved funct3 codes 100-111.
    {"1000 ??? 00000000000000000000000 10", decode_sys},
    {"1000 ??? ????? ????? ????? ????? ??? 10", illegal},                           // reserved SYS encodings

    // CFG_SHAPE (R4 layout): every field is an immediate; the handler rejects
    // the reserved SAT/cols5 and out-of-range register spans.
    {"1001 ??? ??? ????? ????? ????? ????? 10", decode_cfg_shape},

    {"???? ??? ???? ???? ????? ????? ????? 10", illegal},                           // reserved major encodings
};

// clang-format on

} // namespace

DecodeResult decode_npu(std::uint32_t word) { return match(NPU_TABLE, word); }

} // namespace tile_npu
