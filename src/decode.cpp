#include "tile_npu/decode.h"

#include "decode_rv64im.h"
#include "encoding.h"
#include "pattern.h"
#include "tile_npu/state.h"

#include <cstdint>

namespace tile_npu {
namespace {

// E_BIN add on the tile domain with BX=0. The pattern pins sel3, so only the
// register bounds remain to check here.
DecodeResult decode_tadd(std::uint32_t word) {
  const auto f = encoding::NpuR3::from(word);
  if (f.dest >= TILE_REGISTERS || f.src1 >= TILE_REGISTERS ||
      f.src2 >= TILE_REGISTERS) {
    return DecodeError::illegal;
  }
  return ElementwiseBinaryInstruction{ElementwiseBinaryOp::add,
                                      {NpuDomain::tile, f.dest},
                                      {NpuDomain::tile, f.src1},
                                      NpuRegister{NpuDomain::tile, f.src2}};
}

// Defined E_BIN funct4 values other than add are not implemented yet;
// reserved funct4 values are illegal.
DecodeResult decode_binary_other(std::uint32_t word) {
  switch (static_cast<ElementwiseBinaryOp>(encoding::field<24, 21>(word))) {
  case ElementwiseBinaryOp::sub:
  case ElementwiseBinaryOp::mul:
  case ElementwiseBinaryOp::div:
  case ElementwiseBinaryOp::min:
  case ElementwiseBinaryOp::max:
  case ElementwiseBinaryOp::bit_and:
  case ElementwiseBinaryOp::bit_or:
  case ElementwiseBinaryOp::bit_xor:
  case ElementwiseBinaryOp::shl:
  case ElementwiseBinaryOp::shr:
  case ElementwiseBinaryOp::sra: return DecodeError::not_implemented;
  default: return DecodeError::illegal;
  }
}

// clang-format off

// Pattern fields: major4 sel3 funct4 reserved4 src2 src1 dest q.
// E_BIN reads sel3 as {dom2, BX}: dom tile=00 acc=01 vec8=10 vec32=11.
constexpr DecodeEntry NPU_TABLE[] = {
    {"0000 000 0000 0000 ????? ????? ????? 00", decode_tadd},
    {"0000 0?? 0000 0000 ????? ????? ????? 00", fail<DecodeError::not_implemented>}, // taddx, add @acc
    {"0000 10? ???? 0000 ????? ????? ????? 00", fail<DecodeError::illegal>},          // vec8 has no E_BIN
    {"0000 11? 0000 0000 ????? ????? ????? 00", fail<DecodeError::not_implemented>},  // add @vec32
    {"0000 ??? ???? 0000 ????? ????? ????? 00", decode_binary_other},
    {"0000 ??? ???? ???? ????? ????? ????? 00", fail<DecodeError::illegal>},          // reserved4 != 0
    {"???? ??? ???? ???? ????? ????? ????? 00", fail<DecodeError::not_implemented>},  // other elementwise majors
    {"???? ??? ???? ???? ????? ????? ????? 01", fail<DecodeError::not_implemented>},  // data movement
    {"???? ??? ???? ???? ????? ????? ????? 10", fail<DecodeError::not_implemented>},  // cross-domain/control
};

// clang-format on

} // namespace

DecodeResult decode(std::uint32_t word) {
  if (encoding::field<1, 0>(word) == 0b11) {
    return decode_rv64im(word);
  }
  return match(NPU_TABLE, word);
}

} // namespace tile_npu
