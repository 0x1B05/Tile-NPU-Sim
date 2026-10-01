#include "tile_npu/decode.h"

#include "decode_rv64im.h"
#include "encoding.h"
#include "tile_npu/state.h"

#include <cstdint>

namespace tile_npu {
namespace {

enum class Quadrant : std::uint8_t {
  elementwise = 0b00,
  data_movement = 0b01,
  cross_domain = 0b10,
  scalar = 0b11,
};

enum class ElementwiseMajor : std::uint8_t { binary = 0b0000 };
DecodeResult decode_binary_add(const encoding::NpuR3 &fields,
                               const encoding::BinarySelector &selector) {
  switch (static_cast<NpuDomain>(selector.domain)) {
  case NpuDomain::tile:
    if (selector.scalar_source) {
      return DecodeError::not_implemented; // taddx
    }
    if (fields.dest >= kTileRegisters || fields.src1 >= kTileRegisters ||
        fields.src2 >= kTileRegisters) {
      return DecodeError::illegal;
    }
    return ElementwiseBinaryInstruction{
        ElementwiseBinaryOp::add,
        {NpuDomain::tile, fields.dest},
        {NpuDomain::tile, fields.src1},
        NpuRegister{NpuDomain::tile, fields.src2}};
  case NpuDomain::acc:
  case NpuDomain::vec32:
    return DecodeError::not_implemented;
  case NpuDomain::vec8:
    return DecodeError::illegal;
  }
  return DecodeError::illegal;
}

DecodeResult decode_binary(std::uint32_t word) {
  const auto fields = encoding::NpuR3::from(word);
  const auto selector = encoding::BinarySelector::from(fields.sel3);
  if (fields.reserved4 != 0 ||
      selector.domain == static_cast<std::uint8_t>(NpuDomain::vec8)) {
    return DecodeError::illegal;
  }

  switch (static_cast<ElementwiseBinaryOp>(fields.funct4)) {
  case ElementwiseBinaryOp::add:
    return decode_binary_add(fields, selector);
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
  case ElementwiseBinaryOp::sra:
    return DecodeError::not_implemented;
  default:
    return DecodeError::illegal;
  }
}

DecodeResult decode_elementwise(std::uint32_t word) {
  switch (static_cast<ElementwiseMajor>(encoding::field<31, 28>(word))) {
  case ElementwiseMajor::binary:
    return decode_binary(word);
  default:
    return DecodeError::not_implemented;
  }
}

} // namespace

DecodeResult decode(std::uint32_t word) {
  switch (static_cast<Quadrant>(encoding::field<1, 0>(word))) {
  case Quadrant::scalar:
    return decode_rv64im(word);
  case Quadrant::elementwise:
    return decode_elementwise(word);
  case Quadrant::data_movement:
  case Quadrant::cross_domain:
    return DecodeError::not_implemented;
  }
  return DecodeError::illegal;
}

} // namespace tile_npu
