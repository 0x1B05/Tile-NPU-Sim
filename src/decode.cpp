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
enum class Domain : std::uint8_t {
  tile = 0b00,
  acc = 0b01,
  vec8 = 0b10,
  vec32 = 0b11,
};
enum class BinaryFunction : std::uint8_t {
  add = 0b0000,
  sub = 0b0001,
  mul = 0b0010,
  div = 0b0011,
  min = 0b0100,
  max = 0b0101,
  bit_and = 0b1000,
  bit_or = 0b1001,
  bit_xor = 0b1010,
  shl = 0b1011,
  shr = 0b1100,
  sra = 0b1101,
};

DecodeResult decode_binary_add(const encoding::NpuR3 &fields,
                               const encoding::BinarySelector &selector) {
  switch (static_cast<Domain>(selector.domain)) {
  case Domain::tile:
    if (selector.scalar_source) {
      return DecodeError::not_implemented; // taddx
    }
    if (fields.dest >= kTileRegisters || fields.src1 >= kTileRegisters ||
        fields.src2 >= kTileRegisters) {
      return DecodeError::illegal;
    }
    return DecodedNpuInstr{NpuOp::tadd,
                                 {fields.dest, fields.src1, fields.src2}};
  case Domain::acc:
  case Domain::vec32:
    return DecodeError::not_implemented;
  case Domain::vec8:
    return DecodeError::illegal;
  }
  return DecodeError::illegal;
}

DecodeResult decode_binary(std::uint32_t word) {
  const auto fields = encoding::NpuR3::from(word);
  const auto selector = encoding::BinarySelector::from(fields.sel3);
  if (fields.reserved4 != 0 ||
      selector.domain == static_cast<std::uint8_t>(Domain::vec8)) {
    return DecodeError::illegal;
  }

  switch (static_cast<BinaryFunction>(fields.funct4)) {
  case BinaryFunction::add:
    return decode_binary_add(fields, selector);
  case BinaryFunction::sub:
  case BinaryFunction::mul:
  case BinaryFunction::div:
  case BinaryFunction::min:
  case BinaryFunction::max:
  case BinaryFunction::bit_and:
  case BinaryFunction::bit_or:
  case BinaryFunction::bit_xor:
  case BinaryFunction::shl:
  case BinaryFunction::shr:
  case BinaryFunction::sra:
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
