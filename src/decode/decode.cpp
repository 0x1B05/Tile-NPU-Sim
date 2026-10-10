#include "tile_npu/decode.h"

#include "decode_npu.h"
#include "decode_rv64im.h"
#include "encoding.h"

#include <cstdint>

namespace tile_npu {

DecodeResult decode(std::uint32_t word) {
  if (encoding::field<1, 0>(word) == encoding::QUADRANT_RV64IM) {
    return decode_rv64im(word);
  }
  return decode_npu(word);
}

} // namespace tile_npu
