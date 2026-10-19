#pragma once

#include "tile_npu/decode.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

DecodeResult decode_npu(std::uint32_t word);

struct PairedDomains {
  NpuDomain matrix;
  NpuDomain vector;
};

NpuDomain sel3_domain(std::uint8_t sel3);
PairedDomains paired_domains(std::uint8_t sel3);
std::optional<NpuOperand> decode_bx_src2(std::uint8_t sel3, std::uint8_t src2, NpuDomain domain);
IndexOperand decode_index(RegisterIndex raw, bool is_scalar);
ConfigRegType cfg_reg_type(std::uint8_t sel2, bool vector_side);

// Elementwise decoders.
DecodeResult decode_ebin(std::uint32_t word);
DecodeResult decode_eun(std::uint32_t word);
DecodeResult decode_ecmp(std::uint32_t word);
DecodeResult decode_ebcast(std::uint32_t word);
DecodeResult decode_er4(std::uint32_t word);
DecodeResult decode_emask(std::uint32_t word);

// Numeric decoders.
DecodeResult decode_mma(std::uint32_t word);
DecodeResult decode_cvt(std::uint32_t word);
DecodeResult decode_areduce(std::uint32_t word);
DecodeResult decode_vreduce(std::uint32_t word);
DecodeResult decode_vreduce_argmax(std::uint32_t word);
DecodeResult decode_qnt(std::uint32_t word);

// Memory and movement decoders.
DecodeResult decode_mblk_load(std::uint32_t word);
DecodeResult decode_mblk_store(std::uint32_t word);
DecodeResult decode_mblk_gather(std::uint32_t word);
DecodeResult decode_mblk_row_load(std::uint32_t word);
DecodeResult decode_mblk_row_store(std::uint32_t word);
DecodeResult decode_fill(std::uint32_t word);
DecodeResult decode_fillx(std::uint32_t word);
DecodeResult decode_move_copy(std::uint32_t word);
DecodeResult decode_move_transpose(std::uint32_t word);
DecodeResult decode_row_insert(std::uint32_t word);
DecodeResult decode_row_extract(std::uint32_t word);
DecodeResult decode_lane_extract(std::uint32_t word);
DecodeResult decode_lane_insert(std::uint32_t word);
DecodeResult decode_lane_broadcast(std::uint32_t word);

// Configuration and system decoders.
DecodeResult decode_cfg_seti(std::uint32_t word);
DecodeResult decode_cfg_setx(std::uint32_t word);
DecodeResult decode_cfg_copy(std::uint32_t word);
DecodeResult decode_cfg_get(std::uint32_t word);
DecodeResult decode_cfg_setw(std::uint32_t word);
DecodeResult decode_cfg_getw(std::uint32_t word);
DecodeResult decode_sys(std::uint32_t word);

} // namespace tile_npu
