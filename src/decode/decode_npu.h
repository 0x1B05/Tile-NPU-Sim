#pragma once

#include "tile_npu/decode.h"

#include <cstdint>

namespace tile_npu {

std::optional<DecodedInstr> decode_npu(std::uint32_t word);

NpuDomain sel3_domain(std::uint8_t sel3);
IndexOperand decode_index(RegisterIndex raw, bool is_scalar);

// Elementwise decoders.
std::optional<DecodedInstr> decode_ebin(std::uint32_t word);
std::optional<DecodedInstr> decode_eun(std::uint32_t word);
std::optional<DecodedInstr> decode_ecmp(std::uint32_t word);
std::optional<DecodedInstr> decode_ebcast(std::uint32_t word);
std::optional<DecodedInstr> decode_er4(std::uint32_t word);
std::optional<DecodedInstr> decode_emask(std::uint32_t word);

// Numeric decoders.
std::optional<DecodedInstr> decode_mma(std::uint32_t word);
std::optional<DecodedInstr> decode_areduce(std::uint32_t word);
std::optional<DecodedInstr> decode_areduce_argmax(std::uint32_t word);
std::optional<DecodedInstr> decode_vreduce(std::uint32_t word);
std::optional<DecodedInstr> decode_vreduce_argmax(std::uint32_t word);
std::optional<DecodedInstr> decode_qnt(std::uint32_t word);

// Memory and movement decoders.
std::optional<DecodedInstr> decode_mblk_load(std::uint32_t word);
std::optional<DecodedInstr> decode_mblk_store(std::uint32_t word);
std::optional<DecodedInstr> decode_mblk_gather(std::uint32_t word);
std::optional<DecodedInstr> decode_fill(std::uint32_t word);
std::optional<DecodedInstr> decode_fillx(std::uint32_t word);
std::optional<DecodedInstr> decode_move_copy(std::uint32_t word);
std::optional<DecodedInstr> decode_row_insert(std::uint32_t word);
std::optional<DecodedInstr> decode_row_extract(std::uint32_t word);
std::optional<DecodedInstr> decode_lane_extract(std::uint32_t word);
std::optional<DecodedInstr> decode_lane_insert(std::uint32_t word);
std::optional<DecodedInstr> decode_lane_broadcast(std::uint32_t word);

// Configuration and system decoders.
std::optional<DecodedInstr> decode_cfg_seti(std::uint32_t word);
std::optional<DecodedInstr> decode_cfg_setx(std::uint32_t word);
std::optional<DecodedInstr> decode_cfg_copy(std::uint32_t word);
std::optional<DecodedInstr> decode_cfg_get(std::uint32_t word);
std::optional<DecodedInstr> decode_cfg_setw(std::uint32_t word);
std::optional<DecodedInstr> decode_cfg_getw(std::uint32_t word);
std::optional<DecodedInstr> decode_sys(std::uint32_t word);

} // namespace tile_npu
