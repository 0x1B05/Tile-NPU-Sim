#pragma once

#include "tile_npu/decode.h"
#include "uarch/engine/clock.h"

#include <cstddef>
#include <optional>
#include <type_traits>
#include <variant>

namespace tile_npu {

// NPU-space instructions that execute in the scalar front end.
template <typename T>
inline constexpr bool is_frontend_npu_v =
    std::is_same_v<T, CfgSetiInstr> || std::is_same_v<T, CfgSetxInstr> ||
    std::is_same_v<T, CfgCopyInstr> || std::is_same_v<T, CfgGetInstr> ||
    std::is_same_v<T, CfgSetwInstr> || std::is_same_v<T, CfgGetwInstr> ||
    std::is_same_v<T, SysInstr>;

inline bool is_frontend_npu(const DecodedNpuInstr &instr) {
  return std::visit(
      [](const auto &decoded) { return is_frontend_npu_v<std::remove_cvref_t<decltype(decoded)>>; },
      instr);
}

class UarchFrontend {
public:
  virtual ~UarchFrontend() = default;

  // process_cur_cycle() runs at clock.now() and returns the number of completed instructions;
  virtual std::size_t process_cur_cycle() = 0;
  // next_activity_time() reports the next time the front end may need another cycle.
  virtual std::optional<UarchClock::Time> next_activity_time() const = 0;
  virtual bool empty() const = 0;
};

} // namespace tile_npu
