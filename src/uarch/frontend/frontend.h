#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/state.h"
#include "uarch/backend/backend.h"
#include "uarch/components/inst_buffer.h"
#include "uarch/components/scalar_writeback.h"
#include "uarch/engine/clock.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <variant>

namespace tile_npu {

class NpuExecutor;

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

class UarchFrontend final {
public:
  UarchFrontend(const UarchClock &clock, UarchBackend &backend, State &state, NpuExecutor &executor,
                ScalarWritebackPort &writeback);

  bool enqueue(InstBuffer::Entry entry);

  std::size_t process_cur_cycle();
  std::optional<UarchClock::Time> next_activity_time() const;
  bool empty() const;
  void reset();

private:
  const UarchClock &clock_;
  UarchBackend &backend_;
  State &state_;
  NpuExecutor &executor_;
  ScalarWritebackPort &writeback_;
  InstBuffer inst_buffer_;
  std::uint64_t next_id_ = 0;
};

} // namespace tile_npu
