#pragma once

#include "uarch/dataflow/dyn_inst.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace tile_npu {

// Register scoreboard and in-flight capacity for the NPU data domains. The front
// end runs ahead of timing, so hazards are resolved from dispatch-time DynInst
// snapshots rather than from the live State:
//   * RAW: a source with a live producer waits for that producer;
//   * WAW: a destination with an old writer waits for the old writer;
//   * WAR: a destination waits for every live reader of the old value.
//
// Scalar operands are resolved before an instruction reaches the scoreboard and
// never appear in DynInst, so the scoreboard only tracks tile, acc, and vec32.
//
// Entries are keyed by the monotonically increasing DynInst id, which is also
// dispatch order. Completion may happen out of order; each entry releases its
// own register references by id.
class Scoreboard {
public:
  struct Entry {
    DynInst inst{};
    std::vector<std::uint64_t> deps;
    bool issued = false;
  };

  void reset();

  bool empty() const { return entries_.empty(); }

  std::size_t size() const { return entries_.size(); }

  // Adds one instruction and records its hazards. The caller owns capacity.
  void dispatch(const DynInst &inst);

  bool ready(std::uint64_t id) const;

  const DynInst &inst(std::uint64_t id) const;

  void mark_issued(std::uint64_t id);

  // Releases register references and erases the entry. Returns false when the
  // id is not in the scoreboard (for example a stale event after reset).
  bool complete(std::uint64_t id);

  // Entries in dispatch order.
  const std::map<std::uint64_t, Entry> &entries() const { return entries_; }

private:
  struct RegisterStat {
    bool busy = false;
    std::uint64_t producer = 0;
    std::vector<std::uint64_t> readers;
  };

  // NpuDomain values are tile=0, acc=1, reserved=2, vec32=3. The builder
  // rejects reserved and out-of-range registers, so the reserved row is never
  // touched.
  static constexpr std::size_t DOMAIN_COUNT = static_cast<std::size_t>(NpuDomain::vec32) + 1;
  static constexpr std::size_t REG_COUNT =
      std::max({TILE_REGISTERS, ACC_REGISTERS, VECTOR_REGISTERS});

  // Producer ids this instruction must wait for. Must run before update_reg_stats().
  std::vector<std::uint64_t> dependencies(const DynInst &inst) const;
  void update_reg_stats(const DynInst &inst);

  std::array<std::array<RegisterStat, REG_COUNT>, DOMAIN_COUNT> regs_{};
  std::map<std::uint64_t, Entry> entries_;
};

} // namespace tile_npu
