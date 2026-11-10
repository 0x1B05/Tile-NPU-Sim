#pragma once

#include "uarch/dyn_inst.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace tile_npu {

// Register scoreboard and dispatch window for the NPU data domains. The front
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
  // id is not in the window (for example a stale event after reset).
  bool complete(std::uint64_t id);

  // Entries in dispatch order.
  const std::map<std::uint64_t, Entry> &entries() const { return entries_; }

private:
  struct RegisterStat {
    bool busy = false;
    std::uint64_t producer = 0;
    std::vector<std::uint64_t> readers;
  };

  static constexpr std::size_t DOMAIN_COUNT = 3;
  static constexpr std::array<std::size_t, DOMAIN_COUNT> DOMAIN_REGISTERS{
      TILE_REGISTERS, ACC_REGISTERS, VECTOR_REGISTERS};
  static constexpr std::size_t REG_COUNT =
      std::max({TILE_REGISTERS, ACC_REGISTERS, VECTOR_REGISTERS});

  static std::size_t domain_index(NpuDomain domain);
  static void validate(const RegisterRef &ref);

  // Producer ids this instruction must wait for. Must run before
  // register_operands(), so a read-modify-write operand does not self-block.
  std::vector<std::uint64_t> dependencies(const DynInst &inst) const;
  void register_operands(const DynInst &inst);

  std::array<std::array<RegisterStat, REG_COUNT>, DOMAIN_COUNT> regs_{};
  std::map<std::uint64_t, Entry> entries_;
};

} // namespace tile_npu
