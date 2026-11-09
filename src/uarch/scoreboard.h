#pragma once

#include "uarch/dyn_inst.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace tile_npu {

// Register scoreboard and dispatch window for the uarch engine.
//
// The functional front end runs ahead of timing, so hazards are resolved from
// dispatch-time snapshots rather than from the live State:
//   * RAW: a source with a live producer waits for that producer;
//   * WAW: a destination with an old writer waits for the old writer;
//   * WAR: a destination waits for every live reader of the old value.
// x0 is hard-wired and therefore neither produces nor consumes dependencies.
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

  bool empty() const { return window_.empty(); }

  std::size_t size() const { return window_.size(); }

  // Adds one instruction and records its hazards. The caller owns capacity.
  void dispatch(const DynInst &inst);

  bool ready(std::uint64_t id) const;

  const DynInst &inst(std::uint64_t id) const;

  void mark_issued(std::uint64_t id);

  // Releases register references and erases the entry. Returns false when the
  // id is not in the window (for example a stale event after reset).
  bool complete(std::uint64_t id);

  // Entries in program order.
  const std::map<std::uint64_t, Entry> &entries() const { return window_; }

private:
  struct RegisterStat {
    bool busy = false;
    std::uint64_t producer = 0;
    std::vector<std::uint64_t> readers;
  };

  // Domain order matches the enum: scalar, tile, acc, vec32.
  static constexpr std::array<std::size_t, DOMAIN_COUNT> DOMAIN_REGISTERS{
      SCALAR_REGISTERS, TILE_REGISTERS, ACC_REGISTERS, VECTOR_REGISTERS};
  static constexpr std::size_t REG_COUNT =
      std::max({SCALAR_REGISTERS, TILE_REGISTERS, ACC_REGISTERS, VECTOR_REGISTERS});

  static bool is_zero(const DataRef &ref);

  // Domain and index of a legal register reference. Aborts on an operand the
  // functional builder could not have produced.
  static std::pair<std::size_t, std::size_t> locate(const DataRef &ref);

  const RegisterStat &stat(const DataRef &ref) const;
  RegisterStat &stat(const DataRef &ref);

  // Producer ids this instruction must wait for. Must run before
  // register_operands(), so a read-modify-write operand does not self-block.
  std::vector<std::uint64_t> dependencies(const DynInst &inst) const;
  void register_operands(const DynInst &inst);

  std::array<std::array<RegisterStat, REG_COUNT>, DOMAIN_COUNT> regs_{};
  std::map<std::uint64_t, Entry> window_;
};

} // namespace tile_npu
