#include "uarch/scoreboard.h"

#include "utils/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tile_npu {

void Scoreboard::reset() {
  regs_ = {};
  window_.clear();
}

bool Scoreboard::is_zero(const DataRef &ref) {
  return ref.domain == Domain::scalar && ref.index == 0;
}

Scoreboard::RegisterStat &Scoreboard::stat(const DataRef &ref) {
  const auto domain = static_cast<std::size_t>(ref.domain);
  if (domain >= DOMAIN_COUNT || ref.index >= REG_COUNT) {
    unreachable("dynamic instruction references an invalid register");
  }
  return regs_[domain][ref.index];
}

void Scoreboard::dispatch(const DynInst &inst) {
  std::vector<std::uint64_t> deps;

  // Snapshot the hazards before registering this instruction, so a register
  // that is both a source and a destination does not depend on itself.
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    if (is_zero(src)) continue;
    const auto &reg = stat(src);
    if (reg.busy) deps.push_back(reg.producer);
  }
  if (inst.dst && !is_zero(*inst.dst)) {
    const auto &reg = stat(*inst.dst);
    if (reg.busy) deps.push_back(reg.producer);                      // WAW
    deps.insert(deps.end(), reg.readers.begin(), reg.readers.end()); // WAR
  }

  std::sort(deps.begin(), deps.end());
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());

  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    if (!is_zero(src)) stat(src).readers.push_back(inst.id);
  }
  if (inst.dst && !is_zero(*inst.dst)) {
    auto &reg = stat(*inst.dst);
    reg.busy = true;
    reg.producer = inst.id;
  }

  const auto inserted = window_.emplace(inst.id, Entry{.inst = inst, .deps = std::move(deps)});
  if (!inserted.second) unreachable("duplicate dynamic instruction id");
}

bool Scoreboard::ready(std::uint64_t id) const {
  const auto it = window_.find(id);
  return it != window_.end() && it->second.deps.empty();
}

const DynInst &Scoreboard::inst(std::uint64_t id) const { return window_.at(id).inst; }

void Scoreboard::mark_issued(std::uint64_t id) { window_.at(id).issued = true; }

bool Scoreboard::complete(std::uint64_t id) {
  const auto it = window_.find(id);
  if (it == window_.end()) return false;

  const auto &inst = it->second.inst;
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    if (!is_zero(src)) std::erase(stat(src).readers, id);
  }
  if (inst.dst && !is_zero(*inst.dst)) {
    auto &reg = stat(*inst.dst);
    if (reg.producer == id) reg.busy = false;
  }

  for (auto &pair : window_) {
    std::erase(pair.second.deps, id);
  }
  window_.erase(it);
  return true;
}

} // namespace tile_npu
