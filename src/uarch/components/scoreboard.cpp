#include "uarch/components/scoreboard.h"

#include "utils/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tile_npu {

void Scoreboard::reset() {
  regs_ = {};
  entries_.clear();
}

std::vector<std::uint64_t> Scoreboard::dependencies(const DynInst &inst) const {
  std::vector<std::uint64_t> deps;

  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    const auto &src_stat = regs_[static_cast<std::size_t>(src.domain)][src.index];
    if (src_stat.busy) deps.push_back(src_stat.producer);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    const auto &dst_stat = regs_[static_cast<std::size_t>(dst.domain)][dst.index];
    if (dst_stat.busy) deps.push_back(dst_stat.producer);                      // WAW
    deps.insert(deps.end(), dst_stat.readers.begin(), dst_stat.readers.end()); // WAR
  }

  std::sort(deps.begin(), deps.end());
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
  return deps;
}

void Scoreboard::update_reg_stats(const DynInst &inst) {
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    auto &src_stat = regs_[static_cast<std::size_t>(src.domain)][src.index];
    src_stat.readers.push_back(inst.id);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    auto &dst_stat = regs_[static_cast<std::size_t>(dst.domain)][dst.index];
    dst_stat.busy = true;
    dst_stat.producer = inst.id;
  }
}

void Scoreboard::dispatch(const DynInst &inst) {
  if (inst.src_count > inst.src.size() || inst.dst_count > inst.dst.size()) {
    unreachable("dynamic instruction operand count exceeds capacity");
  }

  auto deps = dependencies(inst);
  update_reg_stats(inst);
  const auto inserted = entries_.emplace(inst.id, Entry{.inst = inst, .deps = std::move(deps)});
  if (!inserted.second) unreachable("duplicate dynamic instruction id");
}

bool Scoreboard::ready(std::uint64_t id) const {
  const auto it = entries_.find(id);
  return it != entries_.end() && it->second.deps.empty();
}

const DynInst &Scoreboard::inst(std::uint64_t id) const {
  const auto it = entries_.find(id);
  if (it == entries_.end()) unreachable("dynamic instruction is not in the scoreboard");
  return it->second.inst;
}

void Scoreboard::mark_issued(std::uint64_t id) {
  const auto it = entries_.find(id);
  if (it == entries_.end()) unreachable("issued instruction is not in the scoreboard");
  it->second.issued = true;
}

bool Scoreboard::complete(std::uint64_t id) {
  const auto it = entries_.find(id);
  if (it == entries_.end()) return false;

  const auto &inst = it->second.inst;
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    auto &src_stat = regs_[static_cast<std::size_t>(src.domain)][src.index];
    std::erase(src_stat.readers, id);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    auto &dst_stat = regs_[static_cast<std::size_t>(dst.domain)][dst.index];
    if (dst_stat.producer == id) dst_stat.busy = false;
  }

  for (auto &pair : entries_) {
    std::erase(pair.second.deps, id);
  }
  entries_.erase(it);
  return true;
}

} // namespace tile_npu
