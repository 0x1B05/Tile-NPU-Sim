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

std::pair<std::size_t, std::size_t> Scoreboard::locate(const DataRef &ref) {
  const auto domain = static_cast<std::size_t>(ref.domain);
  if (domain >= DOMAIN_COUNT || ref.index >= DOMAIN_REGISTERS[domain]) {
    unreachable("dynamic instruction references an invalid register");
  }
  return {domain, ref.index};
}

const Scoreboard::RegisterStat &Scoreboard::stat(const DataRef &ref) const {
  const auto [domain, index] = locate(ref);
  return regs_[domain][index];
}

Scoreboard::RegisterStat &Scoreboard::stat(const DataRef &ref) {
  const auto [domain, index] = locate(ref);
  return regs_[domain][index];
}

std::vector<std::uint64_t> Scoreboard::dependencies(const DynInst &inst) const {
  std::vector<std::uint64_t> deps;

  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    if (is_zero(src)) continue;
    const auto &reg = stat(src);
    if (reg.busy) deps.push_back(reg.producer);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    if (is_zero(dst)) continue;
    const auto &reg = stat(dst);
    if (reg.busy) deps.push_back(reg.producer);                      // WAW
    deps.insert(deps.end(), reg.readers.begin(), reg.readers.end()); // WAR
  }

  std::sort(deps.begin(), deps.end());
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
  return deps;
}

void Scoreboard::register_operands(const DynInst &inst) {
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    if (!is_zero(src)) stat(src).readers.push_back(inst.id);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    if (is_zero(dst)) continue;
    auto &reg = stat(dst);
    reg.busy = true;
    reg.producer = inst.id;
  }
}

void Scoreboard::dispatch(const DynInst &inst) {
  auto deps = dependencies(inst);
  register_operands(inst);
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
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    if (is_zero(dst)) continue;
    auto &reg = stat(dst);
    if (reg.producer == id) reg.busy = false;
  }

  for (auto &pair : window_) {
    std::erase(pair.second.deps, id);
  }
  window_.erase(it);
  return true;
}

} // namespace tile_npu
