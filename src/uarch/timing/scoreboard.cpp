#include "uarch/timing/scoreboard.h"

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

std::size_t Scoreboard::domain_index(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return 0;
  case NpuDomain::acc: return 1;
  case NpuDomain::vec32: return 2;
  case NpuDomain::reserved: break;
  }
  unreachable("reserved NPU domain in the scoreboard");
}

void Scoreboard::validate(const RegisterRef &ref) {
  const auto domain = domain_index(ref.domain);
  if (ref.index >= DOMAIN_REGISTERS[domain]) {
    unreachable("dynamic instruction references an invalid register");
  }
}

std::vector<std::uint64_t> Scoreboard::dependencies(const DynInst &inst) const {
  std::vector<std::uint64_t> deps;

  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    const auto &reg = regs_[domain_index(src.domain)][src.index];
    if (reg.busy) deps.push_back(reg.producer);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    const auto &reg = regs_[domain_index(dst.domain)][dst.index];
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
    regs_[domain_index(src.domain)][src.index].readers.push_back(inst.id);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    auto &reg = regs_[domain_index(dst.domain)][dst.index];
    reg.busy = true;
    reg.producer = inst.id;
  }
}

void Scoreboard::dispatch(const DynInst &inst) {
  if (inst.src_count > inst.src.size() || inst.dst_count > inst.dst.size()) {
    unreachable("dynamic instruction operand count exceeds capacity");
  }
  for (std::uint8_t i = 0; i < inst.src_count; ++i)
    validate(inst.src[i]);
  for (std::uint8_t i = 0; i < inst.dst_count; ++i)
    validate(inst.dst[i]);

  auto deps = dependencies(inst);
  register_operands(inst);
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
    std::erase(regs_[domain_index(src.domain)][src.index].readers, id);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    auto &reg = regs_[domain_index(dst.domain)][dst.index];
    if (reg.producer == id) reg.busy = false;
  }

  for (auto &pair : entries_) {
    std::erase(pair.second.deps, id);
  }
  entries_.erase(it);
  return true;
}

} // namespace tile_npu
