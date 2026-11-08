#include "uarch/event_engine.h"

#include "utils/common.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

constexpr std::uint64_t SINGLE_CYCLE = 1;
constexpr std::size_t UNIT_COUNT = 6;

std::size_t unit_index(Unit unit) { return static_cast<std::size_t>(unit); }

} // namespace

void EventUarchEngine::reset() {
  state_ = UarchState{};
  scoreboard_.reset();
  mma_.reset();
  dma_.reset();
  vector_.reset();
}

void EventUarchEngine::dispatch(const DynInst &inst) {
  if (scoreboard_.size() >= WINDOW_DEPTH) {
    unreachable("dispatch into a full uarch window");
  }

  state_.events.run_until(state_.events.now());

  // A fence is a completion boundary for the backends it names. Waiting here
  // also delays every younger instruction, because dispatch advances in
  // program order.
  if (inst.unit == Unit::sync) fence(inst.fence);

  ++state_.stats.unit_count[unit_index(inst.unit)];
  scoreboard_.dispatch(inst);
  issue_ready(state_.events.now());

  // The front end accepts one instruction per cycle.
  state_.events.run_until(state_.events.now() + SINGLE_CYCLE);
  state_.stats.cycles = state_.events.now();
}

bool EventUarchEngine::issue_ready(std::uint64_t now) {
  bool issued = false;
  std::array<bool, UNIT_COUNT> unit_busy{};
  bool memory_seen = false;

  for (const auto &[id, entry] : scoreboard_.entries()) {
    if (entry.issued) continue;

    if (entry.inst.unit == Unit::dma) {
      // The memory queue issues in program order. Only the oldest unissued
      // memory operation is eligible, and a younger one may not pass it while
      // it waits for dependencies.
      if (memory_seen) continue;
      memory_seen = true;
    }

    const auto index = unit_index(entry.inst.unit);
    if (unit_busy[index] || !entry.deps.empty()) continue;

    unit_busy[index] = true;
    issue_one(id, now);
    issued = true;
  }

  // Instructions left over because their unit already issued this cycle get
  // another look next cycle; completion events alone would serialize them.
  if (issued && has_ready_work()) {
    state_.events.schedule(now + SINGLE_CYCLE, [this] { issue_ready(state_.events.now()); });
  }
  return issued;
}

void EventUarchEngine::issue_one(std::uint64_t id, std::uint64_t now) {
  const auto &inst = scoreboard_.inst(id);
  scoreboard_.mark_issued(id);

  std::uint64_t finish = now + SINGLE_CYCLE;
  switch (inst.unit) {
  case Unit::mma: finish = mma_.issue(inst, now); break;
  case Unit::dma: finish = dma_.issue(inst, now); break;
  case Unit::vector: finish = vector_.issue(inst, now); break;
  case Unit::scalar:
  case Unit::config:
  case Unit::sync: break;
  }

  state_.events.schedule(finish, [this, id] { complete(id); });
}

void EventUarchEngine::complete(std::uint64_t id) {
  if (!scoreboard_.complete(id)) return;
  ++state_.stats.retired;
  issue_ready(state_.events.now());
  state_.stats.cycles = state_.events.now();
}

void EventUarchEngine::fence(FenceKind kind) {
  while (has_work(kind))
    advance();
  state_.stats.cycles = state_.events.now();
}

bool EventUarchEngine::can_accept() const { return scoreboard_.size() < WINDOW_DEPTH; }

void EventUarchEngine::advance() {
  if (const auto next = state_.events.next_time()) {
    state_.events.run_until(*next);
  } else if (!issue_ready(state_.events.now())) {
    unreachable("uarch window cannot make progress");
  }
  state_.stats.cycles = state_.events.now();
}

void EventUarchEngine::finish() {
  while (!scoreboard_.empty())
    advance();
  state_.stats.cycles = state_.events.now();
}

bool EventUarchEngine::has_work(FenceKind kind) const {
  for (const auto &pair : scoreboard_.entries()) {
    const auto unit = pair.second.inst.unit;
    const bool covered = kind == FenceKind::all || (kind == FenceKind::mem && unit == Unit::dma) ||
                         (kind == FenceKind::sa && unit == Unit::mma);
    if (covered) return true;
  }
  return false;
}

bool EventUarchEngine::has_ready_work() const {
  bool memory_seen = false;
  for (const auto &pair : scoreboard_.entries()) {
    const auto &entry = pair.second;
    if (entry.issued) continue;
    if (entry.inst.unit == Unit::dma) {
      if (memory_seen) continue;
      memory_seen = true;
    }
    if (entry.deps.empty()) return true;
  }
  return false;
}

const UarchStats &EventUarchEngine::stats() const { return state_.stats; }

} // namespace tile_npu
