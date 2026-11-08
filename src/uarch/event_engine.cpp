#include "uarch/event_engine.h"

#include "utils/common.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

constexpr std::uint64_t SINGLE_CYCLE = 1;

} // namespace

void EventUarchEngine::reset() {
  state_ = UarchState{};
  scoreboard_.reset();
  mma_.reset();
  dma_.reset();
  vector_.reset();
  issue_cycle_.reset();
  unit_used_.fill(false);
  wakeup_.reset();
}

void EventUarchEngine::dispatch(const DynInst &inst) {
  if (!can_accept()) unreachable("dispatch into a full uarch window");

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
  // Completion callbacks at the same timestamp each call this function; the
  // first call starts the cycle's arbitration and the rest join it.
  if (issue_cycle_ != now) {
    issue_cycle_ = now;
    unit_used_.fill(false);
  }

  bool issued = false;
  const auto memory_head = next_memory_id();

  for (const auto &[id, entry] : scoreboard_.entries()) {
    if (!can_issue(entry, memory_head)) continue;

    const auto index = unit_index(entry.inst.unit);
    if (unit_used_[index]) continue;

    unit_used_[index] = true;
    issue_one(id, now);
    issued = true;
  }

  // Instructions left over because their unit already issued this cycle get
  // another look next cycle; completion events alone would serialize them.
  if (issued && has_ready_work()) schedule_wakeup(now + SINGLE_CYCLE);
  return issued;
}

void EventUarchEngine::schedule_wakeup(std::uint64_t when) {
  if (wakeup_ == when) return;
  wakeup_ = when;
  state_.events.schedule(when, [this, when] {
    if (wakeup_ == when) wakeup_.reset();
    issue_ready(when);
  });
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
}

void EventUarchEngine::fence(FenceKind kind) {
  while (has_work(kind)) {
    advance();
  }
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
  while (!scoreboard_.empty()) {
    advance();
  }
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

std::optional<std::uint64_t> EventUarchEngine::next_memory_id() const {
  for (const auto &[id, entry] : scoreboard_.entries()) {
    if (entry.inst.unit == Unit::dma && !entry.issued) return id;
  }
  return std::nullopt;
}

bool EventUarchEngine::can_issue(const Scoreboard::Entry &entry,
                                 std::optional<std::uint64_t> memory_head) const {
  if (entry.issued || !entry.deps.empty()) return false;
  // The memory queue issues in program order: only the oldest unissued memory
  // operation may issue, and younger ones may not pass it while it waits.
  return entry.inst.unit != Unit::dma || memory_head == entry.inst.id;
}

bool EventUarchEngine::has_ready_work() const {
  const auto memory_head = next_memory_id();
  for (const auto &pair : scoreboard_.entries()) {
    if (can_issue(pair.second, memory_head)) return true;
  }
  return false;
}

const UarchStats &EventUarchEngine::stats() const { return state_.stats; }

} // namespace tile_npu
