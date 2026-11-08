#include "uarch/event_engine.h"

#include "utils/common.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace tile_npu {
namespace {

constexpr std::uint64_t SINGLE_CYCLE = 1;

constexpr bool fence_covers(FenceKind kind, Unit unit) {
  switch (kind) {
  case FenceKind::mem: return unit == Unit::dma;
  case FenceKind::sa: return unit == Unit::mma;
  case FenceKind::all: return true;
  }
  return false;
}

} // namespace

EventUarchEngine::EventUarchEngine(UarchConfig config)
    : config_(std::move(config)), sram_(config_.sram, &state_.stats.sram),
      mma_(&sram_, config_.mma),
      dma_(&sram_, std::make_unique<SimpleMemoryService>(config_.memory)),
      vector_(&sram_, config_.vector) {
  for (std::size_t i = 0; i < UNIT_COUNT; ++i) {
    queues_[i] = Queue(config_.queue_depth[i]);
  }
}

void EventUarchEngine::reset() {
  state_ = UarchState{};
  sram_.reset();
  scoreboard_.reset();
  mma_.reset();
  dma_.reset();
  vector_.reset();
  for (auto &queue : queues_) {
    queue.clear();
  }
  issue_cycle_.reset();
  unit_used_.fill(false);
  wakeup_.reset();
  sample_cycle_ = 0;
  sample_ = {};
}

void EventUarchEngine::dispatch(const DynInst &inst) {
  if (!can_dispatch(inst)) unreachable("dispatch into a full uarch window or unit queue");

  // A fence is a completion boundary for the backends it names. Waiting here
  // also delays every younger instruction, because dispatch advances in
  // program order.
  if (inst.unit == Unit::sync) fence(inst.fence);

  ++state_.stats.unit_count[unit_index(inst.unit)];
  scoreboard_.dispatch(inst);
  if (!queues_[unit_index(inst.unit)].push(inst.id)) {
    unreachable("dispatch into a full unit queue");
  }
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

  for (std::size_t unit = 0; unit < UNIT_COUNT; ++unit) {
    if (unit_used_[unit]) continue;

    auto &queue = queues_[unit];
    std::optional<std::uint64_t> candidate;
    for (const auto id : queue.entries()) {
      if (can_issue(scoreboard_.entries().at(id), memory_head)) {
        candidate = id;
        break;
      }
    }
    if (!candidate) continue;

    unit_used_[unit] = true;
    queues_[unit].remove(*candidate);
    issue_one(*candidate, now);
    issued = true;
  }

  // Instructions left over because their unit already issued this cycle get
  // another look next cycle; completion events alone would serialize them.
  if (issued && has_ready_work()) schedule_wakeup(now + SINGLE_CYCLE);
  sample_queues(now);
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
  case Unit::dma:
    finish = dma_.issue(inst, now);
    state_.stats.memory = dma_.memory_stats();
    break;
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

bool EventUarchEngine::can_accept() const { return scoreboard_.size() < config_.window_depth; }

bool EventUarchEngine::window_full() const { return !can_accept(); }

bool EventUarchEngine::queue_full(Unit unit) const { return queues_[unit_index(unit)].full(); }

bool EventUarchEngine::can_dispatch(const DynInst &inst) const {
  return can_accept() && !queue_full(inst.unit);
}

void EventUarchEngine::record_window_stall(std::uint64_t cycles) {
  state_.stats.window_stall_cycles += cycles;
}

void EventUarchEngine::record_dispatch_stall(Unit unit, std::uint64_t cycles) {
  state_.stats.dispatch_stall_cycles[unit_index(unit)] += cycles;
}

std::uint64_t EventUarchEngine::advance() {
  const auto before = state_.stats.cycles;
  if (const auto next = state_.events.next_time()) {
    state_.events.run_until(*next);
  } else if (!issue_ready(state_.events.now())) {
    unreachable("uarch window cannot make progress");
  }
  state_.stats.cycles = state_.events.now();
  return state_.stats.cycles - before;
}

void EventUarchEngine::finish() {
  while (!scoreboard_.empty()) {
    advance();
  }
}

bool EventUarchEngine::has_work(FenceKind kind) const {
  for (const auto &pair : scoreboard_.entries()) {
    if (fence_covers(kind, pair.second.inst.unit)) return true;
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

void EventUarchEngine::refresh_sample() {
  sample_ = {};
  const auto memory_head = next_memory_id();
  for (const auto &pair : scoreboard_.entries()) {
    const auto &entry = pair.second;
    if (entry.issued) {
      const auto unit = unit_index(entry.inst.unit);
      ++sample_.active[unit];
      state_.stats.unit_in_flight_peak[unit] = std::max(
          state_.stats.unit_in_flight_peak[unit], static_cast<std::uint64_t>(sample_.active[unit]));
    }
  }
  for (std::size_t unit = 0; unit < UNIT_COUNT; ++unit) {
    const auto &queue = queues_[unit];
    sample_.occupancy[unit] = queue.size();
    state_.stats.queue_peak[unit] = std::max(state_.stats.queue_peak[unit], queue.size());
    for (const auto id : queue.entries()) {
      const auto &entry = scoreboard_.entries().at(id);
      if (!entry.deps.empty()) {
        ++sample_.dependency_waits[unit];
      } else if (entry.inst.unit == Unit::dma && memory_head != id) {
        ++sample_.memory_waits;
      }
    }
  }
}

void EventUarchEngine::sample_queues(std::uint64_t now) {
  if (now > sample_cycle_) {
    const auto delta = now - sample_cycle_;
    for (std::size_t unit = 0; unit < UNIT_COUNT; ++unit) {
      state_.stats.queue_occupancy_cycles[unit] +=
          static_cast<std::uint64_t>(sample_.occupancy[unit]) * delta;
      if (sample_.active[unit] != 0) state_.stats.unit_busy_cycles[unit] += delta;
      state_.stats.dependency_stall_cycles[unit] += sample_.dependency_waits[unit] * delta;
    }
    state_.stats.memory_order_stall_cycles += sample_.memory_waits * delta;
    sample_cycle_ = now;
  }
  refresh_sample();
}

const UarchStats &EventUarchEngine::stats() const { return state_.stats; }

} // namespace tile_npu
