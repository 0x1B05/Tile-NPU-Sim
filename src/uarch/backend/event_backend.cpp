#include "uarch/backend/event_backend.h"

#include "utils/common.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {
namespace {

constexpr std::size_t WINDOW_DEPTH = 32;

} // namespace

EventUarchBackend::EventUarchBackend(UarchClock &clock) : clock_(clock) {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    queues_[unit] = IssueQueue(queue_depth(static_cast<Unit>(unit)));
  }
}

void EventUarchBackend::reset() {
  // UarchStats and UarchClock are owned by the machine and reset by
  // UarchCore::reset().
  events_.clear();
  scoreboard_.reset();
  for (auto &queue : queues_)
    queue.clear();
  issue_after_.clear();
  next_issue_cycle_.fill(0);
}

bool EventUarchBackend::dispatch(const DynInst &inst) {
  const auto unit = static_cast<std::size_t>(inst.unit);

  if (scoreboard_.size() >= WINDOW_DEPTH || !queues_[unit].push(inst.id)) return false;

  scoreboard_.dispatch(inst);
  // A newly dispatched instruction becomes eligible on the next cycle without
  // delaying older entries in the same unit.
  issue_after_[inst.id] = clock_.now() + 1;
  return true;
}

void EventUarchBackend::cycle() {
  events_.run_until(clock_.now());
  issue();
  counters().cycles = clock_.now();
}

std::uint64_t EventUarchBackend::advance() {
  const auto before = clock_.now();

  while (true) {
    const auto next = events_.next_time();
    const bool at_event = next && *next <= clock_.now();

    events_.run_until(clock_.now());
    issue();

    // Stop after processing this completion boundary. Remaining ready work
    // issues on the next advance() call.
    if (at_event) break;

    // A unit that issued one instruction may have another ready entry for the
    // next cycle. Do not jump over those issue opportunities.
    if (any_ready_inst()) {
      clock_.tick();
      continue;
    }

    // issue() may have scheduled a completion earlier than the snapshot taken
    // at the top of this iteration.
    const auto next_after_issue = events_.next_time();

    if (!next_after_issue) {
      if (!scoreboard_.empty()) {
        unreachable("uarch backend cannot make progress");
      }
      break;
    }

    // Jump to the completion boundary; the next iteration processes it.
    clock_.advance_to(*next_after_issue);
  }

  counters().cycles = clock_.now();
  return clock_.now() - before;
}

void EventUarchBackend::drain() {
  while (!scoreboard_.empty()) {
    events_.run_until(clock_.now());
    issue();
    if (scoreboard_.empty()) break;

    if (any_ready_inst()) {
      clock_.tick();
      continue;
    }

    if (const auto next = events_.next_time()) {
      events_.run_until(*next);
      continue;
    }

    unreachable("uarch backend cannot drain dispatched work");
  }
  counters().cycles = clock_.now();
}

void EventUarchBackend::issue() {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    if (next_issue_cycle_[unit] > clock_.now()) continue;

    const auto candidate = next_ready(unit);
    if (!candidate) continue;

    const auto id = *candidate;
    const auto after = issue_after_.find(id);
    if (after != issue_after_.end() && after->second > clock_.now()) continue;

    queues_[unit].remove(id);
    issue_after_.erase(id);
    scoreboard_.mark_issued(id);
    if (executor_ != nullptr) executor_->execute(scoreboard_.inst(id));
    events_.schedule(clock_.now() + latency(static_cast<Unit>(unit)), [this, id] { complete(id); });
    next_issue_cycle_[unit] = clock_.now() + 1;
  }
}

bool EventUarchBackend::any_ready_inst() const {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    if (next_ready(unit)) return true;
  }
  return false;
}

std::optional<std::uint64_t> EventUarchBackend::next_ready(std::size_t unit) const {
  const auto &queue = queues_[unit];

  if (static_cast<Unit>(unit) == Unit::dma) {
    // All memory operations share one program-order queue.
    auto candidate = queue.front();
    if (candidate && !scoreboard_.ready(*candidate)) candidate.reset();
    return candidate;
  }

  // General queues pick the oldest ready entry.
  for (const auto id : queue.entries()) {
    if (scoreboard_.ready(id)) return id;
  }
  return std::nullopt;
}

void EventUarchBackend::complete(std::uint64_t id) {
  const auto &entries = scoreboard_.entries();
  const auto it = entries.find(id);
  if (it == entries.end()) return;

  const auto inst = it->second.inst;
  if (!scoreboard_.complete(id)) return;
  issue_after_.erase(id);
  if (executor_ != nullptr) executor_->complete(inst);
  auto &stats = counters();
  ++stats.retired;
  ++stats.unit_count[static_cast<std::size_t>(inst.unit)];
}

UarchStats &EventUarchBackend::counters() {
  if (stats_ == nullptr) unreachable("backend stats are not attached");
  return *stats_;
}

std::size_t EventUarchBackend::queue_depth(Unit unit) {
  switch (unit) {
  case Unit::mma: return 2;
  case Unit::dma: return 16;
  default: return 8;
  }
}

std::uint64_t EventUarchBackend::latency(Unit unit) {
  switch (unit) {
  case Unit::mma: return 10;
  case Unit::dma: return 5;
  case Unit::reduce: return 3;
  case Unit::quant: return 3;
  case Unit::vec32: return 2;
  default: return 1;
  }
}

} // namespace tile_npu
