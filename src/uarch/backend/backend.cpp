#include "uarch/backend/backend.h"

#include "utils/common.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {
namespace {

constexpr std::size_t INFLIGHT_CAPACITY = 32;

} // namespace

UarchBackend::UarchBackend(UarchClock &clock, UarchStats &stats, UarchExecutor &executor)
    : clock_(clock), stats_(stats), executor_(executor) {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    queues_[unit] = IssueQueue(queue_depth(static_cast<Unit>(unit)));
  }
}

void UarchBackend::reset() {
  events_.clear();
  scoreboard_.reset();
  for (auto &queue : queues_)
    queue.clear();
  inst_issue_not_before_.clear();
  unit_issue_not_before_.fill(0);
  failure_.reset();
}

std::optional<StepResult> UarchBackend::take_failure() {
  if (!failure_) return std::nullopt;

  auto failure = failure_;
  reset();
  return failure;
}

bool UarchBackend::outstanding(Unit unit) const {
  for (const auto &pair : scoreboard_.entries()) {
    if (pair.second.inst.unit == unit) return true;
  }
  return false;
}

bool UarchBackend::dispatch(const DynInst &inst) {
  if (failure_) return false;

  const auto unit = static_cast<std::size_t>(inst.unit);

  if (scoreboard_.size() >= INFLIGHT_CAPACITY || !queues_[unit].push(inst.id)) return false;

  scoreboard_.dispatch(inst);
  inst_issue_not_before_[inst.id] = clock_.now() + 1;
  return true;
}

void UarchBackend::cycle() {
  events_.run_due(clock_.now());
  issue();
  stats_.cycles = clock_.now();
}

std::uint64_t UarchBackend::advance() {
  const auto start = clock_.now();

  while (true) {
    const auto next_event_time = events_.next_time();
    // True when this cycle starts with a completion to process.
    const bool at_event = next_event_time && *next_event_time <= clock_.now();

    // Fire every event scheduled exactly at the current clock time.
    events_.run_due(clock_.now());
    issue();

    // Stop after processing this completion boundary. Remaining ready work
    // issues on the next advance() call.
    if (at_event) break;

    // A unit that issued one instruction may have another ready entry for the
    // next cycle. Do not jump over those issue opportunities.
    if (any_issuable()) {
      clock_.tick();
      continue;
    }

    // issue() may have scheduled a completion earlier than the snapshot taken
    // at the top of this iteration.
    const auto next_completion_time = events_.next_time();

    if (!next_completion_time) {
      if (!scoreboard_.empty()) {
        unreachable("uarch backend cannot make progress");
      }
      break;
    }

    // Jump to the completion boundary; the next iteration processes it.
    clock_.advance_to(*next_completion_time);
  }

  stats_.cycles = clock_.now();
  return clock_.now() - start;
}

void UarchBackend::drain() {
  while (!scoreboard_.empty()) {
    events_.run_due(clock_.now());
    issue();
    if (scoreboard_.empty()) break;

    if (any_issuable()) {
      clock_.tick();
      continue;
    }

    if (const auto next_event_time = events_.next_time()) {
      clock_.advance_to(*next_event_time);
      continue;
    }

    unreachable("uarch backend cannot drain dispatched work");
  }
  stats_.cycles = clock_.now();
}

void UarchBackend::issue() {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    if (unit_issue_not_before_[unit] > clock_.now()) continue;

    const auto candidate = next_issuable(unit);
    if (!candidate) continue;

    const auto id = *candidate;
    const auto after = inst_issue_not_before_.find(id);
    if (after != inst_issue_not_before_.end() && after->second > clock_.now()) continue;

    queues_[unit].remove(id);
    inst_issue_not_before_.erase(id);
    scoreboard_.mark_issued(id);

    const auto result = executor_.execute(scoreboard_.inst(id));
    if (result.status != StepStatus::retired) {
      reset();
      failure_ = result;
      return;
    }

    events_.schedule(clock_.now() + latency(static_cast<Unit>(unit)), [this, id] { complete(id); });
    unit_issue_not_before_[unit] = clock_.now() + 1;
  }
}

std::optional<std::uint64_t> UarchBackend::next_issuable(std::size_t unit) const {
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

bool UarchBackend::any_issuable() const {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    if (next_issuable(unit)) return true;
  }
  return false;
}

void UarchBackend::complete(std::uint64_t id) {
  const auto &entries = scoreboard_.entries();
  const auto it = entries.find(id);
  if (it == entries.end()) return;

  const auto inst = it->second.inst;
  if (!scoreboard_.complete(id)) return;
  inst_issue_not_before_.erase(id);
  executor_.complete(inst);
  ++stats_.retired;
  ++stats_.unit_count[static_cast<std::size_t>(inst.unit)];
}

std::size_t UarchBackend::queue_depth(Unit unit) {
  switch (unit) {
  case Unit::mma: return 2;
  case Unit::dma: return 16;
  default: return 8;
  }
}

std::uint64_t UarchBackend::latency(Unit unit) {
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
