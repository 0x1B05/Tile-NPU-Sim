#include "uarch/sram_model.h"

#include "utils/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tile_npu {
namespace {

std::size_t domain_index(Domain domain) { return static_cast<std::size_t>(domain); }

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

} // namespace

SramModel::SramModel(SramConfig config, SramStats *stats)
    : config_(config), stats_(stats != nullptr ? stats : &local_stats_) {
  for (std::size_t domain = 1; domain < SRAM_DOMAIN_COUNT; ++domain) {
    const auto &domain_config_value = config_.domains[domain];
    auto &domain_banks = banks_[domain];
    domain_banks.reserve(domain_config_value.banks);
    for (std::size_t bank = 0; bank < domain_config_value.banks; ++bank) {
      domain_banks.emplace_back(domain_config_value.ports_per_bank);
    }
  }
}

void SramModel::reset() {
  for (auto &domain_banks : banks_) {
    for (auto &bank : domain_banks)
      bank.reset();
  }
  *stats_ = SramStats{};
}

const SramDomainConfig &SramModel::domain_config(Domain domain) const {
  const auto index = domain_index(domain);
  if (index >= SRAM_DOMAIN_COUNT) unreachable("SRAM access to an invalid domain");
  return config_.domains[index];
}

void SramModel::touch_stats(Domain domain, bool write, std::uint32_t bytes,
                            std::uint64_t busy_cycles, bool conflict) {
  auto update = [&](SramDomainStats &target) {
    ++target.accesses;
    if (write) {
      ++target.writes;
      target.write_bytes += bytes;
    } else {
      ++target.reads;
      target.read_bytes += bytes;
    }
    target.busy_cycles += busy_cycles;
    if (conflict) ++target.conflicts;
  };

  ++stats_->accesses;
  if (write) {
    ++stats_->writes;
    stats_->write_bytes += bytes;
  } else {
    ++stats_->reads;
    stats_->read_bytes += bytes;
  }
  stats_->busy_cycles += busy_cycles;
  if (conflict) ++stats_->conflicts;
  update(stats_->domains[domain_index(domain)]);
}

SramReservation SramModel::reserve_stream(Domain domain, std::uint8_t index, std::uint64_t now,
                                          std::uint64_t earliest, std::uint64_t cycles,
                                          bool write) {
  if (domain == Domain::scalar) {
    return SramReservation{.start = std::max(now, earliest),
                           .end = std::max(now, earliest) + cycles,
                           .conflict = false};
  }

  const auto &config = domain_config(domain);
  auto &domain_banks = banks_[domain_index(domain)];
  if (domain_banks.empty() || index >= config.registers) {
    unreachable("SRAM stream access outside the register file");
  }
  if (cycles == 0) {
    const auto point = std::max(now, earliest);
    return SramReservation{.start = point, .end = point, .conflict = false};
  }

  const auto bank_index =
      std::min(domain_banks.size() - 1, static_cast<std::size_t>(index) * domain_banks.size() /
                                            std::max<std::size_t>(1, config.registers));
  const auto result = domain_banks[bank_index].reserve(now, earliest, cycles);
  touch_stats(domain, write, 0, cycles, result.conflict);
  return result;
}

SramReservation SramModel::access(Domain domain, std::uint8_t index, std::uint32_t offset_bytes,
                                  std::uint32_t bytes, std::uint64_t now, std::uint64_t earliest,
                                  bool write) {
  if (domain == Domain::scalar || bytes == 0) {
    const auto point = std::max(now, earliest);
    return SramReservation{.start = point, .end = point, .conflict = false};
  }

  const auto &config = domain_config(domain);
  auto &domain_banks = banks_[domain_index(domain)];
  if (domain_banks.empty() || index >= config.registers) {
    unreachable("SRAM byte access outside the register file");
  }

  const auto width = std::max<std::size_t>(1, config.bank_width_bytes);
  const auto beats = ceil_div(bytes, width);
  const auto base = static_cast<std::uint64_t>(index) * config.bytes_per_register +
                    static_cast<std::uint64_t>(offset_bytes);
  const auto first_bank = (base / width) % domain_banks.size();

  std::vector<std::uint64_t> per_bank(domain_banks.size(), 0);
  for (std::uint64_t beat = 0; beat < beats; ++beat) {
    ++per_bank[(first_bank + beat) % domain_banks.size()];
  }

  auto start = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t end = 0;
  bool conflict = false;
  for (std::size_t bank = 0; bank < domain_banks.size(); ++bank) {
    if (per_bank[bank] == 0) continue;
    const auto result = domain_banks[bank].reserve(now, earliest, per_bank[bank]);
    start = std::min(start, result.start);
    end = std::max(end, result.end);
    conflict = conflict || result.conflict;
  }

  if (start == std::numeric_limits<std::uint64_t>::max()) start = std::max(now, earliest);
  touch_stats(domain, write, bytes, beats, conflict);
  return SramReservation{.start = start, .end = end, .conflict = conflict};
}

std::size_t SramModel::retained_intervals() const {
  std::size_t count = 0;
  for (const auto &domain_banks : banks_) {
    for (const auto &bank : domain_banks)
      count += bank.retained_intervals();
  }
  return count;
}

} // namespace tile_npu
