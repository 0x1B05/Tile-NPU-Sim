#pragma once

#include "tile_npu/dyn_inst.h"
#include "uarch/sram_bank.h"
#include "uarch/uarch_stats.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tile_npu {

inline constexpr std::size_t SRAM_DOMAIN_COUNT = static_cast<std::size_t>(Domain::vec32) + 1;

struct SramDomainConfig {
  std::size_t banks = 2;
  std::size_t ports_per_bank = 1;
  std::size_t bank_width_bytes = 32;
  std::size_t registers = 0;
  std::size_t bytes_per_register = 0;
};

// Defaults preserve the original two-bank Tile/Acc organisation: tile
// registers 0..15 and 16..31 sit on different banks, and the two banks can
// service one MMA operand stream each.
struct SramConfig {
  std::array<SramDomainConfig, SRAM_DOMAIN_COUNT> domains = [] {
    std::array<SramDomainConfig, SRAM_DOMAIN_COUNT> result{};
    result[static_cast<std::size_t>(Domain::tile)] =
        SramDomainConfig{.banks = 2,
                         .ports_per_bank = 1,
                         .bank_width_bytes = 32,
                         .registers = TILE_REGISTERS,
                         .bytes_per_register = MATRIX_ELEMENTS};
    result[static_cast<std::size_t>(Domain::acc)] =
        SramDomainConfig{.banks = 2,
                         .ports_per_bank = 1,
                         .bank_width_bytes = 32,
                         .registers = ACC_REGISTERS,
                         .bytes_per_register = MATRIX_ELEMENTS * sizeof(std::uint32_t)};
    result[static_cast<std::size_t>(Domain::vec32)] =
        SramDomainConfig{.banks = 2,
                         .ports_per_bank = 1,
                         .bank_width_bytes = 32,
                         .registers = VECTOR_REGISTERS,
                         .bytes_per_register = VECTOR_LANES * sizeof(std::uint32_t)};
    return result;
  }();
};

// Shared on-chip SRAM model for the Tile, Acc, and Vec32 register files. All
// of DMA, MMA, and Vector issue through this object, so bank and port
// contention is visible across units.
class SramModel {
public:
  explicit SramModel(SramConfig config = {}, SramStats *stats = nullptr);

  void reset();

  // Reserves `cycles` on the bank that owns the register. This is the shape
  // used by the MMA and Vector pipelines, which move a stream rather than a
  // byte range.
  SramReservation reserve_stream(Domain domain, std::uint8_t index, std::uint64_t now,
                                 std::uint64_t earliest, std::uint64_t cycles, bool write);

  // Byte-accurate access used by DMA. The transfer is striped across banks
  // starting at the bank selected by the register and offset.
  SramReservation access(Domain domain, std::uint8_t index, std::uint32_t offset_bytes,
                         std::uint32_t bytes, std::uint64_t now, std::uint64_t earliest,
                         bool write);

  const SramStats &stats() const { return *stats_; }

  const SramConfig &config() const { return config_; }

  std::size_t retained_intervals() const;

private:
  const SramDomainConfig &domain_config(Domain domain) const;
  void touch_stats(bool write, std::uint32_t bytes, std::uint64_t busy_cycles, bool conflict);

  SramConfig config_{};
  std::array<std::vector<SramBank>, SRAM_DOMAIN_COUNT> banks_{};
  SramStats local_stats_{};
  SramStats *stats_ = nullptr;
};

} // namespace tile_npu
