#include "config.h"
#include "tile_npu/platform/ram.h"
#include "tile_npu/simulator.h"

#include <cstddef>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>

namespace {
constexpr std::uint64_t ram_base = 0x80000000;
constexpr std::size_t ram_size = 128u << 20;

struct Reference {
  tile_npu::Ram memory{ram_base, ram_size};
  tile_npu::Simulator sim{memory};
};

std::optional<Reference> reference;

Reference &instance() {
  if (!reference) {
    std::fputs("Tile-NPU DiffTest reference was used before initialization\n", stderr);
    std::abort();
  }
  return *reference;
}
} // namespace

extern "C" void difftest_init(int /*port*/) {
  reference.emplace();
  reference->sim.reset(ram_base);
}

extern "C" void difftest_memcpy(std::uint64_t addr, void *buf, std::size_t size, bool to_ref) {
  auto &memory = instance().memory;
  auto bytes = std::span(static_cast<std::byte *>(buf), size);
  const bool ok = to_ref ? memory.write(addr, bytes) : memory.read(addr, bytes);
  if (!ok) {
    std::fprintf(stderr, "Tile-NPU DiffTest memory copy out of range at 0x%llx (%zu bytes)\n",
                 static_cast<unsigned long long>(addr), size);
    std::abort();
  }
}

// gpr points to 32 uint64_t values; pc points to one uint64_t.
extern "C" void difftest_regcpy(std::uint64_t *gpr, std::uint64_t *pc, bool to_ref) {
  auto &state = instance().sim.state().scalar;
  if (to_ref) {
    for (std::size_t i = 0; i < 32; ++i) {
      state.write_x(i, gpr[i]);
    }
    state.pc = *pc;
  } else {
    for (std::size_t i = 0; i < 32; ++i) {
      gpr[i] = state.read_x(i);
    }
    *pc = state.pc;
  }
}

// One complete CFG_REG type per call; indices and 32-bit limbs increase in
// address order (least-significant limb first). Restore also accepts the
// zero extents present after reset or a partially initialized config.
extern "C" void difftest_configcpy(unsigned type, std::uint32_t *words, bool to_ref) {
  using namespace tile_npu;
  constexpr ConfigRegType types[] = {ConfigRegType::tc, ConfigRegType::tm, ConfigRegType::ac,
                                     ConfigRegType::am, ConfigRegType::vc, ConfigRegType::vm};
  if (type >= std::size(types)) std::abort();
  auto &state = instance().sim.state();
  const auto reg_type = types[type];
  dispatch_configs(state, reg_type, [&](auto &regs) {
    for (auto &r : regs) {
      using T = std::remove_cvref_t<decltype(r)>;
      if constexpr (DomainConfig<T>) {
        if (!to_ref) {
          *words = read_compute_config(state, reg_type, &r - regs.data());
        } else if constexpr (MatrixConfig<T>) {
          r.rows = *words;
          r.cols = *words >> 8;
          r.dtype = static_cast<DType>((*words >> 16) & 15);
          r.layout = static_cast<Layout>((*words >> 20) & 7);
        } else {
          r.len = *words;
          r.dtype = static_cast<DType>((*words >> 8) & 15);
        }
        ++words;
      } else {
        constexpr bool matrix = std::is_same_v<T, MatrixDescriptor>;
        if (to_ref) {
          r.base_addr = words[0] | (std::uint64_t{words[1] & 0xffff} << 32);
          r.flags = words[1] >> (matrix ? 20 : 16);
          if constexpr (matrix) {
            r.transform = (words[1] >> 16) & 15;
            r.row_stride_bytes = std::bit_cast<std::int32_t>(words[2]);
            r.col_stride_bytes = std::bit_cast<std::int32_t>(words[3]);
          } else {
            r.stride_bytes = std::bit_cast<std::int32_t>(words[2]);
          }
        } else {
          words[0] = r.base_addr;
          words[1] = (r.base_addr >> 32) | (std::uint32_t{r.flags} << (matrix ? 20 : 16));
          if constexpr (matrix) {
            words[1] |= std::uint32_t{r.transform} << 16;
            words[2] = r.row_stride_bytes;
            words[3] = r.col_stride_bytes;
          } else {
            words[2] = r.stride_bytes;
          }
        }
        words += matrix ? 4 : 3;
      }
    }
  });
}

extern "C" void difftest_exec(std::uint64_t count) {
  auto &sim = instance().sim;
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto result = sim.step();
    if (result.status != tile_npu::StepStatus::retired) {
      const auto name = tile_npu::status_name(result.status);
      std::fprintf(stderr, "Tile-NPU DiffTest reference stopped at 0x%llx (%.*s)\n",
                   static_cast<unsigned long long>(result.pc), static_cast<int>(name.size()),
                   name.data());
      std::abort();
    }
  }
}
