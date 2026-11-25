#include "tile_npu/platform/ram.h"
#include "tile_npu/simulator.h"

#include <cstddef>
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
