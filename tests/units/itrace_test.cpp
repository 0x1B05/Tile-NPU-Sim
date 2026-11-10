#ifdef CONFIG_ITRACE

#include "func/core.h"

#include <cstdint>
#include <vector>

#include "units/execute_helpers.h"

using namespace tile_npu::test;

TEST_CASE(itrace_pc_window_filters_callback) {
  tile_npu::Ram memory(CODE_BASE, 16);
  for (std::uint64_t offset = 0; offset < 12; offset += 4) {
    write_word(memory, CODE_BASE + offset, 0x00000013u); // nop
  }

  tile_npu::FuncCore sim(memory);
  std::vector<std::uint64_t> pcs;
  sim.set_itrace_callback([&pcs](std::uint64_t pc, std::uint32_t, const tile_npu::DecodeResult &,
                                 tile_npu::StepStatus) { pcs.push_back(pc); });
  sim.reset(CODE_BASE);
  sim.set_itrace_pc_window(CODE_BASE + 4, CODE_BASE + 8);

  for (int i = 0; i < 3; ++i) {
    check_eq(sim.step().status, tile_npu::StepStatus::retired, "nop retires");
  }

  check_eq(pcs.size(), std::size_t{1}, "only the in-window instruction is traced");
  check_eq(hex(pcs[0]), hex(CODE_BASE + 4), "traced PC is inside the window");
}

#endif // CONFIG_ITRACE
