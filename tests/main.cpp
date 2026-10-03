#include "check.h"

#include <cstddef>
#include <exception>
#include <iostream>

int main() {
  std::size_t failed = 0;
  for (const auto &[name, run] : tile_npu::test::registry()) {
    try {
      run();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception &error) {
      ++failed;
      std::cout << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  const std::size_t total = tile_npu::test::registry().size();
  std::cout << (total - failed) << '/' << total << " tests passed\n";
  return failed == 0 ? 0 : 1;
}
