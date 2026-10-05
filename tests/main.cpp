#include "check.h"

#include <cstddef>
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char **argv) {
  // Usage: sim_tests [--list] [--filter SUBSTR]
  std::string_view filter;
  bool list_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--list") {
      list_only = true;
    } else if (arg == "--filter" && i + 1 < argc) {
      filter = argv[++i];
    } else {
      std::cerr << "usage: " << argv[0] << " [--list] [--filter SUBSTR]\n";
      return 2;
    }
  }
  if (list_only) {
    for (const auto &[name, run] : tile_npu::test::registry()) {
      std::cout << name << '\n';
    }
    return 0;
  }
  std::size_t failed = 0;
  std::size_t selected = 0;
  for (const auto &[name, run] : tile_npu::test::registry()) {
    if (!filter.empty() && name.find(filter) == std::string_view::npos) {
      continue;
    }
    ++selected;
    try {
      run();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception &error) {
      ++failed;
      std::cout << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  if (selected == 0) {
    std::cerr << "no tests match the filter\n";
    return 2;
  }
  std::cout << (selected - failed) << '/' << selected << " tests passed\n";
  return failed == 0 ? 0 : 1;
}
