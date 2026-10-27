#include "framework.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string_view>

#include "guest/guest_image.h"

int main(int argc, char **argv) {
  // Guest tests load .bin images, so their runner alone accepts --guest-dir.
  std::string_view filter;
  bool list_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--list") {
      list_only = true;
    } else if (arg == "--filter" && i + 1 < argc) {
      filter = argv[++i];
    } else if (arg == "--guest-dir" && i + 1 < argc) {
      tile_npu::test::set_guest_image_dir(argv[++i]);
    } else {
      std::cerr << "usage: " << argv[0] << " [--list] [--filter SUBSTR] [--guest-dir DIR]\n";
      return 2;
    }
  }
  // A stable name order keeps --list output and failure reruns independent of
  // link order; duplicate names would make both ambiguous.
  auto &tests = tile_npu::test::registry();
  std::sort(tests.begin(), tests.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.name < rhs.name; });
  for (std::size_t i = 1; i < tests.size(); ++i) {
    if (tests[i - 1].name == tests[i].name) {
      std::cerr << "duplicate test name: " << tests[i - 1].name << '\n';
      return 2;
    }
  }
  if (list_only) {
    for (const auto &[name, run] : tests) {
      std::cout << name << '\n';
    }
    return 0;
  }
  std::size_t failed = 0;
  std::size_t selected = 0;
  for (const auto &[name, run] : tests) {
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
