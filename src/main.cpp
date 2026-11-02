#include "monitor/monitor.h"

#include <CLI/CLI.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char **argv) {
  tile_npu::monitor::Options options;
  CLI::App app{"Tile-NPU simulator"};
  std::vector<std::string> image_args;
  std::int64_t steps = -1;
  std::string dump_range;
#if defined(CONFIG_ITRACE) || defined(CONFIG_MTRACE)
  std::string trace_selection;
  std::string trace_dir;
#endif
#ifdef CONFIG_MTRACE
  std::string mtrace_flush;
#endif
  // clang-format off
#ifdef CONFIG_SDB
  app.add_flag("-i,--interactive", options.interactive, "Open the sdb shell");
#endif
  app.add_option("-m,--memory"   , options.memory_mib , "RAM size in MiB")->check(CLI::PositiveNumber);
  app.add_option("-s,--steps"    , steps              , "Retired-instruction limit (-1: unlimited)");
#ifdef CONFIG_ITRACE
  app.add_flag("--itrace"        , options.itrace     , "Trace executed instructions to a log file");
  app.add_option("--itrace-log"  , options.itrace_log , "Trace output path (implies --itrace)");
#endif
#ifdef CONFIG_MTRACE
  app.add_flag("--mtrace"        , options.mtrace     , "Trace NPU data movement to a log file");
  app.add_option("--mtrace-log"  , options.mtrace_log , "Data-path trace output path (implies --mtrace)");
  app.add_option("--mtrace-flush", mtrace_flush       , "Flush mode: batch (default) or line");
#endif
#if defined(CONFIG_ITRACE) || defined(CONFIG_MTRACE)
  app.add_option("--trace"       , trace_selection    , "Select traces: itrace, mtrace, or all")
      ->check(CLI::IsMember({"itrace", "mtrace", "all"}));
  app.add_option("--trace-dir"   , trace_dir          , "Directory for default itrace/mtrace logs");
#endif
  app.add_option("--dump-range"  , dump_range         , "Dump FILE@0xADDR:0xSIZE after the run");
  app.add_option("images"        , image_args         , "Program and data images (path[@0xADDR])") ->required();
  // clang-format on
  CLI11_PARSE(app, argc, argv);

#ifdef CONFIG_ITRACE
  if (!options.itrace_log.empty()) {
    options.itrace = true;
  }
#endif
#ifdef CONFIG_MTRACE
  if (!options.mtrace_log.empty()) {
    options.mtrace = true;
  }
  if (!mtrace_flush.empty()) {
    if (mtrace_flush != "batch" && mtrace_flush != "line") {
      std::cerr << "invalid mtrace flush mode: " << mtrace_flush << '\n';
      return 2;
    }
    options.mtrace = true;
    options.mtrace_flush_line = mtrace_flush == "line";
  }
#endif
#if defined(CONFIG_ITRACE) || defined(CONFIG_MTRACE)
  if (!trace_selection.empty()) {
    const bool want_itrace = trace_selection == "itrace" || trace_selection == "all";
    const bool want_mtrace = trace_selection == "mtrace" || trace_selection == "all";
#ifndef CONFIG_ITRACE
    if (want_itrace) {
      std::cerr << "itrace was not compiled; rebuild with --trace=itrace or --trace=all\n";
      return 2;
    }
#endif
#ifndef CONFIG_MTRACE
    if (want_mtrace) {
      std::cerr << "mtrace was not compiled; rebuild with --trace=mtrace or --trace=all\n";
      return 2;
    }
#endif
    options.itrace = options.itrace || want_itrace;
#ifdef CONFIG_MTRACE
    options.mtrace = options.mtrace || want_mtrace;
#endif
  }
  options.trace_dir = trace_dir;
#endif
  options.step_limit = static_cast<std::uint64_t>(steps);
  if (!dump_range.empty()) {
    tile_npu::monitor::DumpRange range{};
    if (!tile_npu::monitor::parse_dump_range(dump_range, range)) {
      std::cerr << "invalid dump range: " << dump_range << '\n';
      return 2;
    }
    options.dump_range = std::move(range);
  }
  for (const auto &arg : image_args) {
    tile_npu::monitor::Image image{};
    if (!tile_npu::monitor::parse_image(arg, image)) {
      std::cerr << "invalid image: " << arg << '\n';
      return 2;
    }
    options.images.push_back(std::move(image));
  }
  return tile_npu::monitor::run_images(options);
}
