#include "frontend/common/image.h"
#include "frontend/common/parse.h"
#include "frontend/func/run.h"

#include <CLI/CLI.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char **argv) {
  tile_npu::frontend::func::Options options;
  CLI::App app{"Tile-NPU simulator"};
  std::vector<std::string> image_args;
  std::int64_t steps = -1;
  std::string dump_range;
#ifdef CONFIG_TRACE
  std::string trace_selection;
  std::string trace_dir;
  std::string trace_pc_start_arg;
  std::string trace_pc_end_arg;
  std::string trace_flush;
  std::vector<std::string> trace_choices;
#endif
  // clang-format off
#ifdef CONFIG_SDB
  app.add_flag("-i,--interactive", options.interactive, "Open the sdb shell");
#endif
  app.add_option("-m,--memory"   , options.memory_mib , "RAM size in MiB")->check(CLI::PositiveNumber);
  app.add_option("-s,--steps"    , steps              , "Retired-instruction limit (-1: unlimited)");
#ifdef CONFIG_ITRACE
  app.add_flag("--itrace"        , options.itrace     , "Trace executed instructions to a log file");
#endif
#ifdef CONFIG_MTRACE
  app.add_flag("--mtrace"        , options.mtrace     , "Trace NPU data movement to a log file");
#endif
#ifdef CONFIG_TRACE
#ifdef CONFIG_ITRACE
  trace_choices.emplace_back("itrace");
#endif
#ifdef CONFIG_MTRACE
  trace_choices.emplace_back("mtrace");
#endif
  if (trace_choices.size() > 1) trace_choices.emplace_back("all");
  std::string trace_help = trace_choices.size() == 1 ? "Select trace: " : "Select traces: ";
  for (std::size_t i = 0; i < trace_choices.size(); ++i) {
    if (i != 0) trace_help += (i + 1 == trace_choices.size() ? ", or " : ", ");
    trace_help += trace_choices[i];
  }
  app.add_option("--trace"       , trace_selection    , trace_help)
      ->check(CLI::IsMember(trace_choices));
  app.add_option("--trace-dir"   , trace_dir          , "Directory for default itrace/mtrace logs");
  app.add_option("--trace-start" , trace_pc_start_arg , "First PC to trace (hex or decimal)");
  app.add_option("--trace-end"   , trace_pc_end_arg   , "One-past-last PC to trace (0 = unlimited)");
  app.add_option("--trace-flush" , trace_flush        , "Flush mode: batch (default) or line");
#endif
  app.add_option("--dump-range"  , dump_range         , "Dump FILE@0xADDR:0xSIZE after the run");
  app.add_option("images"        , image_args         , "Program and data images (path[@0xADDR])") ->required();
  // clang-format on
  CLI11_PARSE(app, argc, argv);

#ifdef CONFIG_TRACE
  if (!trace_flush.empty()) {
    if (trace_flush != "batch" && trace_flush != "line") {
      std::cerr << "invalid trace flush mode: " << trace_flush << '\n';
      return 2;
    }
    options.trace_flush_line = trace_flush == "line";
  }
  if (!trace_selection.empty()) {
    const bool all = trace_selection == "all";
#ifdef CONFIG_ITRACE
    options.itrace = options.itrace || all || trace_selection == "itrace";
#endif
#ifdef CONFIG_MTRACE
    options.mtrace = options.mtrace || all || trace_selection == "mtrace";
#endif
  }
  options.trace_dir = trace_dir;
  const auto parse_pc = [](const std::string &text, std::uint64_t &value) {
    return !text.empty() && text.front() != '-' && tile_npu::frontend::parse_u64(text, value);
  };
  if (!trace_pc_start_arg.empty() && !parse_pc(trace_pc_start_arg, options.trace_pc_start)) {
    std::cerr << "invalid trace PC start: " << trace_pc_start_arg << '\n';
    return 2;
  }
  if (!trace_pc_end_arg.empty() && !parse_pc(trace_pc_end_arg, options.trace_pc_end)) {
    std::cerr << "invalid trace PC end: " << trace_pc_end_arg << '\n';
    return 2;
  }
  if (options.trace_pc_end != 0 && options.trace_pc_end <= options.trace_pc_start) {
    std::cerr << "trace PC end must be zero or greater than trace PC start\n";
    return 2;
  }
#endif
  options.step_limit = static_cast<std::uint64_t>(steps);
  if (!dump_range.empty()) {
    tile_npu::frontend::func::DumpRange range{};
    if (!tile_npu::frontend::func::parse_dump_range(dump_range, range)) {
      std::cerr << "invalid dump range: " << dump_range << '\n';
      return 2;
    }
    options.dump_range = std::move(range);
  }
  for (const auto &arg : image_args) {
    tile_npu::frontend::Image image{};
    if (!tile_npu::frontend::parse_image(arg, image)) {
      std::cerr << "invalid image: " << arg << '\n';
      return 2;
    }
    options.images.push_back(std::move(image));
  }
  return tile_npu::frontend::func::run_images(options);
}
