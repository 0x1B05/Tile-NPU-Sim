#include "frontend/common/constants.h"
#include "frontend/common/image.h"
#include "frontend/common/parse.h"
#include "frontend/common/platform.h"
#include "uarch/machine.h"
#include "uarch/state_digest.h"
#include "uarch/trace_replay.h"
#include "uarch/uarch_config_json.h"
#include "uarch/uarch_names.h"
#include "uarch/uarch_trace.h"
#include "utils/format.h"

#include <CLI/CLI.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using tile_npu::SRAM_DOMAIN_NAMES;
using tile_npu::UNIT_NAMES;

double per_cycle(std::uint64_t value, std::uint64_t cycles) {
  return cycles == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(cycles);
}

void write_u64_object(std::ostream &out,
                      const std::array<std::uint64_t, tile_npu::UNIT_COUNT> &values) {
  out << '{';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << '"' << UNIT_NAMES[i] << "\": " << values[i];
  }
  out << '}';
}

void write_ratio_object(std::ostream &out,
                        const std::array<std::uint64_t, tile_npu::UNIT_COUNT> &values,
                        std::uint64_t cycles) {
  out << '{';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << '"' << UNIT_NAMES[i] << "\": " << per_cycle(values[i], cycles);
  }
  out << '}';
}

template <typename Stats> void write_sram_fields(std::ostream &out, const Stats &stats) {
  out << "\"accesses\": " << stats.accesses << ", \"reads\": " << stats.reads
      << ", \"writes\": " << stats.writes << ", \"read_bytes\": " << stats.read_bytes
      << ", \"write_bytes\": " << stats.write_bytes << ", \"conflicts\": " << stats.conflicts
      << ", \"busy_cycles\": " << stats.busy_cycles;
}

void write_sram_domain(std::ostream &out, const tile_npu::SramDomainStats &stats) {
  out << '{';
  write_sram_fields(out, stats);
  out << '}';
}

void write_sram(std::ostream &out, const tile_npu::SramStats &stats, std::uint64_t cycles) {
  out << '{';
  write_sram_fields(out, stats);
  for (std::size_t domain = tile_npu::FIRST_SRAM_DOMAIN; domain < tile_npu::SRAM_DOMAIN_COUNT;
       ++domain) {
    out << ", \"" << SRAM_DOMAIN_NAMES[domain] << "\": ";
    write_sram_domain(out, stats.domains[domain]);
  }
  out << ", \"bandwidth_bytes_per_cycle\": "
      << per_cycle(stats.read_bytes + stats.write_bytes, cycles) << '}';
}

void write_memory(std::ostream &out, const tile_npu::MemoryStats &stats, std::uint64_t cycles) {
  out << "{\"transfers\": " << stats.transfers << ", \"reads\": " << stats.reads
      << ", \"writes\": " << stats.writes << ", \"read_bytes\": " << stats.read_bytes
      << ", \"write_bytes\": " << stats.write_bytes << ", \"busy_cycles\": " << stats.busy_cycles
      << ", \"row_hits\": " << stats.row_hits << ", \"row_misses\": " << stats.row_misses
      << ", \"bank_conflicts\": " << stats.bank_conflicts
      << ", \"outstanding_peak\": " << stats.outstanding_peak << ", \"bandwidth_bytes_per_cycle\": "
      << per_cycle(stats.read_bytes + stats.write_bytes, cycles) << '}';
}

void write_uarch_json_fields(std::ostream &out, const tile_npu::UarchConfig &config,
                             const tile_npu::UarchStats &stats) {
  out << "\"uarch_config\": " << tile_npu::uarch_config_json(config);
  out << ",\n  \"cycles\": " << stats.cycles;
  out << ",\n  \"retired\": " << stats.retired;
  out << ",\n  \"macs\": " << stats.macs;
  out << ",\n  \"bytes\": " << stats.bytes;
  out << ",\n  \"ipc\": " << per_cycle(stats.retired, stats.cycles);
  out << ",\n  \"macs_per_cycle\": " << per_cycle(stats.macs, stats.cycles);
  out << ",\n  \"arithmetic_intensity\": "
      << (stats.bytes == 0 ? 0.0 : static_cast<double>(stats.macs) / stats.bytes);
  out << ",\n  \"window_stall_cycles\": " << stats.window_stall_cycles;
  out << ",\n  \"memory_order_stall_cycles\": " << stats.memory_order_stall_cycles;
  out << ",\n  \"unit_count\": ";
  write_u64_object(out, stats.unit_count);
  out << ",\n  \"unit_busy_cycles\": ";
  write_u64_object(out, stats.unit_busy_cycles);
  out << ",\n  \"unit_utilization\": ";
  write_ratio_object(out, stats.unit_busy_cycles, stats.cycles);
  out << ",\n  \"unit_in_flight_peak\": ";
  write_u64_object(out, stats.unit_in_flight_peak);
  out << ",\n  \"dispatch_stall_cycles\": ";
  write_u64_object(out, stats.dispatch_stall_cycles);
  out << ",\n  \"dependency_stall_cycles\": ";
  write_u64_object(out, stats.dependency_stall_cycles);
  out << ",\n  \"queue_occupancy_cycles\": ";
  write_u64_object(out, stats.queue_occupancy_cycles);
  out << ",\n  \"queue_peak\": ";
  write_u64_object(out, stats.queue_peak);
  out << ",\n  \"sram\": ";
  write_sram(out, stats.sram, stats.cycles);
  out << ",\n  \"memory\": ";
  write_memory(out, stats.memory, stats.cycles);
}

bool write_run_json(const std::string &path, tile_npu::ExecutionMode mode,
                    const tile_npu::UarchConfig &config, const tile_npu::Machine &machine,
                    const tile_npu::StepResult &result, int exit_code) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "cannot open stats file: " << path << '\n';
    return false;
  }

  const auto &state = machine.state();
  out << "{\n";
  out << "  \"mode\": \"" << (mode == tile_npu::ExecutionMode::uarch ? "uarch" : "functional")
      << "\",\n";
  out << "  \"status\": \"" << tile_npu::status_name(result.status) << "\",\n";
  out << "  \"pc\": \"" << tile_npu::hex64(result.pc) << "\",\n";
  out << "  \"a0\": " << state.scalar.read_x(tile_npu::frontend::EXIT_CODE_REG) << ",\n";
  out << "  \"exit_code\": " << exit_code << ",\n";
  out << "  \"arch_digest\": \"" << tile_npu::hex64(tile_npu::architectural_digest(state)) << "\"";
  if (mode == tile_npu::ExecutionMode::uarch) {
    out << ",\n  ";
    write_uarch_json_fields(out, config, machine.stats());
  }
  out << "\n}\n";
  return out.good();
}

bool write_replay_json(const std::string &path, const tile_npu::UarchConfig &config,
                       const tile_npu::UarchStats &stats) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "cannot open stats file: " << path << '\n';
    return false;
  }
  out << "{\n  \"mode\": \"replay\",\n  \"status\": \"trace_end\",\n  ";
  write_uarch_json_fields(out, config, stats);
  out << "\n}\n";
  return out.good();
}

std::string csv_escape(std::string_view value) {
  if (value.find_first_of(",\"\n\r") == std::string_view::npos) return std::string(value);
  std::string escaped = "\"";
  for (const char ch : value) {
    if (ch == '"') escaped += '"';
    escaped += ch;
  }
  escaped += '"';
  return escaped;
}

std::string number(double value) {
  std::ostringstream text;
  text << value;
  return text.str();
}

bool write_timeline_json(const std::string &path,
                         const std::vector<tile_npu::TimelineRecord> &timeline) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "cannot open timeline file: " << path << '\n';
    return false;
  }
  out << "[\n";
  for (std::size_t i = 0; i < timeline.size(); ++i) {
    const auto &record = timeline[i];
    out << "  {\"id\": " << record.id << ", \"pc\": \"" << tile_npu::hex64(record.pc)
        << "\", \"op\": \"" << tile_npu::dyn_op_name(record.op) << "\", \"unit\": \""
        << tile_npu::UNIT_NAMES[tile_npu::unit_index(record.unit)]
        << "\", \"dispatch\": " << record.dispatch_cycle << ", \"ready\": " << record.ready_cycle
        << ", \"issue\": " << record.issue_cycle << ", \"complete\": " << record.complete_cycle
        << ", \"dependency_wait\": " << record.dependency_wait
        << ", \"memory_order_wait\": " << record.memory_order_wait
        << ", \"config_wait\": " << record.config_wait
        << ", \"arbitration_wait\": " << record.arbitration_wait << ", \"reason\": \""
        << tile_npu::stall_reason_name(record.reason) << "\"}";
    if (i + 1 != timeline.size()) out << ',';
    out << '\n';
  }
  out << "]\n";
  return out.good();
}

bool write_timeline_csv(const std::string &path,
                        const std::vector<tile_npu::TimelineRecord> &timeline) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "cannot open timeline file: " << path << '\n';
    return false;
  }
  out << "id,pc,op,unit,dispatch,ready,issue,complete,dependency_wait,memory_order_wait,"
         "config_wait,arbitration_wait,reason\n";
  for (const auto &record : timeline) {
    out << record.id << ',' << tile_npu::hex64(record.pc) << ',' << tile_npu::dyn_op_name(record.op)
        << ',' << tile_npu::UNIT_NAMES[tile_npu::unit_index(record.unit)] << ','
        << record.dispatch_cycle << ',' << record.ready_cycle << ',' << record.issue_cycle << ','
        << record.complete_cycle << ',' << record.dependency_wait << ',' << record.memory_order_wait
        << ',' << record.config_wait << ',' << record.arbitration_wait << ','
        << tile_npu::stall_reason_name(record.reason) << '\n';
  }
  return out.good();
}

bool write_run_csv(const std::string &path, tile_npu::ExecutionMode mode,
                   const tile_npu::UarchConfig &config, const tile_npu::Machine &machine,
                   const tile_npu::StepResult &result, int exit_code) {
  std::vector<std::pair<std::string, std::string>> fields;
  const auto add = [&fields](std::string key, std::string value) {
    fields.emplace_back(std::move(key), std::move(value));
  };

  const auto &state = machine.state();
  add("mode", mode == tile_npu::ExecutionMode::uarch ? "uarch" : "functional");
  add("status", std::string(tile_npu::status_name(result.status)));
  add("pc", tile_npu::hex64(result.pc));
  add("a0", std::to_string(state.scalar.read_x(tile_npu::frontend::EXIT_CODE_REG)));
  add("exit_code", std::to_string(exit_code));
  add("arch_digest", tile_npu::hex64(tile_npu::architectural_digest(state)));

  if (mode == tile_npu::ExecutionMode::uarch) {
    const auto &stats = machine.stats();
    add("cycles", std::to_string(stats.cycles));
    add("retired", std::to_string(stats.retired));
    add("macs", std::to_string(stats.macs));
    add("bytes", std::to_string(stats.bytes));
    add("ipc", number(per_cycle(stats.retired, stats.cycles)));
    add("macs_per_cycle", number(per_cycle(stats.macs, stats.cycles)));
    add("arithmetic_intensity",
        number(stats.bytes == 0 ? 0.0 : static_cast<double>(stats.macs) / stats.bytes));
    add("window_stall_cycles", std::to_string(stats.window_stall_cycles));
    add("memory_order_stall_cycles", std::to_string(stats.memory_order_stall_cycles));
    add("window_depth", std::to_string(config.window_depth));
    add("mma_read_latency", std::to_string(config.mma.read_latency));
    add("mma_input_skew", std::to_string(config.mma.input_skew));
    add("mma_output_skew", std::to_string(config.mma.output_skew));
    add("vector_lanes", std::to_string(config.vector.lanes));
    add("memory_bytes_per_cycle", std::to_string(config.memory.bytes_per_cycle));
    add("memory_latency", std::to_string(config.memory.latency));

    for (std::size_t unit = 0; unit < tile_npu::UNIT_COUNT; ++unit) {
      const auto prefix = std::string(UNIT_NAMES[unit]);
      add("unit_count_" + prefix, std::to_string(stats.unit_count[unit]));
      add("unit_busy_cycles_" + prefix, std::to_string(stats.unit_busy_cycles[unit]));
      add("unit_utilization_" + prefix,
          number(per_cycle(stats.unit_busy_cycles[unit], stats.cycles)));
      add("unit_in_flight_peak_" + prefix, std::to_string(stats.unit_in_flight_peak[unit]));
      add("queue_depth_" + prefix, std::to_string(config.queue_depth[unit]));
      add("dispatch_stall_cycles_" + prefix, std::to_string(stats.dispatch_stall_cycles[unit]));
      add("dependency_stall_cycles_" + prefix, std::to_string(stats.dependency_stall_cycles[unit]));
      add("queue_occupancy_cycles_" + prefix, std::to_string(stats.queue_occupancy_cycles[unit]));
      add("queue_peak_" + prefix, std::to_string(stats.queue_peak[unit]));
    }

    const auto add_sram_stats = [&add](const std::string &prefix, const auto &sram_stats) {
      add(prefix + "accesses", std::to_string(sram_stats.accesses));
      add(prefix + "reads", std::to_string(sram_stats.reads));
      add(prefix + "writes", std::to_string(sram_stats.writes));
      add(prefix + "read_bytes", std::to_string(sram_stats.read_bytes));
      add(prefix + "write_bytes", std::to_string(sram_stats.write_bytes));
      add(prefix + "conflicts", std::to_string(sram_stats.conflicts));
      add(prefix + "busy_cycles", std::to_string(sram_stats.busy_cycles));
    };

    add_sram_stats("sram_", stats.sram);
    for (std::size_t domain = tile_npu::FIRST_SRAM_DOMAIN; domain < tile_npu::SRAM_DOMAIN_COUNT;
         ++domain) {
      const auto prefix = "sram_" + std::string(SRAM_DOMAIN_NAMES[domain]) + "_";
      const auto &domain_stats = stats.sram.domains[domain];
      add_sram_stats(prefix, domain_stats);
      add(prefix + "banks", std::to_string(config.sram.domains[domain].banks));
      add(prefix + "ports", std::to_string(config.sram.domains[domain].ports_per_bank));
      add(prefix + "width", std::to_string(config.sram.domains[domain].bank_width_bytes));
    }
    add("sram_bandwidth_bytes_per_cycle",
        number(per_cycle(stats.sram.read_bytes + stats.sram.write_bytes, stats.cycles)));

    add("memory_transfers", std::to_string(stats.memory.transfers));
    add("memory_reads", std::to_string(stats.memory.reads));
    add("memory_writes", std::to_string(stats.memory.writes));
    add("memory_read_bytes", std::to_string(stats.memory.read_bytes));
    add("memory_write_bytes", std::to_string(stats.memory.write_bytes));
    add("memory_busy_cycles", std::to_string(stats.memory.busy_cycles));
    add("memory_row_hits", std::to_string(stats.memory.row_hits));
    add("memory_row_misses", std::to_string(stats.memory.row_misses));
    add("memory_bank_conflicts", std::to_string(stats.memory.bank_conflicts));
    add("memory_outstanding_peak", std::to_string(stats.memory.outstanding_peak));
    add("memory_bandwidth_bytes_per_cycle",
        number(per_cycle(stats.memory.read_bytes + stats.memory.write_bytes, stats.cycles)));
  }

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "cannot open stats file: " << path << '\n';
    return false;
  }
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (i != 0) out << ',';
    out << csv_escape(fields[i].first);
  }
  out << '\n';
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (i != 0) out << ',';
    out << csv_escape(fields[i].second);
  }
  out << '\n';
  return out.good();
}

} // namespace

int main(int argc, char **argv) {
  std::string mode_name = "functional";
  std::int64_t memory_mib = static_cast<std::int64_t>(tile_npu::frontend::DEFAULT_MEMORY_MIB);
  std::int64_t steps = -1;
  std::string stats_json;
  std::string stats_csv;
  std::string trace_out;
  std::string trace_in;
  std::string timeline_json;
  std::string timeline_csv;
  std::uint64_t trace_start = 0;
  std::uint64_t trace_count = 0;
  std::string uarch_config_path;
  std::string dump_uarch_config;
  std::vector<std::string> uarch_sets;
  std::uint64_t window_depth = 0;
  std::uint64_t queue_depth = 0;
  std::uint64_t vector_lanes = 0;
  std::uint64_t sram_banks = 0;
  std::uint64_t mem_bytes_per_cycle = 0;
  std::int64_t mem_latency = -1;
  std::vector<std::string> image_args;

  CLI::App app{"Tile-NPU functional/uarch simulator"};
  app.add_option("--mode", mode_name, "Execution mode: functional, uarch, or replay")
      ->check(CLI::IsMember({"functional", "uarch", "replay"}))
      ->default_val("functional");
  app.add_option("-m,--memory", memory_mib, "RAM size in MiB")->check(CLI::PositiveNumber);
  app.add_option("-s,--steps", steps, "Retired-instruction limit (-1: unlimited)");
  app.add_option("--stats-json", stats_json, "Write the final status and counters as JSON");
  app.add_option("--stats-csv", stats_csv, "Write the final status and counters as one CSV row");
  app.add_option("--trace-out", trace_out, "Write the DynInst stream as a UARCH trace");
  app.add_option("--trace-in", trace_in, "Replay a UARCH trace (requires --mode=replay)");
  app.add_option("--timeline-json", timeline_json,
                 "Write per-instruction dispatch/issue/complete history as JSON");
  app.add_option("--timeline-csv", timeline_csv,
                 "Write per-instruction dispatch/issue/complete history as CSV");
  app.add_option("--trace-start", trace_start, "First DynInst id to record (default 0)");
  app.add_option("--trace-count", trace_count, "Maximum records to record (0: all)");
  app.add_option("--uarch-config", uarch_config_path,
                 "Load uarch timing parameters from a JSON object");
  app.add_option("--uarch-set", uarch_sets,
                 "Override one uarch parameter as KEY=VALUE (repeatable, applied after the file)");
  app.add_option("--dump-uarch-config", dump_uarch_config,
                 "Write the effective uarch configuration as JSON");
  app.add_option("--window-depth", window_depth, "Dispatch window capacity")
      ->check(CLI::PositiveNumber);
  app.add_option("--queue-depth", queue_depth, "Per-unit issue queue capacity")
      ->check(CLI::PositiveNumber);
  app.add_option("--vector-lanes", vector_lanes, "Vector datapath lane count")
      ->check(CLI::PositiveNumber);
  app.add_option("--sram-banks", sram_banks, "Bank count for every register-file domain")
      ->check(CLI::PositiveNumber);
  app.add_option("--mem-bytes-per-cycle", mem_bytes_per_cycle, "Off-chip bus width in bytes")
      ->check(CLI::PositiveNumber);
  app.add_option("--mem-latency", mem_latency, "Off-chip memory latency in cycles")
      ->check(CLI::NonNegativeNumber);
  app.add_option("images", image_args, "Program and data images (path[@0xADDR])");
  CLI11_PARSE(app, argc, argv);

  tile_npu::UarchConfig uarch_config;
  std::string config_error;
  if (!uarch_config_path.empty() &&
      !tile_npu::load_uarch_config(uarch_config_path, uarch_config, config_error)) {
    std::cerr << "uarch config: " << config_error << '\n';
    return 2;
  }
  std::vector<std::string> overrides = uarch_sets;
  const auto add_override = [&overrides](std::string_view key, std::uint64_t value) {
    overrides.push_back(std::string(key) + "=" + std::to_string(value));
  };
  if (window_depth != 0) add_override("window_depth", window_depth);
  if (queue_depth != 0) {
    for (const auto unit : UNIT_NAMES) {
      add_override(std::string("queue_depth.") + std::string(unit), queue_depth);
    }
  }
  if (vector_lanes != 0) add_override("vector.lanes", vector_lanes);
  if (sram_banks != 0) {
    for (std::size_t domain = tile_npu::FIRST_SRAM_DOMAIN; domain < tile_npu::SRAM_DOMAIN_COUNT;
         ++domain) {
      add_override(std::string("sram.") + std::string(SRAM_DOMAIN_NAMES[domain]) + ".banks",
                   sram_banks);
    }
  }
  if (mem_bytes_per_cycle != 0) add_override("memory.bytes_per_cycle", mem_bytes_per_cycle);
  if (mem_latency >= 0) add_override("memory.latency", static_cast<std::uint64_t>(mem_latency));

  for (const auto &setting : overrides) {
    if (!tile_npu::apply_uarch_setting(setting, uarch_config, config_error)) {
      std::cerr << "uarch set: " << config_error << '\n';
      return 2;
    }
  }
  if (!dump_uarch_config.empty()) {
    std::ofstream out(dump_uarch_config, std::ios::out | std::ios::trunc);
    if (!out) {
      std::cerr << "cannot open uarch config output: " << dump_uarch_config << '\n';
      return 1;
    }
    out << tile_npu::uarch_config_json(uarch_config);
    if (!out.good()) {
      std::cerr << "cannot write uarch config output: " << dump_uarch_config << '\n';
      return 1;
    }
  }

  const bool want_timeline = !timeline_json.empty() || !timeline_csv.empty();
  if (mode_name == "replay") {
    if (trace_in.empty()) {
      std::cerr << "replay mode requires --trace-in\n";
      return 2;
    }
    if (!trace_out.empty()) {
      std::cerr << "replay mode cannot write --trace-out\n";
      return 2;
    }
    if (!stats_csv.empty()) {
      std::cerr << "replay mode currently supports --stats-json only\n";
      return 2;
    }
    tile_npu::TraceReplay replay(uarch_config, trace_in, want_timeline);
    std::string replay_error;
    if (!replay.run(replay_error)) {
      std::cerr << "trace replay: " << replay_error << '\n';
      return 1;
    }
    const auto &stats = replay.stats();
    std::cerr << "mode=replay cycles=" << stats.cycles << " retired=" << stats.retired;
    for (std::size_t unit = 0; unit < tile_npu::UNIT_COUNT; ++unit) {
      std::cerr << ' ' << UNIT_NAMES[unit] << '=' << stats.unit_count[unit];
    }
    std::cerr << '\n';
    if (!stats_json.empty() && !write_replay_json(stats_json, uarch_config, stats)) return 1;
    if (!timeline_json.empty() && !write_timeline_json(timeline_json, replay.timeline())) {
      return 1;
    }
    if (!timeline_csv.empty() && !write_timeline_csv(timeline_csv, replay.timeline())) {
      return 1;
    }
    return 0;
  }

  if (want_timeline && mode_name == "functional") {
    std::cerr << "timeline output requires --mode=uarch or --mode=replay\n";
    return 2;
  }
  if (image_args.empty()) {
    std::cerr << "at least one program image is required\n";
    return 2;
  }
  if (!trace_in.empty()) {
    std::cerr << "--trace-in is only valid with --mode=replay\n";
    return 2;
  }

  std::vector<tile_npu::frontend::Image> images;
  images.reserve(image_args.size());
  for (const auto &arg : image_args) {
    tile_npu::frontend::Image image{};
    if (!tile_npu::frontend::parse_image(arg, image)) {
      std::cerr << "invalid image: " << arg << '\n';
      return 2;
    }
    images.push_back(std::move(image));
  }

  tile_npu::frontend::Platform platform(static_cast<std::size_t>(memory_mib));
  if (!platform.load_images(images)) return 1;

  const auto mode =
      mode_name == "uarch" ? tile_npu::ExecutionMode::uarch : tile_npu::ExecutionMode::functional;
  tile_npu::Machine machine(platform.bus(), mode, uarch_config);
  machine.reset(tile_npu::frontend::MEMORY_BASE);
  if (want_timeline) machine.set_timeline_enabled(true);

  std::unique_ptr<tile_npu::UarchTraceWriter> trace_writer;
  if (!trace_out.empty()) {
    trace_writer =
        std::make_unique<tile_npu::UarchTraceWriter>(trace_out, trace_start, trace_count);
    if (!trace_writer->good()) {
      std::cerr << "cannot open trace output: " << trace_out << '\n';
      return 1;
    }
    machine.set_trace_writer(trace_writer.get());
  }

  const auto limit =
      steps < 0 ? std::numeric_limits<std::uint64_t>::max() : static_cast<std::uint64_t>(steps);
  tile_npu::StepResult result{};
  bool stopped = false;
  for (std::uint64_t i = 0; i < limit; ++i) {
    result = machine.step();
    if (result.status != tile_npu::StepStatus::retired) {
      stopped = true;
      break;
    }
  }
  if (trace_writer) trace_writer->close();

  if (mode == tile_npu::ExecutionMode::uarch) {
    if (!timeline_json.empty() && !write_timeline_json(timeline_json, machine.timeline())) {
      return 1;
    }
    if (!timeline_csv.empty() && !write_timeline_csv(timeline_csv, machine.timeline())) {
      return 1;
    }
    const auto &stats = machine.stats();
    std::cerr << "mode=uarch cycles=" << stats.cycles << " retired=" << stats.retired;
    for (std::size_t unit = 0; unit < tile_npu::UNIT_COUNT; ++unit) {
      std::cerr << ' ' << UNIT_NAMES[unit] << '=' << stats.unit_count[unit];
    }
    std::cerr << '\n';
  }

  int exit_code = 1;
  if (!stopped) {
    std::cerr << "step limit reached\n";
  } else {
    switch (result.status) {
    case tile_npu::StepStatus::breakpoint:
    case tile_npu::StepStatus::kernel_end: {
      const auto a0 = machine.state().scalar.read_x(tile_npu::frontend::EXIT_CODE_REG);
      if (a0 != 0) {
        std::cerr << "guest exited with a0=" << a0 << '\n';
      }
      exit_code = static_cast<int>(a0);
      break;
    }
    default:
      std::cerr << tile_npu::status_name(result.status) << " at pc=" << tile_npu::hex64(result.pc);
      if (result.instr) std::cerr << " instruction=" << tile_npu::hex32(*result.instr);
      if (result.fault_address) std::cerr << " address=" << tile_npu::hex64(*result.fault_address);
      std::cerr << '\n';
      break;
    }
  }

  if (!stats_json.empty() &&
      !write_run_json(stats_json, mode, uarch_config, machine, result, exit_code)) {
    return 1;
  }
  if (!stats_csv.empty() &&
      !write_run_csv(stats_csv, mode, uarch_config, machine, result, exit_code)) {
    return 1;
  }
  return exit_code;
}
