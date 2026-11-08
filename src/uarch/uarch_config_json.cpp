#include "uarch/uarch_config_json.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace tile_npu {
namespace {

constexpr std::array<const char *, UNIT_COUNT> UNIT_NAMES{"scalar", "mma",    "vector",
                                                          "dma",    "config", "sync"};

constexpr std::array<const char *, SRAM_DOMAIN_COUNT> DOMAIN_NAMES{"scalar", "tile", "acc",
                                                                   "vec32"};

std::string full_key(std::string_view prefix, std::string_view key) {
  if (prefix.empty()) return std::string(key);
  return std::string(prefix) + "." + std::string(key);
}

class JsonParser {
public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  bool parse(std::map<std::string, std::string> &values, std::string &error) {
    skip_space();
    if (!parse_object("", values, error)) return false;
    skip_space();
    if (position_ != input_.size()) {
      return fail("trailing characters after the root object", error);
    }
    return true;
  }

private:
  bool fail(std::string_view message, std::string &error) const {
    error = std::string(message) + " at offset " + std::to_string(position_);
    return false;
  }

  void skip_space() {
    while (position_ < input_.size()) {
      const char ch = input_[position_];
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') break;
      ++position_;
    }
  }

  bool consume(char expected) {
    if (position_ >= input_.size() || input_[position_] != expected) return false;
    ++position_;
    return true;
  }

  bool parse_string(std::string &out, std::string &error) {
    if (!consume('"')) return fail("expected a string", error);
    out.clear();
    while (position_ < input_.size()) {
      const char ch = input_[position_++];
      if (ch == '"') return true;
      if (ch != '\\') {
        out.push_back(ch);
        continue;
      }
      if (position_ >= input_.size()) return fail("unterminated escape", error);
      const char escaped = input_[position_++];
      switch (escaped) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      default: return fail("unsupported string escape", error);
      }
    }
    return fail("unterminated string", error);
  }

  bool parse_scalar(std::string &out, std::string &error) {
    const auto begin = position_;
    while (position_ < input_.size()) {
      const char ch = input_[position_];
      if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == ',' || ch == '}') break;
      ++position_;
    }
    if (position_ == begin) return fail("expected a value", error);
    out = std::string(input_.substr(begin, position_ - begin));
    if (out == "true" || out == "false") return true;
    char *end = nullptr;
    std::strtod(out.c_str(), &end);
    if (end == out.c_str() || *end != '\0') return fail("expected a number", error);
    return true;
  }

  bool parse_object(std::string_view prefix, std::map<std::string, std::string> &values,
                    std::string &error) {
    if (!consume('{')) return fail("expected an object", error);
    skip_space();
    if (consume('}')) return true;

    while (true) {
      std::string key;
      if (!parse_string(key, error)) return false;
      skip_space();
      if (!consume(':')) return fail("expected ':' after a key", error);
      skip_space();

      const auto path = full_key(prefix, key);
      if (position_ < input_.size() && input_[position_] == '{') {
        if (!parse_object(path, values, error)) return false;
      } else {
        std::string value;
        if (position_ < input_.size() && input_[position_] == '"') {
          if (!parse_string(value, error)) return false;
        } else if (!parse_scalar(value, error)) {
          return false;
        }
        values[path] = std::move(value);
      }

      skip_space();
      if (consume(',')) {
        skip_space();
        continue;
      }
      if (consume('}')) return true;
      return fail("expected ',' or '}'", error);
    }
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

bool parse_u64(std::string_view text, std::uint64_t &value) {
  const auto *begin = text.data();
  const auto *end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  return result.ec == std::errc{} && result.ptr == end;
}

bool parse_unit(std::string_view text, Unit &unit) {
  for (std::size_t i = 0; i < UNIT_COUNT; ++i) {
    if (text == UNIT_NAMES[i]) {
      unit = static_cast<Unit>(i);
      return true;
    }
  }
  return false;
}

bool parse_domain(std::string_view text, Domain &domain) {
  if (text == "tile") {
    domain = Domain::tile;
    return true;
  }
  if (text == "acc") {
    domain = Domain::acc;
    return true;
  }
  if (text == "vec32" || text == "vec") {
    domain = Domain::vec32;
    return true;
  }
  return false;
}

bool set_value(std::uint64_t value, std::uint64_t minimum, std::string_view key, std::string &error,
               std::uint64_t &slot) {
  if (value < minimum) {
    error = std::string(key) + " must be at least " + std::to_string(minimum);
    return false;
  }
  slot = value;
  return true;
}

} // namespace

bool apply_uarch_setting(std::string_view setting, UarchConfig &config, std::string &error) {
  const auto equals = setting.find('=');
  if (equals == std::string_view::npos) {
    error = "expected key=value";
    return false;
  }
  const auto key = setting.substr(0, equals);
  const auto text = setting.substr(equals + 1);
  std::uint64_t value = 0;
  if (!parse_u64(text, value)) {
    error = "invalid unsigned integer for " + std::string(key) + ": " + std::string(text);
    return false;
  }

  if (key == "window_depth") {
    std::uint64_t bounded = 0;
    if (!set_value(value, 1, key, error, bounded)) return false;
    config.window_depth = static_cast<std::size_t>(bounded);
    return true;
  }

  constexpr std::string_view queue_prefix = "queue_depth.";
  if (key.starts_with(queue_prefix)) {
    Unit unit{};
    if (!parse_unit(key.substr(queue_prefix.size()), unit)) {
      error = "unknown unit in " + std::string(key);
      return false;
    }
    std::uint64_t bounded = 0;
    if (!set_value(value, 1, key, error, bounded)) return false;
    config.queue_depth[unit_index(unit)] = static_cast<std::size_t>(bounded);
    return true;
  }

  if (key == "mma.read_latency") {
    config.mma.read_latency = value;
    return true;
  }
  if (key == "mma.input_skew") {
    config.mma.input_skew = value;
    return true;
  }
  if (key == "mma.output_skew") {
    config.mma.output_skew = value;
    return true;
  }
  if (key == "vector.lanes") {
    std::uint64_t bounded = 0;
    if (!set_value(value, 1, key, error, bounded)) return false;
    config.vector.lanes = bounded;
    return true;
  }
  if (key == "memory.bytes_per_cycle") {
    std::uint64_t bounded = 0;
    if (!set_value(value, 1, key, error, bounded)) return false;
    config.memory.bytes_per_cycle = bounded;
    return true;
  }
  if (key == "memory.latency") {
    config.memory.latency = value;
    return true;
  }

  constexpr std::string_view sram_prefix = "sram.";
  if (key.starts_with(sram_prefix)) {
    const auto rest = key.substr(sram_prefix.size());
    const auto dot = rest.rfind('.');
    if (dot == std::string_view::npos) {
      error = "expected sram.<domain>.<field>: " + std::string(key);
      return false;
    }
    Domain domain{};
    if (!parse_domain(rest.substr(0, dot), domain)) {
      error = "unknown SRAM domain in " + std::string(key);
      return false;
    }
    const auto field = rest.substr(dot + 1);
    auto &domain_config = config.sram.domains[static_cast<std::size_t>(domain)];
    std::uint64_t bounded = 0;
    if (field == "banks") {
      if (!set_value(value, 1, key, error, bounded)) return false;
      domain_config.banks = static_cast<std::size_t>(bounded);
      return true;
    }
    if (field == "ports" || field == "ports_per_bank") {
      if (!set_value(value, 1, key, error, bounded)) return false;
      domain_config.ports_per_bank = static_cast<std::size_t>(bounded);
      return true;
    }
    if (field == "width" || field == "bank_width_bytes") {
      if (!set_value(value, 1, key, error, bounded)) return false;
      domain_config.bank_width_bytes = static_cast<std::size_t>(bounded);
      return true;
    }
    error = "unknown SRAM field in " + std::string(key);
    return false;
  }

  error = "unknown uarch setting: " + std::string(key);
  return false;
}

bool apply_uarch_config_json(std::string_view text, UarchConfig &config, std::string &error) {
  std::map<std::string, std::string> values;
  JsonParser parser(text);
  if (!parser.parse(values, error)) return false;

  for (const auto &[key, value] : values) {
    if (!apply_uarch_setting(key + "=" + value, config, error)) return false;
  }
  return true;
}

bool load_uarch_config(const std::string &path, UarchConfig &config, std::string &error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "cannot open uarch config: " + path;
    return false;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return apply_uarch_config_json(buffer.str(), config, error);
}

std::string uarch_config_json(const UarchConfig &config) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"window_depth\": " << config.window_depth << ",\n";
  out << "  \"queue_depth\": {";
  for (std::size_t i = 0; i < UNIT_COUNT; ++i) {
    if (i != 0) out << ", ";
    out << '"' << UNIT_NAMES[i] << "\": " << config.queue_depth[i];
  }
  out << "},\n";
  out << "  \"mma\": {\"read_latency\": " << config.mma.read_latency
      << ", \"input_skew\": " << config.mma.input_skew
      << ", \"output_skew\": " << config.mma.output_skew << "},\n";
  out << "  \"vector\": {\"lanes\": " << config.vector.lanes << "},\n";
  out << "  \"memory\": {\"bytes_per_cycle\": " << config.memory.bytes_per_cycle
      << ", \"latency\": " << config.memory.latency << "},\n";
  out << "  \"sram\": {\n";
  for (std::size_t domain = 1; domain < SRAM_DOMAIN_COUNT; ++domain) {
    const auto &domain_config = config.sram.domains[domain];
    out << "    \"" << DOMAIN_NAMES[domain] << "\": {\"banks\": " << domain_config.banks
        << ", \"ports\": " << domain_config.ports_per_bank
        << ", \"width\": " << domain_config.bank_width_bytes << "}";
    if (domain + 1 != SRAM_DOMAIN_COUNT) out << ',';
    out << '\n';
  }
  out << "  }\n";
  out << "}\n";
  return out.str();
}

} // namespace tile_npu
