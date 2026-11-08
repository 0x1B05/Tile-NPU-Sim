#include "uarch/uarch_config_json.h"

#include "uarch/uarch_names.h"

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

template <typename T>
bool set_number(std::uint64_t value, std::uint64_t minimum, std::string_view key,
                std::string &error, T &slot) {
  if (value < minimum) {
    error = std::string(key) + " must be at least " + std::to_string(minimum);
    return false;
  }
  slot = static_cast<T>(value);
  return true;
}

bool apply_setting(std::string_view key, std::string_view text, UarchConfig &config,
                   std::string &error) {
  std::uint64_t value = 0;
  if (!parse_u64(text, value)) {
    error = "invalid unsigned integer for " + std::string(key) + ": " + std::string(text);
    return false;
  }

  if (key == "window_depth") return set_number(value, 1, key, error, config.window_depth);

  constexpr std::string_view queue_prefix = "queue_depth.";
  if (key.starts_with(queue_prefix)) {
    const auto unit = parse_unit(key.substr(queue_prefix.size()));
    if (!unit) {
      error = "unknown unit in " + std::string(key);
      return false;
    }
    return set_number(value, 1, key, error, config.queue_depth[unit_index(*unit)]);
  }

  if (key == "mma.read_latency") {
    return set_number(value, 0, key, error, config.mma.read_latency);
  }
  if (key == "mma.input_skew") return set_number(value, 0, key, error, config.mma.input_skew);
  if (key == "mma.output_skew") return set_number(value, 0, key, error, config.mma.output_skew);
  if (key == "vector.lanes") return set_number(value, 1, key, error, config.vector.lanes);
  if (key == "memory.bytes_per_cycle") {
    return set_number(value, 1, key, error, config.memory.bytes_per_cycle);
  }
  if (key == "memory.latency") return set_number(value, 0, key, error, config.memory.latency);

  constexpr std::string_view sram_prefix = "sram.";
  if (key.starts_with(sram_prefix)) {
    const auto rest = key.substr(sram_prefix.size());
    const auto dot = rest.rfind('.');
    if (dot == std::string_view::npos) {
      error = "expected sram.<domain>.<field>: " + std::string(key);
      return false;
    }
    const auto domain = parse_sram_domain(rest.substr(0, dot));
    if (!domain) {
      error = "unknown SRAM domain in " + std::string(key);
      return false;
    }
    const auto field = rest.substr(dot + 1);
    auto &domain_config = config.sram.domains[domain_index(*domain)];
    if (field == "banks") return set_number(value, 1, key, error, domain_config.banks);
    if (field == "ports" || field == "ports_per_bank") {
      return set_number(value, 1, key, error, domain_config.ports_per_bank);
    }
    if (field == "width" || field == "bank_width_bytes") {
      return set_number(value, 1, key, error, domain_config.bank_width_bytes);
    }
    error = "unknown SRAM field in " + std::string(key);
    return false;
  }

  error = "unknown uarch setting: " + std::string(key);
  return false;
}

} // namespace

bool apply_uarch_setting(std::string_view setting, UarchConfig &config, std::string &error) {
  const auto equals = setting.find('=');
  if (equals == std::string_view::npos) {
    error = "expected key=value";
    return false;
  }
  return apply_setting(setting.substr(0, equals), setting.substr(equals + 1), config, error);
}

bool apply_uarch_config_json(std::string_view text, UarchConfig &config, std::string &error) {
  std::map<std::string, std::string> values;
  JsonParser parser(text);
  if (!parser.parse(values, error)) return false;

  for (const auto &[key, value] : values) {
    if (!apply_setting(key, value, config, error)) return false;
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
  for (std::size_t domain = FIRST_SRAM_DOMAIN; domain < SRAM_DOMAIN_COUNT; ++domain) {
    const auto &domain_config = config.sram.domains[domain];
    out << "    \"" << SRAM_DOMAIN_NAMES[domain] << "\": {\"banks\": " << domain_config.banks
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
