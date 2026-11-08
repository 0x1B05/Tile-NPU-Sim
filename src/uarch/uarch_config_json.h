#pragma once

#include "uarch/uarch_config.h"

#include <string>
#include <string_view>

namespace tile_npu {

// Loads a JSON object whose nested keys follow the same dotted paths as
// --uarch-set (for example "sram.tile.banks"). Unknown keys are errors so a
// sweep typo cannot silently run with default parameters.
bool load_uarch_config(const std::string &path, UarchConfig &config, std::string &error);
bool apply_uarch_config_json(std::string_view text, UarchConfig &config, std::string &error);

// Applies one "key=value" override. Used by both JSON loading and CLI flags.
bool apply_uarch_setting(std::string_view setting, UarchConfig &config, std::string &error);

// Effective configuration in the same nested JSON shape accepted by load().
std::string uarch_config_json(const UarchConfig &config);

} // namespace tile_npu
