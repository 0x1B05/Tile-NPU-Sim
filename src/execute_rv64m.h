#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_rv64m(const DecodedInstruction &instruction,
                            ScalarState &scalar);

} // namespace tile_npu
