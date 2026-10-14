#pragma once

// Domain-generic access to the register files: binding a runtime NpuDomain to
// its data/config arrays, and iterating over a config's effective region.
#include "config_check.h"
#include "execute.h"
#include "utils/unreachable.h"

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

namespace tile_npu {

template <NpuDomain Domain> auto &domain_regs(State &state) {
  if constexpr (Domain == NpuDomain::tile) {
    return state.t;
  } else if constexpr (Domain == NpuDomain::acc) {
    return state.a;
  } else if constexpr (Domain == NpuDomain::vec8) {
    return state.b;
  } else {
    return state.v;
  }
}

template <NpuDomain Domain> const auto &domain_configs(const State &state) {
  if constexpr (Domain == NpuDomain::tile) {
    return state.tc;
  } else if constexpr (Domain == NpuDomain::acc) {
    return state.ac;
  } else if constexpr (Domain == NpuDomain::vec8) {
    return state.bc;
  } else {
    return state.vc;
  }
}

template <NpuDomain Domain> auto &domain_descriptors(State &state) {
  if constexpr (Domain == NpuDomain::tile) {
    return state.tm;
  } else if constexpr (Domain == NpuDomain::acc) {
    return state.am;
  } else if constexpr (Domain == NpuDomain::vec8) {
    return state.bm;
  } else {
    return state.vm;
  }
}

// The data, read-only compute config, and memory descriptor of one register.
template <NpuDomain Domain> auto bind_register(State &state, RegisterIndex index) {
  return std::tie(domain_regs<Domain>(state)[index], domain_configs<Domain>(state)[index],
                  domain_descriptors<Domain>(state)[index]);
}

// Runs body(regs, configs) on the register file and config array of the
// runtime domain. Only the listed domains instantiate the body; structural
// validation guarantees the domain is one of them.
template <NpuDomain... Allowed, typename Body>
ExecuteResult with_domain(State &state, NpuDomain domain, Body &&body) {
  std::optional<ExecuteResult> result;
  ((domain == Allowed &&
    (result.emplace(body(domain_regs<Allowed>(state), domain_configs<Allowed>(state))), true)) ||
   ...);
  if (!result) {
    unreachable("domain rejected by structural validation");
  }
  return *result;
}

// Calls fn(offset) for every storage offset in the config's effective
// region: row * MATRIX_EXTENT + col for matrix configs, the lane for vector
// configs. Same-shape operands share offsets, so elementwise bodies index
// every register with the same offset.
template <DomainConfig Config, typename Fn> void for_each_offset(const Config &config, Fn &&fn) {
  if constexpr (MatrixConfig<Config>) {
    for (std::size_t row = 0; row < config.rows; ++row) {
      for (std::size_t col = 0; col < config.cols; ++col) {
        fn(row * MATRIX_EXTENT + col);
      }
    }
  } else {
    for (std::size_t lane = 0; lane < config.len; ++lane) {
      fn(lane);
    }
  }
}

} // namespace tile_npu
