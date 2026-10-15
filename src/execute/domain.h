#pragma once

#include "config_check.h"
#include "execute.h"
#include "utils/unreachable.h"

#include <cstddef>
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

// The data, compute config, and memory descriptor of one register.
template <NpuDomain Domain> auto bind_register(State &state, RegisterIndex index) {
  return std::tie(domain_regs<Domain>(state)[index], domain_configs<Domain>(state)[index],
                  domain_descriptors<Domain>(state)[index]);
}

// Calls fn(regs, configs) on the register file and config array of the
// runtime domain. Only the listed domains instantiate fn.
template <NpuDomain First, NpuDomain... Rest, typename Fn>
ExecuteResult with_domain(State &state, NpuDomain domain, Fn &&fn) {
  if (domain == First) {
    return fn(domain_regs<First>(state), domain_configs<First>(state));
  }

  if constexpr (sizeof...(Rest) > 0) {
    return with_domain<Rest...>(state, domain, std::forward<Fn>(fn));
  }

  unreachable("domain rejected by structural validation");
}

// Calls fn(idx) for every index in the config's effective region:
// row * MATRIX_EXTENT + col for matrix configs, the lane for vector configs.
template <DomainConfig Config, typename Fn> void for_each_idx(const Config &config, Fn &&fn) {
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
