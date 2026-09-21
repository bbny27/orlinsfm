#pragma once

#include "orlin/orlin_sfm.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace orlin::verification
{

  struct ExhaustiveResult
  {
    std::vector<std::size_t> minimizer;
    Scalar minimum_value;
    std::uint64_t evaluations = 0;
    std::chrono::nanoseconds elapsed{};
  };

  // one bit encoding of a subset of {0,1}^n from a mask
  inline Subset subset_from_mask(std::size_t n, std::uint64_t mask)
  {
    Subset subset(n, 0);
    for (std::size_t i = 0; i < n; ++i)
    {
      subset[i] = static_cast<std::uint8_t>((mask >> i) & 1U);
    }
    return subset;
  }

  inline std::vector<std::size_t> indices(const Subset &subset)
  {
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < subset.size(); ++i)
    {
      if (subset[i])
        result.push_back(i);
    }
    return result;
  }

  // enumerates all subsets of the ground set and returns the minimum value and a minimizer
  inline ExhaustiveResult exhaustive_minimize(const SubmodularOracle &oracle,
                                              std::size_t maximum_n = 25)
  {
    using Clock = std::chrono::steady_clock;
    const std::size_t n = oracle.size();
    if (n > maximum_n || n >= 63)
    {
      throw std::invalid_argument("ground set is too large for exhaustive search");
    }
    const auto start = Clock::now();
    const std::uint64_t count = std::uint64_t{1} << n;
    ExhaustiveResult result;
    std::optional<Scalar> best;
    for (std::uint64_t mask = 0; mask < count; ++mask)
    {
      const Subset subset = subset_from_mask(n, mask);
      const Scalar value = oracle.evaluate(subset);
      ++result.evaluations;
      if (!best.has_value() || value < *best)
      {
        best = value;
        result.minimum_value = value;
        result.minimizer = indices(subset);
      }
    }
    result.elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
    return result;
  }

  struct SubmodularityCheck
  {
    bool is_submodular = true;
    std::string failure;
    std::uint64_t evaluations = 0;
    std::chrono::nanoseconds elapsed{};
  };

  // Checks the elementary two-element diminishing-returns inequalities
  // f(S+i)+f(S+j) >= f(S)+f(S+i+j), which are equivalent to submodularity.
  inline SubmodularityCheck exhaustive_submodularity_check(
      const SubmodularOracle &oracle, std::size_t maximum_n = 20)
  {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    SubmodularityCheck result;
    const std::size_t n = oracle.size();
    if (n > maximum_n || n >= 63)
    {
      throw std::invalid_argument(
          "ground set is too large for exhaustive submodularity checking");
    }
    const std::uint64_t count = std::uint64_t{1} << n;
    std::vector<Scalar> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t mask = 0; mask < count; ++mask)
    {
      values.push_back(oracle.evaluate(subset_from_mask(n, mask)));
      ++result.evaluations;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = i + 1; j < n; ++j)
      {
        const std::uint64_t i_bit = std::uint64_t{1} << i;
        const std::uint64_t j_bit = std::uint64_t{1} << j;
        for (std::uint64_t mask = 0; mask < count; ++mask)
        {
          if ((mask & (i_bit | j_bit)) != 0)
            continue;
          if (values[mask | i_bit] + values[mask | j_bit] <
              values[mask] + values[mask | i_bit | j_bit])
          {
            result.is_submodular = false;
            result.failure = "failed for S-mask=" + std::to_string(mask) +
                             ", i=" + std::to_string(i) +
                             ", j=" + std::to_string(j);
            result.elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - start);
            return result;
          }
        }
      }
    }
    result.elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
    return result;
  }

} // namespace orlin::verification
