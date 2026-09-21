#include "orlin/examples.hpp"
#include "orlin/orlin_sfm.hpp"
#include "orlin/verification.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using orlin::Scalar;
using orlin::SolverResult;
using orlin::SubmodularOracle;

void verify_one(const SubmodularOracle& oracle, std::uint64_t& total_iterations,
                std::uint64_t& singular_steps,
                std::uint64_t& nonsingular_steps,
                std::uint64_t& reduce_calls) {
  try {
  const auto check = orlin::verification::exhaustive_submodularity_check(oracle);
  if (!check.is_submodular) {
    throw std::runtime_error(oracle.name() + " failed submodularity check: " +
                             check.failure);
  }
  orlin::SolverOptions options;
  options.verify_invariants = true;
  orlin::OrlinSFM solver(oracle, options);
  const SolverResult result = solver.minimize();
  const auto exact = orlin::verification::exhaustive_minimize(oracle);
  if (result.minimum_value != exact.minimum_value) {
    throw std::runtime_error(oracle.name() +
                             " disagrees with exhaustive minimization: Orlin=" +
                             result.minimum_value.str() + " exact=" +
                             exact.minimum_value.str());
  }
  orlin::Subset returned(oracle.size(), 0);
  for (const auto vertex : result.minimizer) returned[vertex] = 1;
  if (oracle.evaluate(returned) != exact.minimum_value) {
    throw std::runtime_error(oracle.name() + " returned an inconsistent set");
  }
  total_iterations += result.statistics.iterations;
  singular_steps += result.statistics.singular_auxiliary_systems;
  nonsingular_steps += result.statistics.nonsingular_auxiliary_systems;
  reduce_calls += result.statistics.reduce_calls;
  } catch (const std::exception& error) {
    if (const auto* cut =
            dynamic_cast<const orlin::examples::DirectedCutOracle*>(&oracle)) {
      std::cerr << "failing unary:";
      for (const auto value : cut->unary()) std::cerr << ' ' << value;
      std::cerr << "\nfailing arcs:";
      for (const auto& arc : cut->arcs()) {
        std::cerr << " (" << arc.tail << ',' << arc.head << ','
                  << arc.capacity << ')';
      }
      std::cerr << '\n';
    }
    throw std::runtime_error(oracle.name() + ": " + error.what());
  }
}

}  // namespace

int main() {
  std::uint64_t total_iterations = 0;
  std::uint64_t singular_steps = 0;
  std::uint64_t nonsingular_steps = 0;
  std::uint64_t reduce_calls = 0;
  std::uint64_t cases = 0;

  for (const auto& name : orlin::examples::fixture_names()) {
    const auto oracle = orlin::examples::make_fixture(name);
    verify_one(*oracle, total_iterations, singular_steps, nonsingular_steps,
               reduce_calls);
    ++cases;
  }

  {
    const std::vector<Scalar> weights{
        Scalar(-1) / 2, Scalar(2) / 3, Scalar(-5) / 7, Scalar(0)};
    orlin::LambdaOracle oracle(
        weights.size(),
        [weights](const orlin::Subset& subset) {
          Scalar value = Scalar(1) / 3;
          for (std::size_t i = 0; i < weights.size(); ++i) {
            if (subset[i]) value += weights[i];
          }
          return value;
        },
        "fractional_modular");
    verify_one(oracle, total_iterations, singular_steps, nonsingular_steps,
               reduce_calls);
    ++cases;
  }

  for (auto oracle : std::vector<orlin::examples::ModularOracle>{
           orlin::examples::ModularOracle({}, 7, "empty_ground"),
           orlin::examples::ModularOracle({0, 0, 0}, 5,
                                          "constant_function"),
           orlin::examples::ModularOracle({1, 2, 3}, 0, "all_positive"),
           orlin::examples::ModularOracle({-1, -2}, 0, "all_negative")}) {
    verify_one(oracle, total_iterations, singular_steps, nonsingular_steps,
               reduce_calls);
    ++cases;
  }

  std::mt19937_64 random(0x0A11CE5F0ULL);
  std::uniform_int_distribution<int> size_distribution(1, 8);
  std::uniform_int_distribution<int> unary_distribution(-9, 9);
  std::uniform_int_distribution<int> capacity_distribution(1, 7);
  std::bernoulli_distribution include_arc(0.30);
  std::bernoulli_distribution include_item(0.40);

  for (int trial = 0; trial < 120; ++trial) {
    const std::size_t n = static_cast<std::size_t>(size_distribution(random));
    std::vector<std::int64_t> unary(n);
    for (auto& value : unary) value = unary_distribution(random);
    std::vector<orlin::examples::Arc> arcs;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (i != j && include_arc(random)) {
          arcs.push_back({i, j, capacity_distribution(random)});
        }
      }
    }
    orlin::examples::DirectedCutOracle oracle(
        std::move(unary), std::move(arcs),
        "random_directed_cut_" + std::to_string(trial));
    verify_one(oracle, total_iterations, singular_steps, nonsingular_steps,
               reduce_calls);
    ++cases;
  }

  for (int trial = 0; trial < 120; ++trial) {
    const std::size_t n = static_cast<std::size_t>(size_distribution(random));
    std::vector<std::int64_t> modular(n);
    for (auto& value : modular) value = unary_distribution(random);
    const std::size_t feature_count = n + 2U;
    std::vector<orlin::examples::CoverageFeature> features;
    for (std::size_t feature = 0; feature < feature_count; ++feature) {
      orlin::examples::CoverageFeature item;
      item.weight = capacity_distribution(random);
      for (std::size_t vertex = 0; vertex < n; ++vertex) {
        if (include_item(random)) item.items.push_back(vertex);
      }
      if (item.items.empty()) item.items.push_back(feature % n);
      features.push_back(std::move(item));
    }
    orlin::examples::CoverageOracle oracle(
        std::move(modular), std::move(features),
        "random_coverage_" + std::to_string(trial));
    verify_one(oracle, total_iterations, singular_steps, nonsingular_steps,
               reduce_calls);
    ++cases;
  }

  for (int trial = 0; trial < 80; ++trial) {
    const std::size_t n = static_cast<std::size_t>(size_distribution(random));
    std::vector<std::int64_t> modular(n);
    for (auto& value : modular) value = unary_distribution(random);
    const std::int64_t scale = capacity_distribution(random);
    orlin::examples::ConcaveCardinalityOracle oracle(
        std::move(modular), scale,
        "random_concave_cardinality_" + std::to_string(trial));
    verify_one(oracle, total_iterations, singular_steps, nonsingular_steps,
               reduce_calls);
    ++cases;
  }

  if (singular_steps == 0 || nonsingular_steps == 0) {
    throw std::runtime_error(
        "random regression suite did not exercise both auxiliary cases");
  }

  std::cout << "Orlin regression tests passed: cases=" << cases
            << " iterations=" << total_iterations
            << " singular_gamma=" << singular_steps
            << " nonsingular_gamma=" << nonsingular_steps
            << " Reduce_calls=" << reduce_calls << '\n';
}
