#include "orlin/examples.hpp"
#include "orlin/orlin_sfm.hpp"
#include "orlin/verification.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    void verify_double(const orlin::SubmodularOracle &oracle)
    {
        orlin::SolverOptions options;
        options.verify_invariants = true;
        const auto result = orlin::DoubleOrlinSFM(oracle, options).minimize();
        const auto exact = orlin::verification::exhaustive_minimize(oracle);
        if (result.minimum_value != exact.minimum_value)
        {
            throw std::runtime_error(oracle.name() +
                                     ": double backend found the wrong objective");
        }
        orlin::Subset returned(oracle.size(), 0);
        for (const auto vertex : result.minimizer)
            returned[vertex] = 1;
        if (oracle.evaluate(returned) != exact.minimum_value)
        {
            throw std::runtime_error(oracle.name() +
                                     ": double backend returned an inconsistent set");
        }
    }

} // namespace

int main()
{
    std::size_t cases = 0;
    for (const auto &name : orlin::examples::fixture_names())
    {
        const auto oracle = orlin::examples::make_fixture(name);
        verify_double(*oracle);
        ++cases;
    }

    const std::vector<orlin::Scalar> weights{
        orlin::Scalar(-1) / 2, orlin::Scalar(2) / 3,
        orlin::Scalar(-5) / 7, orlin::Scalar(0)};
    orlin::LambdaOracle fractional(
        weights.size(),
        [weights](const orlin::Subset &subset)
        {
            orlin::Scalar value = orlin::Scalar(1) / 3;
            for (std::size_t i = 0; i < weights.size(); ++i)
            {
                if (subset[i])
                    value += weights[i];
            }
            return value;
        },
        "fractional_modular_double");
    verify_double(fractional);
    ++cases;

    std::mt19937_64 random(0xD0B1EULL);
    std::uniform_int_distribution<int> size_distribution(1, 8);
    std::uniform_int_distribution<int> unary_distribution(-9, 9);
    std::uniform_int_distribution<int> capacity_distribution(1, 7);
    std::bernoulli_distribution include(0.30);

    for (int trial = 0; trial < 80; ++trial)
    {
        const std::size_t n = static_cast<std::size_t>(size_distribution(random));
        std::vector<std::int64_t> unary(n);
        for (auto &value : unary)
            value = unary_distribution(random);
        std::vector<orlin::examples::Arc> arcs;
        for (std::size_t tail = 0; tail < n; ++tail)
        {
            for (std::size_t head = 0; head < n; ++head)
            {
                if (tail != head && include(random))
                {
                    arcs.push_back({tail, head, capacity_distribution(random)});
                }
            }
        }
        orlin::examples::DirectedCutOracle oracle(
            std::move(unary), std::move(arcs),
            "random_double_cut_" + std::to_string(trial));
        verify_double(oracle);
        ++cases;
    }

    for (int trial = 0; trial < 80; ++trial)
    {
        const std::size_t n = static_cast<std::size_t>(size_distribution(random));
        std::vector<std::int64_t> modular(n);
        for (auto &value : modular)
            value = unary_distribution(random);
        std::vector<orlin::examples::CoverageFeature> features;
        for (std::size_t feature = 0; feature < n + 2; ++feature)
        {
            orlin::examples::CoverageFeature item;
            item.weight = capacity_distribution(random);
            for (std::size_t vertex = 0; vertex < n; ++vertex)
            {
                if (include(random))
                    item.items.push_back(vertex);
            }
            if (item.items.empty())
                item.items.push_back(feature % n);
            features.push_back(std::move(item));
        }
        orlin::examples::CoverageOracle oracle(
            std::move(modular), std::move(features),
            "random_double_coverage_" + std::to_string(trial));
        verify_double(oracle);
        ++cases;
    }

    for (int trial = 0; trial < 60; ++trial)
    {
        const std::size_t n = static_cast<std::size_t>(size_distribution(random));
        std::vector<std::int64_t> modular(n);
        for (auto &value : modular)
            value = unary_distribution(random);
        orlin::examples::ConcaveCardinalityOracle oracle(
            std::move(modular), capacity_distribution(random),
            "random_double_concave_" + std::to_string(trial));
        verify_double(oracle);
        ++cases;
    }

    std::cout << "Double-backend regression tests passed: cases=" << cases
              << '\n';
}
