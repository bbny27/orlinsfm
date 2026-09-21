#pragma once

#include "orlin/orlin_sfm.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace orlin::examples
{

  // Modular Functions
  // Submodular with equality in every elementary inequality
  // its optimum includes all negative-weight items and excludes all positive-weight items; zero-weight choices tie.
  class ModularOracle final : public SubmodularOracle
  {
  public:
    ModularOracle(std::vector<std::int64_t> weights, std::int64_t constant = 0,
                  std::string name = "modular")
        : weights_(std::move(weights)), constant_(constant), name_(std::move(name)) {} // accept weights and constant

    std::size_t size() const override { return weights_.size(); } // number of weights = ground set size

    Scalar evaluate(const Subset &subset) const override
    {
      Scalar result(constant_); // include constant term for every set
      for (std::size_t i = 0; i < weights_.size(); ++i)
      {
        if (subset[i])
          result += Scalar(weights_[i]);
      }
      return result;
    }

    std::string name() const override { return name_; }

  private:
    std::vector<std::int64_t> weights_;
    std::int64_t constant_;
    std::string name_;
  };

  // \F(S)=\sum_{i\in S} c_i-s\frac{|S|(|S|-1)}2
  // Concave cardinality function

  class ConcaveCardinalityOracle final : public SubmodularOracle
  {
  public:
    // f(S) = -scale * |S|(|S|-1)/2 + sum_{i in S} modular_i.
    // The successive cardinality increments -scale*(k-1) are non-increasing.
    ConcaveCardinalityOracle(std::vector<std::int64_t> modular,
                             std::int64_t scale,
                             std::string name = "concave_cardinality")
        : modular_(std::move(modular)), scale_(scale), name_(std::move(name))
    {
      if (scale_ < 0)
      {
        throw std::invalid_argument("concavity scale must be non-negative");
      }
    }

    std::size_t size() const override { return modular_.size(); }

    Scalar evaluate(const Subset &subset) const override
    {
      std::int64_t cardinality = 0;
      Scalar result(0);
      for (std::size_t i = 0; i < modular_.size(); ++i)
      {
        if (subset[i])
        {
          ++cardinality;
          result += Scalar(modular_[i]);
        }
      }
      result -= Scalar(scale_) * Scalar(cardinality) * Scalar(cardinality - 1) /
                Scalar(2);
      return result;
    }

    std::string name() const override { return name_; }

  private:
    std::vector<std::int64_t> modular_;
    std::int64_t scale_;
    std::string name_;
  };

  // Cut functions are submodular,
  // given a directed graph with non-negative edge weights,
  // the value of a subset S of vertices is the sum of the weights of edges leaving S
  struct Arc
  {
    std::size_t tail = 0;
    std::size_t head = 0;
    std::int64_t capacity = 0;
  };

  class DirectedCutOracle final : public SubmodularOracle
  {
  public:
    // f(S) = sum_{(i,j): i in S, j not in S} c_ij + sum_{i in S} unary_i.
    DirectedCutOracle(std::vector<std::int64_t> unary, std::vector<Arc> arcs,
                      std::string name = "directed_cut")
        : unary_(std::move(unary)), arcs_(std::move(arcs)), name_(std::move(name))
    {
      for (const auto &arc : arcs_)
      {
        if (arc.tail >= unary_.size() || arc.head >= unary_.size() ||
            arc.capacity < 0)
        {
          throw std::invalid_argument("invalid directed-cut arc");
        }
      }
    }

    std::size_t size() const override { return unary_.size(); }

    Scalar evaluate(const Subset &subset) const override
    {
      Scalar result(0);
      for (std::size_t i = 0; i < unary_.size(); ++i)
      {
        if (subset[i])
          result += Scalar(unary_[i]);
      }
      for (const auto &arc : arcs_)
      {
        if (subset[arc.tail] && !subset[arc.head])
        {
          result += Scalar(arc.capacity);
        }
      }
      return result;
    }

    std::string name() const override { return name_; }

    const std::vector<std::int64_t> &unary() const { return unary_; }
    const std::vector<Arc> &arcs() const { return arcs_; }

  private:
    std::vector<std::int64_t> unary_;
    std::vector<Arc> arcs_;
    std::string name_;
  };

  // The function adds the weights of all covered features,
  // counting each feature only once, plus the individual costs of the selected items.

  struct CoverageFeature
  {
    std::int64_t weight = 0;
    std::vector<std::size_t> items;
  };

  class CoverageOracle final : public SubmodularOracle
  {
  public:
    // f(S) = sum_j w_j 1{S intersects feature_j} + sum_{i in S} modular_i.
    CoverageOracle(std::vector<std::int64_t> modular,
                   std::vector<CoverageFeature> features,
                   std::string name = "coverage")
        : modular_(std::move(modular)),
          features_(std::move(features)),
          name_(std::move(name))
    {
      for (const auto &feature : features_)
      {
        if (feature.weight < 0)
        {
          throw std::invalid_argument("coverage weights must be non-negative");
        }
        for (const auto item : feature.items)
        {
          if (item >= modular_.size())
          {
            throw std::invalid_argument("coverage feature has invalid item");
          }
        }
      }
    }

    std::size_t size() const override { return modular_.size(); }

    Scalar evaluate(const Subset &subset) const override
    {
      Scalar result(0);
      for (std::size_t item = 0; item < modular_.size(); ++item)
      {
        if (subset[item])
          result += Scalar(modular_[item]);
      }
      for (const auto &feature : features_)
      {
        bool covered = false;
        for (const auto item : feature.items)
        {
          covered = covered || subset[item];
        }
        if (covered)
          result += Scalar(feature.weight);
      }
      return result;
    }

    std::string name() const override { return name_; }

    const std::vector<std::int64_t> &modular() const
    {
      return modular_;
    }
    const std::vector<CoverageFeature> &features() const
    {
      return features_;
    }

  private:
    std::vector<std::int64_t> modular_;
    std::vector<CoverageFeature> features_;
    std::string name_;
  };

  // Uniform matroid rank functions are submodular
  // given a vector of non-negative weights and a rank r,
  // the value of a subset S is the sum of the weights of the r largest elements in S (or all of S if |S|<r),
  // plus a non-negative weight for each element in S.

  class UniformMatroidOracle final : public SubmodularOracle
  {
  public:
    UniformMatroidOracle(std::vector<std::int64_t> modular, std::size_t rank,
                         std::int64_t rank_weight,
                         std::string name = "uniform_matroid_rank")
        : modular_(std::move(modular)),
          rank_(rank),
          rank_weight_(rank_weight),
          name_(std::move(name))
    {
      if (rank_weight_ < 0)
      {
        throw std::invalid_argument("matroid-rank weight must be non-negative");
      }
    }

    std::size_t size() const override { return modular_.size(); }

    Scalar evaluate(const Subset &subset) const override
    {
      std::size_t cardinality = 0;
      Scalar result(0);
      for (std::size_t i = 0; i < modular_.size(); ++i)
      {
        if (subset[i])
        {
          ++cardinality;
          result += Scalar(modular_[i]);
        }
      }
      result += Scalar(rank_weight_) *
                Scalar(static_cast<std::int64_t>(std::min(rank_, cardinality)));
      return result;
    }

    std::string name() const override { return name_; }

  private:
    std::vector<std::int64_t> modular_;
    std::size_t rank_;
    std::int64_t rank_weight_;
    std::string name_;
  };

  inline std::vector<std::string> fixture_names()
  {
    return {"modular", "concave_cardinality", "directed_cut", "coverage",
            "uniform_matroid_rank"};
  }

  inline std::unique_ptr<SubmodularOracle> make_fixture(const std::string &name)
  {
    if (name == "modular")
    {
      // Nonzero f(empty) deliberately tests Orlin's WLOG normalization.
      return std::make_unique<ModularOracle>(
          std::vector<std::int64_t>{-8, 3, -2, 5, -7, 4}, 11, name);
    }
    if (name == "concave_cardinality")
    {
      return std::make_unique<ConcaveCardinalityOracle>(
          std::vector<std::int64_t>{0, 1, 2, 4, 7, 9, 12, 15, 100}, 3,
          name);
    }
    if (name == "directed_cut")
    {
      return std::make_unique<DirectedCutOracle>(
          std::vector<std::int64_t>{-7, 4, -3, 6, -5, 2, 1, -4},
          std::vector<Arc>{{0, 1, 5}, {1, 2, 2}, {2, 3, 4}, {3, 0, 1}, {0, 4, 3}, {4, 5, 6}, {5, 6, 2}, {6, 7, 5}, {7, 4, 1}, {2, 6, 3}, {5, 1, 4}, {7, 3, 2}, {1, 0, 1}},
          name);
    }
    if (name == "coverage")
    {
      return std::make_unique<CoverageOracle>(
          std::vector<std::int64_t>{-8, -7, -8, 5, -10, 4, 3, -8},
          std::vector<CoverageFeature>{{4, {0, 1}},
                                       {3, {1, 2, 3}},
                                       {5, {0, 3, 4}},
                                       {2, {4, 5}},
                                       {6, {2, 5, 6}},
                                       {4, {6, 7}},
                                       {3, {1, 7}},
                                       {5, {3, 5, 7}}},
          name);
    }
    if (name == "uniform_matroid_rank")
    {
      return std::make_unique<UniformMatroidOracle>(
          std::vector<std::int64_t>{-8, -7, -6, -4, 1, 2, 3}, 3, 5, name);
    }
    throw std::invalid_argument("unknown fixture: " + name);
  }

} // namespace orlin::examples
