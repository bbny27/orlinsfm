#pragma once

#include "orlin/types.hpp"
#include "orlin/incremental_rref.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace orlin
{

  class SubmodularOracle
  {
  public:
    virtual ~SubmodularOracle() = default;                   // destructor
    virtual std::size_t size() const = 0;                    // number of ground-set elements
    virtual Scalar evaluate(const Subset &subset) const = 0; // return function value
    virtual std::string name() const { return "unnamed-oracle"; }
  };

  class LambdaOracle final : public SubmodularOracle
  {
  public:
    using Function = std::function<Scalar(const Subset &)>;

    LambdaOracle(std::size_t size, Function function, std::string name)
        : size_(size), function_(std::move(function)), name_(std::move(name))
    {
      if (!function_)
      {
        throw std::invalid_argument("LambdaOracle needs a callable");
      }
    }

    std::size_t size() const override { return size_; }

    Scalar evaluate(const Subset &subset) const override
    {
      if (subset.size() != size_)
      {
        throw std::invalid_argument("oracle subset has the wrong size");
      }
      return function_(subset);
    }

    std::string name() const override { return name_; }

  private:
    std::size_t size_;
    Function function_;
    std::string name_;
  };

  struct SolverOptions
  {
    // Full invariant checking is intentionally opt-in: it makes additional
    // oracle calls and is not part of Orlin's stated complexity bound.
    bool verify_invariants = false;
    bool trace_iterations = false;
    std::ostream *trace_stream = nullptr;
    std::uint64_t iteration_limit = 10'000'000;
    // Used only by the double backend. The exact backend ignores both values.
    double absolute_tolerance = 1e-10;
    double relative_tolerance = 1e-12;
  };

  struct SolverStatistics
  {
    std::uint64_t iterations = 0;              // number of iterations performed
    std::uint64_t oracle_calls = 0;            // number of oracle calls
    std::uint64_t validation_oracle_calls = 0; // number of oracle calls for validation
    std::uint64_t greedy_calls = 0;            // number of greedy base calls
    std::uint64_t auxiliary_systems = 0;       // number of auxiliary systems constructed
    std::uint64_t auxiliary_rebuilds = 0;
    std::uint64_t auxiliary_column_updates = 0;
    std::uint64_t singular_auxiliary_systems = 0;    // number of singular auxiliary systems constructed
    std::uint64_t nonsingular_auxiliary_systems = 0; // number of nonsingular auxiliary systems constructed
    std::uint64_t reduce_calls = 0;                  //  number of calls to reduce()
    std::uint64_t distance_gap_events = 0;           // number of distance gap events
    std::size_t maximum_active_elements = 0;         // maximum number of active elements
    std::size_t maximum_distance_functions = 0;      // maximum number of distance functions
    std::size_t maximum_rational_bits = 0;           // maximum number of bits in rational numbers
    std::chrono::nanoseconds elapsed{};
  };

  template <class Number>
  struct SolverResultT
  {
    std::vector<std::size_t> minimizer;
    ExactScalar minimum_value; // Re-evaluated by the exact input oracle.
    VectorT<Number> final_base;
    SolverStatistics statistics;
  };

  using SolverResult = SolverResultT<ExactScalar>;
  using DoubleSolverResult = SolverResultT<double>;

  // linear algebra
  namespace detail
  {

    inline void require(bool condition, const std::string &message)
    {
      if (!condition)
      {
        throw std::logic_error("Orlin invariant failure: " + message);
      }
    }

    template <class Number>
    VectorT<Number> subtract(const VectorT<Number> &lhs,
                             const VectorT<Number> &rhs)
    {
      require(lhs.size() == rhs.size(), "vector dimension mismatch");
      VectorT<Number> result(lhs.size(), Number(0));
      for (std::size_t i = 0; i < lhs.size(); ++i)
      {
        result[i] = lhs[i] - rhs[i];
      }
      return result;
    }

    template <class Number>
    Number dot_sum(const VectorT<Number> &vector,
                   const std::vector<std::size_t> &indices)
    {
      Number result(0);
      for (const auto index : indices)
      {
        result += vector[index];
      }
      return result;
    }

    template <class Number>
    std::vector<Eigen::Index> independent_columns(
        const MatrixT<Number> &matrix,
        const Numerics<Number> &numerics = Numerics<Number>())
    {
      if (matrix.size() == 0)
        return {};
      Eigen::FullPivLU<MatrixT<Number>> lu(matrix);
      lu.setThreshold(numerics.lu_threshold());
      std::vector<Eigen::Index> result(static_cast<std::size_t>(lu.rank()));
      for (Eigen::Index i = 0; i < lu.rank(); ++i)
      {
        result[static_cast<std::size_t>(i)] = lu.permutationQ().indices()[i];
      }
      return result;
    }

  } // namespace detail

  // function starts here!
  template <class Number>
  class BasicOrlinSFM
  {
  public:
    using Scalar = Number;
    using Vector = VectorT<Number>;
    using Matrix = MatrixT<Number>;
    using Column = ColumnT<Number>;
    using Result = SolverResultT<Number>;

    explicit BasicOrlinSFM(const SubmodularOracle &oracle,
                           SolverOptions options = {})
        : oracle_(oracle),
          options_(options),
          numerics_(options.absolute_tolerance, options.relative_tolerance),
          n_(oracle.size()),
          active_(n_, 1), // mark everyone active at first
          x_(n_, Scalar(0)),
          heaps_(n_),
          column_cache_(n_),
          auxiliary_system_(numerics_)
    {
      if (options_.trace_iterations && options_.trace_stream == nullptr)
      {
        options_.trace_stream = &std::cerr;
      }
      statistics_.maximum_active_elements = n_;
    }

    Result minimize()
    {
      using Clock = std::chrono::steady_clock;
      const auto start = Clock::now();

      Subset empty(n_, 0);                  // empty subset
      offset_ = evaluate_raw(empty, false); // evaluate f(empty)

      Distance initial;
      initial.labels.assign(n_, 0); // assign every element a distance label of 0
      Vector initial_base = greedy_base(initial);
      x_ = initial_base;
      add_entry(initial, Scalar(1), std::move(initial_base)); // insert d into D with lambda_d = 1

      update_sign_sets(); // initialize V^+, V^0
      if (options_.verify_invariants)
        update_peak_rational_size(); // initialize peak rational size
      if (options_.verify_invariants)
      {
        verify_invariants("initialization");
      }

      while (!positive_.empty())
      {
        if (statistics_.iterations >= options_.iteration_limit)
        { // guard against infinite loops
          throw std::runtime_error(
              "iteration guard reached; input may not be submodular or an "
              "implementation invariant has been violated");
        }
        ++statistics_.iterations;

        const auto primary = update_primaries(); // Choose the primary distance function p(v) for every active vertex
        std::vector<Distance> secondary(n_);     // allocate space for secondary distance functions s(v)
        for (const auto vertex : active_indices())
        {
          secondary[vertex] = primary[vertex]->distance; // initialize s(v) = p(v)
          if (secondary[vertex].labels[vertex] >=        // guard against distance label overflow; bounded by n+1
              static_cast<int>(n_ + 1U))
          {
            throw std::logic_error("distance label exceeded Orlin's n+1 bound");
          }
          ++secondary[vertex].labels[vertex]; // Implement s(v)=INC(p(v),v) by increasing just coordinate v
        }

        const std::size_t v_star = positive_.front(); // FREE: Choose the smallest-index current positive vertex as v*
        std::vector<std::size_t> relevant = zero_;    // start relevant with the current zero set V^0
        relevant.push_back(v_star);                   // add v* to the relevant set

        std::vector<Vector> differences(n_);
        for (const auto vertex : relevant)
        {
          const auto &column = auxiliary_column(vertex, primary[vertex], secondary[vertex]); // Reuse a cached column or compute its secondary greedy base
          differences[vertex] = column.difference;                                           // Store the full n-coordinate difference column under v
        }

        Vector gamma(n_, Scalar(0));
        if (zero_.empty())
        {
          gamma[v_star] = Scalar(1); // Section 6, Case 1.
        }
        else
        {
          gamma = choose_gamma(zero_, v_star, differences);
        }

        Vector direction(n_, Scalar(0));
        for (const auto vertex : relevant)
        {
          const Scalar &coefficient = gamma[vertex];
          if (numerics_.zero(coefficient))
          {
            continue;
          }
          const auto &difference = differences.at(vertex); // obtain the difference column for this vertex
          for (const auto coordinate : active_indices())
          {                                                                // update every coordinate
            direction[coordinate] += coefficient * difference[coordinate]; // add gamma(v)[y_s(v)-y_p(v)] to each active coordinate of x'
          }
        }
        for (const auto vertex : zero_)
        { // verify that the direction is zero in all coordinates of V^0
          detail::require(numerics_.zero(direction[vertex]),
                          "chosen gamma does not preserve V^0");
        }

        const Scalar alpha = maximum_step(direction, gamma, primary);
        detail::require(numerics_.positive(alpha),
                        "alpha must be strictly positive");

        for (const auto vertex : active_indices())
        {
          x_[vertex] = numerics_.clean(
              x_[vertex] + alpha * direction[vertex]);
        }

        //  Entry: object represening one greedy base + lambda for each distance function in D
        {
          if (numerics_.zero(gamma[vertex]))
            continue;
          const Scalar amount = alpha * gamma[vertex];
          auto &cached = *column_cache_[vertex]; // Previous information in matrix
          if (!cached.secondary_entry || !cached.secondary_entry->alive)
          {
            const auto found = entries_.find(cached.secondary); 
            cached.secondary_entry = found != entries_.end() 
            ? found->second 
            : add_entry(cached.secondary, Scalar(0), cached.secondary_base); // New base for the secondary distance function
          }
          delta[primary[vertex]] -= amount;
          delta[cached.secondary_entry] += amount;
        }
        for (const auto &[entry, amount] : delta)
        {
          entry->lambda = numerics_.clean(entry->lambda + amount);
        }
        erase_zero_entries();

        statistics_.maximum_distance_functions = std::max(statistics_.maximum_distance_functions, entries_.size());
        if (n_ != 0 && entries_.size() >= 3U * n_)
        {
          reduce();
        }

        if (const auto gap = find_distance_gap(); gap.has_value())
        {
          eliminate_at_gap(*gap);
        }

        update_sign_sets(); // Refresh V^0 and V^+ after movement and possible elimination.
        if (options_.verify_invariants)
          update_peak_rational_size();
        if (options_.trace_iterations)
        {
          trace_iteration(alpha, v_star);
        }
        if (options_.verify_invariants)
        {
          verify_invariants("iteration " +
                            std::to_string(statistics_.iterations));
        }
      }

      Subset minimizing_subset = active_;
      const ExactScalar minimum_value = evaluate_exact(minimizing_subset, false);
      statistics_.elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);

      Result result;
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      { // indices of the final minimizer are the active vertices
        if (active_[vertex])
        {
          result.minimizer.push_back(vertex);
        }
      }
      result.minimum_value = minimum_value;
      result.final_base = x_;
      result.statistics = statistics_;
      return result;
    }

  private:
    friend struct OrlinTestAccess;

    struct Distance
    {
      std::vector<int> labels;

      friend bool operator<(const Distance &lhs, const Distance &rhs)
      {
        return lhs.labels < rhs.labels;
      }
      friend bool operator==(const Distance &lhs, const Distance &rhs)
      {
        return lhs.labels == rhs.labels;
      }
    };

    struct Entry
    { // entry associated with a distance function d in D
      std::uint64_t id = 0;
      Distance distance;
      Scalar lambda;
      Vector base;
      std::int64_t label_sum = 0;
      bool alive = true;
    };

    struct HeapItem
    {
      int label = 0;
      std::int64_t label_sum = 0;
      std::uint64_t id = 0; // FREE: unique insertion id to break ties in the heap
      std::shared_ptr<Entry> entry;
    };

    struct HeapGreater
    {
      bool operator()(const HeapItem &lhs, const HeapItem &rhs) const
      {
        return std::tie(lhs.label, lhs.label_sum, lhs.id) >
               std::tie(rhs.label, rhs.label_sum, rhs.id);
      }
    };

    using Heap =
        std::priority_queue<HeapItem, std::vector<HeapItem>, HeapGreater>;

    struct CachedColumn
    {
      Distance primary;
      Distance secondary;
      Vector secondary_base;
      Vector difference;
      std::shared_ptr<Entry> secondary_entry;
    };

    const SubmodularOracle &oracle_;
    SolverOptions options_;
    Numerics<Number> numerics_;
    std::size_t n_;
    Subset active_;
    Vector x_;
    Scalar offset_;
    std::vector<std::size_t> zero_;
    std::vector<std::size_t> positive_;
    std::map<Distance, std::shared_ptr<Entry>> entries_;
    std::vector<Heap> heaps_;
    std::vector<std::optional<CachedColumn>> column_cache_;
    detail::IncrementalRrefT<Number> auxiliary_system_;
    std::uint64_t next_entry_id_ = 1;
    SolverStatistics statistics_;

    // Greedy base and auxiliary column computations

    std::vector<std::size_t> active_indices() const
    {
      std::vector<std::size_t> result;
      result.reserve(n_);
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      { // indices of the active vertices are those with active_[vertex] == 1
        if (active_[vertex])
        {
          result.push_back(vertex);
        }
      }
      return result;
    }

    bool vector_equal(const Vector &first, const Vector &second) const
    {
      if (first.size() != second.size())
        return false;
      for (std::size_t i = 0; i < first.size(); ++i)
      {
        if (!numerics_.equal(first[i], second[i]))
          return false;
      }
      return true;
    }

    bool column_is_zero(const Column &column) const
    {
      Scalar scale(1);
      for (Eigen::Index i = 0; i < column.size(); ++i)
      {
        scale = std::max(scale, numerics_.magnitude(column[i]));
      }
      for (Eigen::Index i = 0; i < column.size(); ++i)
      {
        if (!numerics_.zero(column[i], scale))
          return false;
      }
      return true;
    }

    ExactScalar evaluate_exact(const Subset &subset, bool validation)
    {
      if (subset.size() != n_)
      {
        throw std::invalid_argument("oracle subset has the wrong dimension");
      }
      ++statistics_.oracle_calls;
      if (validation)
      {
        ++statistics_.validation_oracle_calls;
      }
      return oracle_.evaluate(subset);
    }

    Scalar evaluate_raw(const Subset &subset, bool validation)
    {
      return numerics_.from_exact(evaluate_exact(subset, validation));
    }

    Scalar evaluate_normalized(const Subset &subset, // Return G(S)=F(S)-offset_
                               bool validation)
    {
      return evaluate_raw(subset, validation) - offset_;
    }

    std::int64_t label_sum(const Distance &distance) const
    {
      std::int64_t result = 0;
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      {
        if (active_[vertex])
        {
          result += distance.labels[vertex];
        }
      }
      return result;
    }

    Vector greedy_base(const Distance &distance)
    {
      ++statistics_.greedy_calls;
      std::vector<std::size_t> order = active_indices();                            // indices of the active vertices
      std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) { // FREE: sort the active vertices by (d(v),v) lexicographically
        return std::tie(distance.labels[lhs], lhs) <
               std::tie(distance.labels[rhs], rhs);
      });

      Subset prefix(n_, 0);
      Vector base(n_, Scalar(0));
      Scalar previous(0);
      for (const auto vertex : order)
      {
        prefix[vertex] = 1;
        const Scalar current = evaluate_normalized(prefix, false);
        base[vertex] = current - previous;
        previous = current;
      }
      return base;
    }

    // Maintaining D and choosing primary orders

    std::shared_ptr<Entry> add_entry(const Distance &distance, Scalar lambda,
                                     Vector base)
    { // insert new distance function d into D with coefficient lambda_d and greedy base y_d
      detail::require(entries_.find(distance) == entries_.end(),
                      "attempt to add duplicate distance function");
      auto entry = std::make_shared<Entry>();
      entry->id = next_entry_id_++;
      entry->distance = distance;
      entry->lambda = std::move(lambda);
      entry->base = std::move(base);
      entry->label_sum = label_sum(entry->distance);
      entry->alive = true;
      entries_.emplace(entry->distance, entry);
      add_to_heaps(entry);
      statistics_.maximum_distance_functions =
          std::max(statistics_.maximum_distance_functions, entries_.size());
      return entry;
    }

    void add_to_heaps(const std::shared_ptr<Entry> &entry)
    {
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      {
        if (!active_[vertex])
        {
          continue; // skip inactive vertices
        }
        heaps_[vertex].push(HeapItem{entry->distance.labels[vertex],
                                     entry->label_sum, entry->id, entry});
      }
    }

    void clean_heap(std::size_t vertex)
    { // delete dead entries from the top of the heap for a given vertex
      while (!heaps_[vertex].empty() && !heaps_[vertex].top().entry->alive)
      {
        heaps_[vertex].pop();
      }
      detail::require(!heaps_[vertex].empty(), // an active vertex must have at least one live distance function in the heap
                      "no distance function remains for an active element");
    }

    std::vector<std::shared_ptr<Entry>> update_primaries()
    {
      // The heap order (d(v), d(V), insertion id) is equivalent to first forming
      // Q(v)={d:d(v)=D_min(v)} and then applying FindMin with key d(V), exactly
      // as in Orlin's Update(D,p,s).
      std::vector<std::shared_ptr<Entry>> primary(n_);
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      {
        if (!active_[vertex])
        {
          continue;
        }
        clean_heap(vertex);
        primary[vertex] = heaps_[vertex].top().entry;
      }
      return primary;
    }

    // As changing only lambda does not change the greedy base, we can cache the greedy base for each distance function and reuse it
    const Vector &base_for_distance(const Distance &distance,
                                    std::map<Distance, Vector> &cache)
    {
      if (const auto found = entries_.find(distance); found != entries_.end())
      {
        return found->second->base;
      }
      if (const auto found = cache.find(distance); found != cache.end())
      {
        return found->second;
      }
      auto [position, inserted] = cache.emplace(distance, greedy_base(distance));
      detail::require(inserted, "candidate base cache insertion failed");
      return position->second;
    }

    const CachedColumn &auxiliary_column(
        std::size_t vertex, const std::shared_ptr<Entry> &primary,
        const Distance &secondary)
    {
      auto &cached = column_cache_[vertex];
      if (cached.has_value() && cached->primary == primary->distance &&
          cached->secondary == secondary)
      {
        return *cached;
      }
      std::map<Distance, Vector> temporary;
      const Vector secondary_base = base_for_distance(secondary, temporary);
      cached = CachedColumn{primary->distance, secondary, secondary_base,
                            detail::subtract(secondary_base, primary->base), nullptr};
      return *cached;
    }

    // Selecting gamma and taking a step
    Vector choose_gamma(
        const std::vector<std::size_t> &zero_vertices, std::size_t v_star,
        const std::vector<Vector> &differences)
    {
      ++statistics_.auxiliary_systems;
      const std::size_t size = zero_vertices.size();
      Matrix auxiliary = Matrix::Zero(
          static_cast<Eigen::Index>(size), static_cast<Eigen::Index>(size));
      for (std::size_t row = 0; row < size; ++row)
      {
        for (std::size_t column = 0; column < size; ++column)
        {
          auxiliary(static_cast<Eigen::Index>(row),
                    static_cast<Eigen::Index>(column)) =
              differences.at(zero_vertices[column])[zero_vertices[row]];
        }
      }

      // Verify properties 6.2--6.4 exactly for rational arithmetic and within
      // the configured tolerance for floating-point arithmetic.
      for (std::size_t column = 0; column < size; ++column)
      {
        Scalar column_sum(0);
        for (std::size_t row = 0; row < size; ++row)
        {
          const Scalar value = auxiliary(static_cast<Eigen::Index>(row),
                                         static_cast<Eigen::Index>(column));
          if (row == column)
          {
            detail::require(numerics_.nonpositive(value),
                            "auxiliary diagonal is not non-positive");
          }
          else
          {
            detail::require(numerics_.nonnegative(value),
                            "auxiliary off-diagonal is not non-negative");
          }
          column_sum += value;
        }
        detail::require(numerics_.nonpositive(column_sum),
                        "auxiliary column sum is not non-positive");
      }

      const bool rebuilt = auxiliary_system_.dimension() !=
                           static_cast<Eigen::Index>(size);
      const auto changed_columns = auxiliary_system_.synchronize(auxiliary);
      statistics_.auxiliary_rebuilds += rebuilt;
      statistics_.auxiliary_column_updates += changed_columns;

      Vector gamma(n_, Scalar(0));
      if (auxiliary_system_.singular())
      {
        // Section 6, Case 2 and Theorem 2.
        ++statistics_.singular_auxiliary_systems;
        const Column null_vector = auxiliary_system_.null_vector();
        Column nonnegative = Column::Zero(
            static_cast<Eigen::Index>(size));
        for (std::size_t i = 0; i < size; ++i)
        {
          if (numerics_.positive(
                  null_vector[static_cast<Eigen::Index>(i)]))
          {
            nonnegative[static_cast<Eigen::Index>(i)] =
                null_vector[static_cast<Eigen::Index>(i)];
          }
        }
        if (column_is_zero(nonnegative))
        {
          for (std::size_t i = 0; i < size; ++i)
          {
            if (numerics_.negative(
                    null_vector[static_cast<Eigen::Index>(i)]))
            {
              nonnegative[static_cast<Eigen::Index>(i)] =
                  -null_vector[static_cast<Eigen::Index>(i)];
            }
          }
        }
        detail::require(!column_is_zero(nonnegative),
                        "failed to construct nonzero nonnegative null vector");
        const Column residual = auxiliary * nonnegative;
        detail::require(column_is_zero(residual),
                        "Theorem 2 positive-part null vector check failed");
        for (std::size_t i = 0; i < size; ++i)
        {
          detail::require(numerics_.nonnegative(
                              nonnegative[static_cast<Eigen::Index>(i)]),
                          "negative Case 2 gamma coefficient");
          gamma[zero_vertices[i]] = numerics_.clean(
              nonnegative[static_cast<Eigen::Index>(i)]);
        }
        gamma[v_star] = Scalar(0);
      }
      else
      {
        // Section 6, Case 3: A* gamma = y_p(v*) - y_s(v*), gamma(v*)=1.
        ++statistics_.nonsingular_auxiliary_systems;
        Column right_hand_side = Column::Zero(
            static_cast<Eigen::Index>(size));
        for (std::size_t row = 0; row < size; ++row)
        {
          right_hand_side[static_cast<Eigen::Index>(row)] =
              -differences.at(v_star)[zero_vertices[row]];
          detail::require(
              numerics_.nonpositive(
                  right_hand_side[static_cast<Eigen::Index>(row)]),
              "Case 3 right-hand side must be non-positive");
        }
        const Column solution =
            auxiliary_system_.solve(right_hand_side);
        for (std::size_t column = 0; column < size; ++column)
        {
          const Scalar value = numerics_.clean(
              solution[static_cast<Eigen::Index>(column)]);
          detail::require(numerics_.nonnegative(value),
                          "M-matrix theorem produced a negative gamma");
          gamma[zero_vertices[column]] = value;
        }
        gamma[v_star] = Scalar(1);
      }
      return gamma;
    }

    // find maximum step size alpha such that x + alpha * direction remains feasible and all coefficients remain non-negative
    Scalar maximum_step(
        const Vector &direction, const Vector &gamma,
        const std::vector<std::shared_ptr<Entry>> &primary) const
    {
      std::optional<Scalar> alpha;
      for (const auto vertex : positive_)
      {
        if (numerics_.negative(direction[vertex]))
        {
          const Scalar bound = x_[vertex] / (-direction[vertex]);
          if (!alpha.has_value() || bound < *alpha)
          {
            alpha = bound;
          }
        }
      }

      std::map<std::shared_ptr<Entry>, Scalar> withdrawal;
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      {
        if (!numerics_.zero(gamma[vertex]))
        {
          withdrawal[primary[vertex]] += gamma[vertex];
        }
      }
      for (const auto &[entry, total] : withdrawal)
      {
        const Scalar bound = entry->lambda / total;
        if (!alpha || bound < *alpha)
          alpha = bound;
      }
      detail::require(alpha.has_value(), "maximum alpha is unbounded");
      return *alpha;
    }

    void erase_zero_entries()
    {
      for (auto iterator = entries_.begin(); iterator != entries_.end();)
      {
        iterator->second->lambda = numerics_.clean(iterator->second->lambda);
        detail::require(numerics_.nonnegative(iterator->second->lambda),
                        "negative lambda while pruning D");
        if (numerics_.zero(iterator->second->lambda))
        {
          iterator->second->alive = false;
          iterator = entries_.erase(iterator);
        }
        else
        {
          ++iterator;
        }
      }
      detail::require(!entries_.empty(), "D became empty");
      // Amortized rebuild after O(n) deletions: bounds live + stale heap storage.
      if (std::any_of(heaps_.begin(), heaps_.end(), [this](const Heap &heap)
                      { return heap.size() > 8U * std::max(std::size_t(1), n_); }))
      {
        heaps_ = std::vector<Heap>(n_);
        for (const auto &pair : entries_)
          add_to_heaps(pair.second);
      }
    }

    void reduce()
    {
      // Required on p.243: preserve x and sum(lambda)=1 while returning
      // an affinely independent support.  This is a basic-feasible-solution
      // reduction.  One initial inverse plus O(n) exact pivots gives O(n^3)
      // arithmetic operations because |D|<4n at a call.
      ++statistics_.reduce_calls;
      std::vector<std::shared_ptr<Entry>> columns;
      columns.reserve(entries_.size());
      for (const auto &[distance, entry] : entries_)
      {
        (void)distance;
        columns.push_back(entry);
      }
      const std::size_t count = columns.size();
      const auto active_vertices = active_indices();

      Matrix augmented(active_vertices.size() + 1U, count);
      for (std::size_t column = 0; column < count; ++column)
      {
        for (std::size_t row = 0; row < active_vertices.size(); ++row)
        {
          augmented(row, column) = columns[column]->base[active_vertices[row]];
        }
        augmented(augmented.rows() - 1, column) = 1;
      }

      const Matrix transposed_augmented = augmented.transpose();
      const auto independent_rows =
          detail::independent_columns(transposed_augmented, numerics_);
      const std::size_t rank = independent_rows.size();
      detail::require(rank > 0, "affine matrix unexpectedly has rank zero");
      Matrix full_row_rank(rank, count);
      for (std::size_t row = 0; row < rank; ++row)
      {
        full_row_rank.row(row) = augmented.row(independent_rows[row]);
      }
      auto basis =
          detail::independent_columns(full_row_rank, numerics_);
      detail::require(basis.size() == rank, "failed to select affine basis");

      Matrix basis_matrix(rank, rank);
      for (std::size_t column = 0; column < rank; ++column)
      {
        basis_matrix.col(column) = full_row_rank.col(basis[column]);
      }
      Eigen::FullPivLU<Matrix> lu(basis_matrix);
      lu.setThreshold(numerics_.lu_threshold());
      detail::require(lu.isInvertible(), "singular affine basis");
      Matrix basis_inverse = lu.inverse();
      std::vector<bool> in_basis(count, false);
      for (const auto index : basis)
      {
        in_basis[index] = true;
      }
      Vector lambda(count, Scalar(0));
      for (std::size_t i = 0; i < count; ++i)
      {
        lambda[i] = columns[i]->lambda;
      }

      while (true)
      {
        std::size_t entering = count;
        for (std::size_t i = 0; i < count; ++i)
        {
          if (!in_basis[i] && numerics_.positive(lambda[i]))
          {
            entering = i;
            break;
          }
        }
        if (entering == count)
        {
          break;
        }

        const Column coordinates = basis_inverse * full_row_rank.col(entering);

        // Null direction z_entering=1, z_basis=-B^{-1}a_entering.
        Scalar theta = lambda[entering];
        for (std::size_t position = 0; position < rank; ++position)
        {
          const Scalar z = -coordinates[position];
          if (numerics_.positive(z))
          {
            const Scalar bound = lambda[basis[position]] / z;
            if (bound < theta)
            {
              theta = bound;
            }
          }
        }
        // Degenerate basic coefficients can make theta zero.  The ensuing
        // zero-length basis exchange is still required and removes one positive
        // nonbasic variable from consideration.
        detail::require(numerics_.nonnegative(theta),
                        "Reduce pivot has negative step");
        lambda[entering] = numerics_.clean(lambda[entering] - theta);
        for (std::size_t position = 0; position < rank; ++position)
        {
          lambda[basis[position]] = numerics_.clean(
              lambda[basis[position]] + theta * coordinates[position]);
          detail::require(numerics_.nonnegative(lambda[basis[position]]),
                          "Reduce produced negative lambda");
        }

        if (numerics_.positive(lambda[entering]))
        {
          std::size_t leaving_position = rank;
          for (std::size_t position = 0; position < rank; ++position)
          {
            if (numerics_.negative(coordinates[position]) &&
                numerics_.zero(lambda[basis[position]]))
            {
              leaving_position = position;
              break;
            }
          }
          detail::require(leaving_position < rank,
                          "Reduce ratio test found no leaving column");
          detail::require(!numerics_.zero(coordinates[leaving_position]),
                          "Reduce pivot coefficient is zero");

          // If B'=B E replaces basis column l by B*c, then
          // B'^{-1}=E^{-1}B^{-1}.  Apply E^{-1} by row operations.
          basis_inverse.row(leaving_position) /= coordinates[leaving_position];
          for (std::size_t row = 0; row < rank; ++row)
          {
            if (row != leaving_position)
            {
              basis_inverse.row(row) -=
                  coordinates[row] * basis_inverse.row(leaving_position);
            }
          }
          in_basis[basis[leaving_position]] = false;
          basis[leaving_position] = entering;
          in_basis[entering] = true;
        }
      }

      for (std::size_t i = 0; i < count; ++i)
      {
        columns[i]->lambda = lambda[i];
      }
      erase_zero_entries();

      Matrix surviving(active_vertices.size() + 1U, entries_.size());
      std::size_t column = 0;
      for (const auto &[distance, entry] : entries_)
      {
        (void)distance;
        for (std::size_t row = 0; row < active_vertices.size(); ++row)
        {
          surviving(row, column) = entry->base[active_vertices[row]];
        }
        surviving(surviving.rows() - 1, column) = 1;
        ++column;
      }
      detail::require(detail::independent_columns(surviving, numerics_).size() ==
                          entries_.size(),
                      "Reduce output is not affinely independent");
    }

    // Distance gap elimination
    std::optional<int> find_distance_gap()
    {
      if (entries_.empty())
      {
        return std::nullopt;
      }
      std::vector<bool> occupied(n_ + 1U, false);
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      {
        if (!active_[vertex])
        {
          continue;
        }
        clean_heap(vertex);
        const int minimum = heaps_[vertex].top().label; // D_min(v) for this vertex
        detail::require(minimum >= 0 && minimum <= static_cast<int>(n_),
                        "D_min exceeded n before a distance gap was processed");
        occupied[static_cast<std::size_t>(minimum)] = true;
      }
      // The paragraph following Lemma 2 permits the boundary level k=n.
      for (std::size_t level = 1; level <= n_; ++level)
      {
        if (occupied[level] && !occupied[level - 1U])
        {
          return static_cast<int>(level);
        }
      }
      return std::nullopt;
    }

    void eliminate_at_gap(int level)
    {
      ++statistics_.distance_gap_events;
      std::vector<std::size_t> eliminated;
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      {
        if (!active_[vertex])
        {
          continue;
        }
        clean_heap(vertex);
        if (heaps_[vertex].top().label >= level)
        {
          eliminated.push_back(vertex);
        }
      }
      detail::require(!eliminated.empty(), "distance gap eliminated no element");
      for (const auto vertex : eliminated)
      {
        active_[vertex] = 0;
        x_[vertex] = Scalar(0);
      }
      // A cached secondary order need not put the newly retained set before the
      // eliminated set: it may have been formed from a primary that left D in
      // this same iteration.  Recompute columns on the restricted ground set.
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      {
        column_cache_[vertex].reset();
      }

      // Restrict every distance function and extreme base to the new ground set.
      // Equal restrictions are merged, preserving their total lambda.
      std::vector<std::shared_ptr<Entry>> old_entries;
      old_entries.reserve(entries_.size());
      for (const auto &[distance, entry] : entries_)
      {
        (void)distance;
        old_entries.push_back(entry);
        entry->alive = false;
      }
      entries_.clear();
      heaps_ = std::vector<Heap>(n_);

      for (const auto &old : old_entries)
      {
        Distance restricted_distance = old->distance;
        Vector restricted_base = old->base;
        for (std::size_t vertex = 0; vertex < n_; ++vertex)
        {
          if (!active_[vertex])
          {
            restricted_distance.labels[vertex] = 0;
            restricted_base[vertex] = Scalar(0);
          }
        }
        const auto found = entries_.find(restricted_distance);
        if (found != entries_.end())
        {
          detail::require(vector_equal(found->second->base, restricted_base),
                          "equal restricted orders yielded unequal bases");
          found->second->lambda = numerics_.clean(
              found->second->lambda + old->lambda);
        }
        else
        {
          auto entry = std::make_shared<Entry>();
          entry->id = next_entry_id_++;
          entry->distance = std::move(restricted_distance);
          entry->lambda = old->lambda;
          entry->base = std::move(restricted_base);
          entry->label_sum = label_sum(entry->distance);
          entry->alive = true;
          entries_.emplace(entry->distance, entry);
        }
      }
      for (const auto &[distance, entry] : entries_)
      {
        (void)distance;
        add_to_heaps(entry);
      }
    }

    void update_sign_sets()
    {
      zero_.clear();
      positive_.clear();
      for (std::size_t vertex = 0; vertex < n_; ++vertex)
      {
        if (!active_[vertex])
        {
          continue;
        }
        x_[vertex] = numerics_.clean(x_[vertex]);
        if (numerics_.zero(x_[vertex]))
        {
          zero_.push_back(vertex);
        }
        else if (numerics_.positive(x_[vertex]))
        {
          positive_.push_back(vertex);
        }
      }
    }

    void update_peak_rational_size()
    {
      auto inspect = [&](const Scalar &value)
      {
        statistics_.maximum_rational_bits =
            std::max(statistics_.maximum_rational_bits, bit_size(value));
      };
      for (const auto &value : x_)
      {
        inspect(value);
      }
      for (const auto &[distance, entry] : entries_)
      {
        (void)distance;
        inspect(entry->lambda);
        for (const auto &value : entry->base)
        {
          inspect(value);
        }
      }
    }

    void verify_invariants(const std::string &stage)
    {
      const auto active_vertices = active_indices();
      Scalar coefficient_sum(0);
      Vector reconstructed(n_, Scalar(0));
      for (const auto &[distance, entry] : entries_)
      {
        detail::require(entry->alive, stage + ": dead entry remains in D");
        detail::require(entry->distance == distance,
                        stage + ": map key differs from stored distance");
        detail::require(numerics_.positive(entry->lambda),
                        stage + ": lambda is not positive");
        coefficient_sum += entry->lambda;
        for (std::size_t vertex = 0; vertex < n_; ++vertex)
        {
          reconstructed[vertex] += entry->lambda * entry->base[vertex];
          if (!active_[vertex])
          {
            detail::require(entry->distance.labels[vertex] == 0,
                            stage + ": inactive distance label not normalized");
            detail::require(numerics_.zero(entry->base[vertex]),
                            stage + ": inactive base coordinate not removed");
          }
        }
      }
      detail::require(numerics_.equal(coefficient_sum, Scalar(1)),
                      stage + ": convex coefficients do not sum to one");
      detail::require(vector_equal(reconstructed, x_),
                      stage + ": x differs from its convex representation");

      for (const auto vertex : active_vertices)
      {
        int minimum = std::numeric_limits<int>::max();
        for (const auto &[distance, entry] : entries_)
        {
          (void)entry;
          minimum = std::min(minimum, distance.labels[vertex]);
        }
        for (const auto &[distance, entry] : entries_)
        {
          (void)entry;
          detail::require(distance.labels[vertex] <= minimum + 1,
                          stage + ": validity condition 2 failed");
          if (numerics_.negative(x_[vertex]))
          {
            detail::require(distance.labels[vertex] == 0,
                            stage + ": validity condition 1 failed");
          }
        }
      }

      const Scalar base_total = detail::dot_sum(x_, active_vertices);
      const Scalar function_total = evaluate_normalized(active_, true);
      detail::require(numerics_.equal(base_total, function_total),
                      stage + ": x(V_in) != normalized f(V_in)");
      for (const auto vertex : zero_)
      {
        detail::require(numerics_.zero(x_[vertex]), stage + ": stale V^0");
      }
      for (const auto vertex : positive_)
      {
        detail::require(numerics_.positive(x_[vertex]),
                        stage + ": stale V^+");
      }
      if (options_.verify_invariants)
        update_peak_rational_size();
    }

    void trace_iteration(const Scalar &alpha, std::size_t v_star) const
    {
      auto &stream = *options_.trace_stream;
      stream << "iteration=" << statistics_.iterations << " v*=" << v_star
             << " alpha=" << alpha << " |V_in|=" << active_indices().size()
             << " |V0|=" << zero_.size() << " |V+|=" << positive_.size()
             << " |D|=" << entries_.size() << '\n';
    }
  };

  using OrlinSFM = BasicOrlinSFM<ExactScalar>;
  using DoubleOrlinSFM = BasicOrlinSFM<double>;

} // namespace orlin
