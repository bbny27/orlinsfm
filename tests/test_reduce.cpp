#include "orlin/orlin_sfm.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace orlin {

struct OrlinTestAccess {
  static void exercise_reduce() {
    LambdaOracle oracle(
        4, [](const Subset&) { return Scalar(0); }, "reduce_test");
    OrlinSFM solver(oracle);
    constexpr std::size_t count = 14;
    Vector expected(4, Scalar(0));

    for (std::size_t index = 0; index < count; ++index) {
      OrlinSFM::Distance distance;
      distance.labels.resize(4);
      std::size_t code = index;
      for (auto& label : distance.labels) {
        label = static_cast<int>(code % 4);
        code /= 4;
      }
      Vector base{
          Scalar(static_cast<std::int64_t>((index * 3) % 11) - 5),
          Scalar(static_cast<std::int64_t>((index * 5) % 13) - 6),
          Scalar(static_cast<std::int64_t>((index * 7) % 17) - 8),
          Scalar(static_cast<std::int64_t>((index * 2) % 7) - 3)};
      const Scalar coefficient = Scalar(1) / count;
      for (std::size_t row = 0; row < 4; ++row) {
        expected[row] += coefficient * base[row];
      }
      solver.add_entry(distance, coefficient, std::move(base));
    }

    solver.reduce();
    if (solver.entries_.size() > 5) {
      throw std::runtime_error("Reduce retained an affinely dependent support");
    }
    Scalar sum(0);
    Vector reconstructed(4, Scalar(0));
    for (const auto& [distance, entry] : solver.entries_) {
      (void)distance;
      if (entry->lambda <= Scalar(0)) {
        throw std::runtime_error("Reduce retained a nonpositive coefficient");
      }
      sum += entry->lambda;
      for (std::size_t row = 0; row < 4; ++row) {
        reconstructed[row] += entry->lambda * entry->base[row];
      }
    }
    if (sum != Scalar(1) || reconstructed != expected) {
      throw std::runtime_error("Reduce changed the convex combination");
    }
  }
};

}  // namespace orlin

int main() { orlin::OrlinTestAccess::exercise_reduce(); }
