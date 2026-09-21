#include "orlin/incremental_rref.hpp"

#include <random>
#include <stdexcept>

namespace {

bool zero(const orlin::Column& vector) {
  return (vector.array() == orlin::Scalar(0)).all();
}

}  // namespace

int main() {
  std::mt19937 random(20260918);
  std::uniform_int_distribution<int> entry(-3, 3);
  for (Eigen::Index size = 1; size <= 5; ++size) {
    orlin::Matrix matrix =
        orlin::Matrix::Zero(size, size);
    for (Eigen::Index row = 0; row < size; ++row) {
      for (Eigen::Index column = 0; column < size; ++column) {
        matrix(row, column) = orlin::Scalar(entry(random));
      }
    }
    orlin::detail::IncrementalRref incremental;
    incremental.reset(matrix);

    for (int update = 0; update < 100; ++update) {
      const Eigen::Index changed = update % size;
      for (Eigen::Index row = 0; row < size; ++row) {
        matrix(row, changed) = orlin::Scalar(entry(random));
      }
      incremental.synchronize(matrix);

      Eigen::FullPivLU<orlin::Matrix> fresh(matrix);
      fresh.setThreshold(orlin::Scalar(0));
      if (incremental.rank() != static_cast<std::size_t>(fresh.rank())) {
        throw std::runtime_error("incremental RREF rank mismatch");
      }
      if (incremental.singular()) {
        const auto vector = incremental.null_vector();
        if (zero(vector) || !zero(matrix * vector)) {
          throw std::runtime_error("incremental RREF null-vector mismatch");
        }
      } else {
        orlin::Column right_hand_side =
            orlin::Column::Zero(size);
        for (Eigen::Index row = 0; row < size; ++row) {
          right_hand_side[row] = orlin::Scalar(entry(random));
        }
        const auto solution = incremental.solve(right_hand_side);
        if (!zero(matrix * solution - right_hand_side)) {
          throw std::runtime_error("incremental RREF solve mismatch");
        }
      }
    }
  }

  // Rank loss, rank gain, unchanged input, and dimension changes.
  orlin::detail::IncrementalRref incremental;
  for (Eigen::Index size : {0, 1, 6, 2}) {
    orlin::Matrix matrix = orlin::Matrix::Zero(size, size);
    incremental.synchronize(matrix);
    if (incremental.rank() != 0 || incremental.synchronize(matrix)) {
      throw std::runtime_error("zero or unchanged matrix mismatch");
    }
    for (Eigen::Index i = 0; i < size; ++i) {
      matrix(i, i) = 1;
      incremental.synchronize(matrix);
      if (incremental.rank() != static_cast<std::size_t>(i + 1)) {
        throw std::runtime_error("rank gain mismatch");
      }
    }
    for (Eigen::Index i = 0; i < size; ++i) {
      matrix.col(i).setZero();
      incremental.synchronize(matrix);
      if (incremental.rank() != static_cast<std::size_t>(size - i - 1)) {
        throw std::runtime_error("rank loss mismatch");
      }
    }
  }
}
