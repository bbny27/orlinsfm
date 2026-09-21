#include "orlin/orlin_sfm.hpp"

#include <stdexcept>

using namespace orlin;

void check(bool condition) {
  if (!condition) throw std::runtime_error("Boost/Eigen integration failure");
}

int main() {
  const Scalar fraction = Scalar(6) / 8;
  check(fraction.str() == "3/4");
  Scalar huge = 1;
  for (int i = 0; i < 500; ++i) huge *= 2;
  check(huge / huge == 1 && bit_size(huge) == 501);

  // Include a tiny nonzero pivot: exact rank must not apply an epsilon cutoff.
  Matrix matrix = Matrix::Zero(3, 3);
  matrix(0, 2) = 7;
  matrix(1, 0) = Scalar(1) / huge;
  matrix(2, 1) = -3;
  Eigen::FullPivLU<Matrix> lu(matrix);
  lu.setThreshold(Scalar(0));
  check(lu.rank() == 3);
  const Matrix identity = matrix * lu.inverse();
  check(identity == Matrix::Identity(3, 3));

  Matrix dependent(3, 4);
  dependent << 0, 1, 2, 0,
               0, 0, 0, 5,
               0, 3, 6, 0;
  const auto indices = detail::independent_columns(dependent);
  check(indices.size() == 2);
  Matrix selected(3, 2);
  for (Eigen::Index j = 0; j < 2; ++j) selected.col(j) = dependent.col(indices[j]);
  Eigen::FullPivLU<Matrix> selected_lu(selected);
  selected_lu.setThreshold(Scalar(0));
  check(selected_lu.rank() == 2);
}
