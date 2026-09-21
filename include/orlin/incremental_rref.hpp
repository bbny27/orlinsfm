#pragma once

#include "orlin/types.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace orlin::detail
{

  // Maintains R = T*A in pivot canonical form (pivot rows need not be sorted).  Replacing one column is a
  // rank-one update: after preserving all unaffected pivot columns, only a
  // rank-at-most-two residual block remains.  Restoring canonical form therefore
  // costs O(n^2) exact arithmetic operations, as required in Orlin, Section 6.
  template <class Number>
  class IncrementalRrefT
  {
  public:
    using Scalar = Number;
    using Matrix = MatrixT<Number>;
    using Column = ColumnT<Number>;

    explicit IncrementalRrefT(Numerics<Number> numerics = {})
        : numerics_(std::move(numerics)) {}

    std::size_t synchronize(const Matrix &matrix)
    {
      if (matrix.rows() != matrix.cols())
      {
        throw std::invalid_argument("auxiliary matrix must be square");
      }
      if (matrix.rows() != original_.rows())
      {
        reset(matrix);
        return static_cast<std::size_t>(matrix.cols());
      }
      std::size_t changed = 0;
      for (Eigen::Index column = 0; column < matrix.cols(); ++column)
      {
        if (!same_column(original_, matrix, column))
        {
          replace_column(column, matrix.col(column));
          ++changed;
        }
      }
      return changed;
    }

    void reset(const Matrix &matrix)
    {
      if (matrix.rows() != matrix.cols())
      {
        throw std::invalid_argument("auxiliary matrix must be square");
      }
      original_ = matrix;
      reduced_ = matrix;
      transform_ = Matrix::Identity(matrix.rows(), matrix.rows());
      pivot_col_by_row_.assign(static_cast<std::size_t>(matrix.rows()), -1);

      Eigen::Index pivot_row = 0;
      for (Eigen::Index column = 0;
           column < reduced_.cols() && pivot_row < reduced_.rows(); ++column)
      {
        const Eigen::Index selected = select_pivot(pivot_row, column,
                                                   reduced_.rows());
        if (selected < 0)
          continue;
        swap_rows(pivot_row, selected);
        scale_row(pivot_row, Scalar(1) / reduced_(pivot_row, column));
        for (Eigen::Index row = 0; row < reduced_.rows(); ++row)
        {
          if (row == pivot_row || numerics_.zero(reduced_(row, column)))
            continue;
          add_row(row, pivot_row, -reduced_(row, column));
        }
        pivot_col_by_row_[static_cast<std::size_t>(pivot_row)] =
            static_cast<int>(column);
        ++pivot_row;
      }
    }

    Eigen::Index dimension() const { return original_.rows(); }

    std::size_t rank() const
    {
      std::size_t result = 0;
      for (const int pivot : pivot_col_by_row_)
        result += pivot >= 0;
      return result;
    }

    bool singular() const
    {
      return rank() < static_cast<std::size_t>(dimension());
    }

    Column solve(const Column &right_hand_side) const
    {
      if (right_hand_side.size() != dimension() || singular())
      {
        throw std::invalid_argument("solve requires a nonsingular matching system");
      }
      const Column transformed = transform_ * right_hand_side;
      Column solution = Column::Zero(dimension());
      for (Eigen::Index row = 0; row < dimension(); ++row)
      {
        const int pivot = pivot_col_by_row_[static_cast<std::size_t>(row)];
        solution[pivot] = numerics_.clean(transformed[row]);
      }
      return solution;
    }

    Column null_vector() const
    {
      if (!singular())
      {
        throw std::logic_error("null_vector requires a singular matrix");
      }
      std::vector<bool> is_pivot(static_cast<std::size_t>(dimension()), false);
      for (const int pivot : pivot_col_by_row_)
      {
        if (pivot >= 0)
          is_pivot[static_cast<std::size_t>(pivot)] = true;
      }
      Eigen::Index free_column = 0;
      while (free_column < dimension() &&
             is_pivot[static_cast<std::size_t>(free_column)])
      {
        ++free_column;
      }
      Column vector = Column::Zero(dimension());
      vector[free_column] = 1;
      for (Eigen::Index row = 0; row < dimension(); ++row)
      {
        const int pivot = pivot_col_by_row_[static_cast<std::size_t>(row)];
        if (pivot >= 0)
        {
          vector[pivot] = numerics_.clean(-reduced_(row, free_column));
        }
      }
      return vector;
    }

    const Matrix &original() const { return original_; }
    const Matrix &reduced() const { return reduced_; }

  private:
    Matrix original_;
    Matrix reduced_;
    Matrix transform_;
    std::vector<int> pivot_col_by_row_;
    Numerics<Number> numerics_;

    bool same_column(const Matrix &lhs, const Matrix &rhs,
                     Eigen::Index column) const
    {
      for (Eigen::Index row = 0; row < lhs.rows(); ++row)
      {
        if (!numerics_.equal(lhs(row, column), rhs(row, column)))
          return false;
      }
      return true;
    }

    void replace_column(Eigen::Index column, const Column &new_column)
    {
      const Column transformed = transform_ * new_column;
      original_.col(column) = new_column;
      reduced_.col(column) = transformed;

      std::vector<bool> fixed_pivot_column(
          static_cast<std::size_t>(dimension()), false);
      std::vector<Eigen::Index> fixed_rows;
      std::vector<Eigen::Index> free_rows;
      for (Eigen::Index row = 0; row < dimension(); ++row)
      {
        const int pivot = pivot_col_by_row_[static_cast<std::size_t>(row)];
        if (pivot >= 0 && pivot != column)
        {
          fixed_rows.push_back(row);
          fixed_pivot_column[static_cast<std::size_t>(pivot)] = true;
        }
        else
        {
          free_rows.push_back(row);
          pivot_col_by_row_[static_cast<std::size_t>(row)] = -1;
        }
      }

      std::vector<Eigen::Index> candidate_columns;
      for (Eigen::Index candidate = 0; candidate < dimension(); ++candidate)
      {
        if (!fixed_pivot_column[static_cast<std::size_t>(candidate)])
        {
          candidate_columns.push_back(candidate);
        }
      }

      std::vector<std::pair<Eigen::Index, Eigen::Index>> new_pivots;
      std::size_t pivot_position = 0;
      for (const Eigen::Index candidate : candidate_columns)
      {
        std::size_t selected = pivot_position;
        Scalar best(0);
        for (std::size_t position = pivot_position;
             position < free_rows.size(); ++position)
        {
          const Scalar magnitude =
              numerics_.magnitude(reduced_(free_rows[position], candidate));
          if (magnitude > best)
          {
            best = magnitude;
            selected = position;
          }
        }
        if (numerics_.zero(best))
          continue;
        if (selected != pivot_position)
        {
          swap_rows(free_rows[pivot_position], free_rows[selected]);
        }
        const Eigen::Index pivot_row = free_rows[pivot_position];
        scale_row(pivot_row,
                  Scalar(1) / reduced_(pivot_row, candidate));
        for (const Eigen::Index row : free_rows)
        {
          if (row == pivot_row || numerics_.zero(reduced_(row, candidate)))
          {
            continue;
          }
          add_row(row, pivot_row, -reduced_(row, candidate));
        }
        new_pivots.emplace_back(pivot_row, candidate);
        ++pivot_position;
      }

      // A single replaced column changes rank by at most one, and the old free
      // block had rank at most one.  Hence this block has rank at most two.
      if (new_pivots.size() > 2)
      {
        throw std::logic_error("rank-one RREF update produced rank above two");
      }
      for (const auto &[pivot_row, pivot_column] : new_pivots)
      {
        for (const Eigen::Index row : fixed_rows)
        {
          if (!numerics_.zero(reduced_(row, pivot_column)))
          {
            add_row(row, pivot_row, -reduced_(row, pivot_column));
          }
        }
        pivot_col_by_row_[static_cast<std::size_t>(pivot_row)] =
            static_cast<int>(pivot_column);
      }
    }

    void swap_rows(Eigen::Index first, Eigen::Index second)
    {
      if (first == second)
        return;
      reduced_.row(first).swap(reduced_.row(second));
      transform_.row(first).swap(transform_.row(second));
      std::swap(pivot_col_by_row_[static_cast<std::size_t>(first)],
                pivot_col_by_row_[static_cast<std::size_t>(second)]);
    }

    void scale_row(Eigen::Index row, Scalar factor)
    {
      reduced_.row(row) *= factor;
      transform_.row(row) *= factor;
    }

    void add_row(Eigen::Index target, Eigen::Index source,
                 Scalar factor)
    {
      reduced_.row(target) += factor * reduced_.row(source);
      transform_.row(target) += factor * transform_.row(source);
    }

    Eigen::Index select_pivot(Eigen::Index first_row, Eigen::Index column,
                              Eigen::Index last_row) const
    {
      Eigen::Index selected = -1;
      Scalar best(0);
      for (Eigen::Index row = first_row; row < last_row; ++row)
      {
        const Scalar magnitude = numerics_.magnitude(reduced_(row, column));
        if (selected < 0 || magnitude > best)
        {
          selected = row;
          best = magnitude;
        }
      }
      return selected >= 0 && !numerics_.zero(best) ? selected : -1;
    }
  };

  using IncrementalRref = IncrementalRrefT<ExactScalar>;
  using DoubleIncrementalRref = IncrementalRrefT<double>;

} // namespace orlin::detail
