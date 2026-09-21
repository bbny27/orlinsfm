#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace orlin
{

  using ExactScalar = boost::multiprecision::cpp_rational;
  using Scalar = ExactScalar; 
  using Subset = std::vector<std::uint8_t>;

  template <class Number>
  using VectorT = std::vector<Number>;

  template <class Number>
  using MatrixT =
      Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template <class Number>
  using ColumnT = Eigen::Matrix<Number, Eigen::Dynamic, 1>;

  using Vector = VectorT<ExactScalar>;
  using Matrix = MatrixT<ExactScalar>;
  using Column = ColumnT<ExactScalar>;
  using DoubleVector = VectorT<double>;
  using DoubleMatrix = MatrixT<double>;
  using DoubleColumn = ColumnT<double>;

  template <class Number>
  class Numerics
  {
  public:
    Numerics(double = 0.0, double = 0.0) {}

    Number from_exact(const ExactScalar &value) const { return Number(value); }
    bool zero(const Number &value, const Number & = Number(1)) const
    {
      return value == Number(0);
    }
    bool equal(const Number &first, const Number &second) const
    {
      return first == second;
    }
    bool positive(const Number &value) const { return value > Number(0); }
    bool negative(const Number &value) const { return value < Number(0); }
    bool nonnegative(const Number &value) const { return value >= Number(0); }
    bool nonpositive(const Number &value) const { return value <= Number(0); }
    Number clean(const Number &value, const Number & = Number(1)) const
    {
      return value;
    }
    Number magnitude(const Number &value) const
    {
      return value < Number(0) ? -value : value;
    }
    Number lu_threshold() const { return Number(0); }
  };

  template <>
  class Numerics<double>
  {
  public:
    Numerics(double absolute_tolerance = 1e-10,
             double relative_tolerance = 1e-12)
        : absolute_(absolute_tolerance), relative_(relative_tolerance)
    {
      if (!(absolute_ >= 0.0) || !(relative_ >= 0.0) ||
          !std::isfinite(absolute_) || !std::isfinite(relative_))
      {
        throw std::invalid_argument(
            "floating tolerances must be finite and nonnegative");
      }
    }

    double from_exact(const ExactScalar &value) const
    {
      const double result = value.convert_to<double>();
      if (!std::isfinite(result))
      {
        throw std::overflow_error("oracle value is not representable as double");
      }
      return result;
    }

    bool zero(double value, double scale = 1.0) const
    {
      require_finite(value);
      require_finite(scale);
      return std::abs(value) <= tolerance(scale);
    }
    bool equal(double first, double second) const
    {
      return zero(first - second,
                  std::max({1.0, std::abs(first), std::abs(second)}));
    }
    bool positive(double value) const
    {
      require_finite(value);
      return value > tolerance(value);
    }
    bool negative(double value) const
    {
      require_finite(value);
      return value < -tolerance(value);
    }
    bool nonnegative(double value) const { return !negative(value); }
    bool nonpositive(double value) const { return !positive(value); }
    double clean(double value, double scale = 1.0) const
    {
      return zero(value, scale) ? 0.0 : value;
    }
    double magnitude(double value) const
    {
      require_finite(value);
      return std::abs(value);
    }

    // FullPivLU interprets this as a relative pivot threshold.
    double lu_threshold() const
    {
      return std::max(relative_,
                      10.0 * std::numeric_limits<double>::epsilon());
    }

  private:
    double absolute_;
    double relative_;

    static void require_finite(double value)
    {
      if (!std::isfinite(value))
      {
        throw std::overflow_error("non-finite value in double Orlin arithmetic");
      }
    }

    double tolerance(double scale) const
    {
      return absolute_ + relative_ * std::max(1.0, std::abs(scale));
    }
  };

  // Diagnostic only; all arithmetic and normalization are provided by Boost.
  inline std::size_t bit_size(const ExactScalar &value)
  {
    auto numerator = boost::multiprecision::numerator(value);
    if (numerator < 0)
      numerator = -numerator;
    const auto denominator = boost::multiprecision::denominator(value);
    return std::max(numerator == 0 ? std::size_t(0) : std::size_t(boost::multiprecision::msb(numerator) + 1),
                    std::size_t(boost::multiprecision::msb(denominator) + 1));
  }

  inline std::size_t bit_size(double) { return 0; }

} // namespace orlin
