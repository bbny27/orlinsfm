#include "orlin/benchmark_instances.hpp"
#include "orlin/orlin_sfm.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

template <class Solver>
void verify_instance(const orlin::benchmarks::Instance& instance,
                     const std::string& label) {
  orlin::SolverOptions options;
  options.verify_invariants = true;
  const auto result = Solver(instance.oracle, options).minimize();
  if (!instance.known_optimum ||
      result.minimum_value != orlin::ExactScalar(*instance.known_optimum)) {
    throw std::runtime_error(label + " optimum mismatch");
  }
}

void verify_both(const orlin::benchmarks::Instance& instance,
                 const std::string& label) {
  verify_instance<orlin::OrlinSFM>(instance, label + " exact");
  verify_instance<orlin::DoubleOrlinSFM>(instance, label + " double");
}

}  // namespace

int main() {
  const std::filesystem::path root = ORLIN_SOURCE_DIR;
  for (const auto& filename : {"small_cut.sfm", "small_coverage.sfm"}) {
    auto instance =
        orlin::benchmarks::load_sfm(root / "datasets" / filename);
    verify_both(instance, filename);
  }
  for (const auto& filename : {"concave_n10.sfm", "matching_3x3.sfm",
                               "facility_4x6.sfm"}) {
    auto instance = orlin::benchmarks::load_sfm(
        root / "datasets" / "varied" / filename);
    verify_both(instance, filename);
  }

  auto dimacs = orlin::benchmarks::load_dimacs_max(
      root / "tests/data/tiny.max", root / "tests/data/tiny.sol");
  verify_both(dimacs, "DIMACS");
}
