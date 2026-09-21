#include "orlin/benchmark_instances.hpp"
#include "orlin/orlin_sfm.hpp"

#include <boost/version.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

  struct Arguments
  {
    std::filesystem::path instance_path;
    std::optional<std::filesystem::path> solution_path;
    std::string backend = "exact";
    bool verify = false;
    int repeats = 3;
    int warmups = 1;
    std::uint64_t iteration_limit = 10'000'000;
    double absolute_tolerance = 1e-10;
    double relative_tolerance = 1e-12;
  };

  Arguments parse_arguments(int argc, char **argv)
  {
    Arguments arguments;
    for (int index = 1; index < argc; ++index)
    {
      const std::string option = argv[index];
      const auto value = [&]() -> std::string
      {
        if (index + 1 >= argc)
        {
          throw std::invalid_argument("missing value after " + option);
        }
        return argv[++index];
      };
      if (option == "--instance")
      {
        arguments.instance_path = value();
      }
      else if (option == "--solution")
      {
        arguments.solution_path = std::filesystem::path(value());
      }
      else if (option == "--backend")
      {
        arguments.backend = value();
      }
      else if (option == "--repeats")
      {
        arguments.repeats = std::stoi(value());
      }
      else if (option == "--warmups")
      {
        arguments.warmups = std::stoi(value());
      }
      else if (option == "--verify-invariants")
      {
        arguments.verify = true;
      }
      else if (option == "--iteration-limit")
      {
        arguments.iteration_limit = std::stoull(value());
      }
      else if (option == "--absolute-tolerance")
      {
        arguments.absolute_tolerance = std::stod(value());
      }
      else if (option == "--relative-tolerance")
      {
        arguments.relative_tolerance = std::stod(value());
      }
      else
      {
        throw std::invalid_argument("unknown option: " + option);
      }
    }
    if (arguments.instance_path.empty())
    {
      throw std::invalid_argument(
          "usage: orlin_benchmark --instance FILE [--solution FILE] "
          "[--backend exact|double] [--verify-invariants] "
          "[--iteration-limit N] [--repeats N] [--warmups N] "
          "[--absolute-tolerance X] [--relative-tolerance X]");
    }
    if (arguments.backend != "exact" && arguments.backend != "double")
    {
      throw std::invalid_argument("--backend must be exact or double");
    }
    if (arguments.repeats < 1 || arguments.warmups < 0)
    {
      throw std::invalid_argument("invalid repeat/warmup count");
    }
    return arguments;
  }

  template <class Solver>
  int run_benchmark(const orlin::benchmarks::Instance &instance,
                    const Arguments &arguments, const char *implementation,
                    const char *arithmetic)
  {
    orlin::SolverOptions options;
    options.verify_invariants = arguments.verify;
    options.iteration_limit = arguments.iteration_limit;
    options.absolute_tolerance = arguments.absolute_tolerance;
    options.relative_tolerance = arguments.relative_tolerance;

    bool matches = true;
    std::ostringstream output;
    output << "{\"implementation\":\"" << implementation
           << "\",\"backend\":\"" << arguments.backend
           << "\",\"arithmetic\":\"" << arithmetic
           << "\",\"compiler\":\""
#ifdef _MSC_VER
           << "MSVC " << _MSC_VER
#else
           << __VERSION__
#endif
           << "\",\"boost_version\":" << BOOST_VERSION
           << ",\"n\":" << instance.oracle.size() << ",\"samples\":[";

    for (int trial = -arguments.warmups; trial < arguments.repeats; ++trial)
    {
      const auto start = std::chrono::steady_clock::now();
      const auto result = Solver(instance.oracle, options).minimize();
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      if (instance.known_optimum &&
          result.minimum_value != orlin::ExactScalar(*instance.known_optimum))
      {
        matches = false;
      }
      if (trial < 0)
        continue;
      if (trial != 0)
        output << ',';
      output << "{\"minimum\":\"" << result.minimum_value
             << "\",\"selected\":[";
      for (std::size_t i = 0; i < result.minimizer.size(); ++i)
      {
        if (i != 0)
          output << ',';
        output << result.minimizer[i];
      }
      output << "],\"elapsed_seconds\":" << std::setprecision(17) << elapsed
             << ",\"oracle_calls\":" << result.statistics.oracle_calls
             << ",\"iterations\":" << result.statistics.iterations
             << ",\"auxiliary_column_updates\":"
             << result.statistics.auxiliary_column_updates
             << ",\"reduce_calls\":" << result.statistics.reduce_calls << '}';
    }
    output << "],\"matches_known\":" << (matches ? "true" : "false")
           << "}\n";
    std::cout << output.str();
    return matches ? 0 : 2;
  }

} // namespace

int main(int argc, char **argv)
try
{
  const Arguments arguments = parse_arguments(argc, argv);
  const orlin::benchmarks::Instance instance =
      arguments.instance_path.extension() == ".max"
          ? orlin::benchmarks::load_dimacs_max(arguments.instance_path,
                                               arguments.solution_path)
          : orlin::benchmarks::load_sfm(arguments.instance_path);

  if (arguments.backend == "double")
  {
    return run_benchmark<orlin::DoubleOrlinSFM>(
        instance, arguments, "Orlin-double", "IEEE 754 binary64");
  }
  return run_benchmark<orlin::OrlinSFM>(
      instance, arguments, "Orlin-exact", "Boost cpp_rational");
}
catch (const std::exception &error)
{
  std::cerr << "orlin_benchmark: " << error.what() << '\n';
  return 1;
}
