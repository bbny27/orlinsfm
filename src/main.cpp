#include "orlin/examples.hpp"
#include "orlin/orlin_sfm.hpp"
#include "orlin/verification.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

  double seconds(std::chrono::nanoseconds duration)
  {
    return std::chrono::duration<double>(duration).count();
  }

  std::string format_set(const std::vector<std::size_t> &set)
  {
    std::ostringstream stream;
    stream << '{';
    for (std::size_t i = 0; i < set.size(); ++i)
    {
      if (i != 0)
        stream << ',';
      stream << set[i];
    }
    stream << '}';
    return stream.str();
  }

  std::string json_array(const std::vector<std::size_t> &set)
  {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < set.size(); ++i)
    {
      if (i != 0)
        stream << ',';
      stream << set[i];
    }
    stream << ']';
    return stream.str();
  }

  struct Arguments
  {
    std::vector<std::string> cases;
    std::string backend = "exact";
    bool verify = true;
    bool trace = false;
    bool exhaustive = true;
    bool check_submodularity = true;
    bool json = false;
    double absolute_tolerance = 1e-10;
    double relative_tolerance = 1e-12;
  };

  void usage(const char *program)
  {
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --case NAME              run one built-in fixture (repeatable)\n"
        << "  --all                    run every built-in fixture (default)\n"
        << "  --backend exact|double   arithmetic backend (default: exact)\n"
        << "  --absolute-tolerance X   double zero tolerance (default: 1e-10)\n"
        << "  --relative-tolerance X   double relative tolerance (default: 1e-12)\n"
        << "  --no-verify              disable internal invariant checks\n"
        << "  --trace                  print one line per Orlin iteration\n"
        << "  --no-exhaustive          skip exact 2^n comparison\n"
        << "  --no-submodularity-check skip exhaustive submodularity check\n"
        << "  --json                   emit one JSON object per fixture\n"
        << "  --list                   list fixtures\n"
        << "  --help                   show this message\n";
  }

  Arguments parse_arguments(int argc, char **argv)
  {
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
    {
      const std::string token = argv[i];
      const auto value = [&]() -> std::string
      {
        if (i + 1 >= argc)
          throw std::invalid_argument(token + " needs a value");
        return argv[++i];
      };
      if (token == "--case")
      {
        arguments.cases.emplace_back(value());
      }
      else if (token == "--all")
      {
        arguments.cases = orlin::examples::fixture_names();
      }
      else if (token == "--backend")
      {
        arguments.backend = value();
      }
      else if (token == "--absolute-tolerance")
      {
        arguments.absolute_tolerance = std::stod(value());
      }
      else if (token == "--relative-tolerance")
      {
        arguments.relative_tolerance = std::stod(value());
      }
      else if (token == "--no-verify")
      {
        arguments.verify = false;
      }
      else if (token == "--trace")
      {
        arguments.trace = true;
      }
      else if (token == "--no-exhaustive")
      {
        arguments.exhaustive = false;
      }
      else if (token == "--no-submodularity-check")
      {
        arguments.check_submodularity = false;
      }
      else if (token == "--json")
      {
        arguments.json = true;
      }
      else if (token == "--list")
      {
        for (const auto &name : orlin::examples::fixture_names())
        {
          std::cout << name << '\n';
        }
        std::exit(0);
      }
      else if (token == "--help" || token == "-h")
      {
        usage(argv[0]);
        std::exit(0);
      }
      else
      {
        throw std::invalid_argument("unknown option: " + token);
      }
    }
    if (arguments.backend != "exact" && arguments.backend != "double")
    {
      throw std::invalid_argument("--backend must be exact or double");
    }
    if (arguments.cases.empty())
    {
      arguments.cases = orlin::examples::fixture_names();
    }
    return arguments;
  }

  void run_case(const std::string &name, const Arguments &arguments)
  {
    const auto oracle = orlin::examples::make_fixture(name);

    std::optional<orlin::verification::SubmodularityCheck> submodularity;
    if (arguments.check_submodularity)
    {
      submodularity =
          orlin::verification::exhaustive_submodularity_check(*oracle);
      if (!submodularity->is_submodular)
      {
        throw std::runtime_error(name + " is not submodular: " +
                                 submodularity->failure);
      }
    }

    orlin::SolverOptions options;
    options.verify_invariants = arguments.verify;
    options.trace_iterations = arguments.trace;
    options.trace_stream = &std::cerr;
    options.absolute_tolerance = arguments.absolute_tolerance;
    options.relative_tolerance = arguments.relative_tolerance;

    const auto solve_and_report = [&](auto solver)
    {
      const auto result = solver.minimize();
      const orlin::Subset returned_subset = [&]
      {
        orlin::Subset subset(oracle->size(), 0);
        for (const auto vertex : result.minimizer)
          subset[vertex] = 1;
        return subset;
      }();
      if (oracle->evaluate(returned_subset) != result.minimum_value)
      {
        throw std::runtime_error("reported objective does not match returned set");
      }

      std::optional<orlin::verification::ExhaustiveResult> exhaustive;
      if (arguments.exhaustive)
      {
        exhaustive = orlin::verification::exhaustive_minimize(*oracle);
        if (exhaustive->minimum_value != result.minimum_value)
        {
          throw std::runtime_error(
              "Orlin result disagrees with exhaustive enumeration for " + name);
        }
      }

      const auto &stats = result.statistics;
      const std::string method = "orlin_" + arguments.backend;
      if (arguments.json)
      {
        std::cout << std::setprecision(17)
                  << "{\"case\":\"" << name << "\",\"n\":" << oracle->size()
                  << ",\"method\":\"" << method << "\",\"value\":\""
                  << result.minimum_value << "\",\"set\":"
                  << json_array(result.minimizer) << ",\"seconds\":"
                  << seconds(stats.elapsed) << ",\"iterations\":"
                  << stats.iterations << ",\"oracle_calls\":"
                  << stats.oracle_calls << ",\"validation_oracle_calls\":"
                  << stats.validation_oracle_calls << ",\"greedy_calls\":"
                  << stats.greedy_calls << ",\"reduce_calls\":"
                  << stats.reduce_calls << ",\"gap_events\":"
                  << stats.distance_gap_events << ",\"max_D\":"
                  << stats.maximum_distance_functions
                  << ",\"auxiliary_systems\":" << stats.auxiliary_systems
                  << ",\"auxiliary_rebuilds\":" << stats.auxiliary_rebuilds
                  << ",\"auxiliary_column_updates\":"
                  << stats.auxiliary_column_updates
                  << ",\"singular_auxiliary_systems\":"
                  << stats.singular_auxiliary_systems
                  << ",\"nonsingular_auxiliary_systems\":"
                  << stats.nonsingular_auxiliary_systems << ",\"max_bits\":"
                  << stats.maximum_rational_bits;
        if (exhaustive.has_value())
        {
          std::cout << ",\"exhaustive_value\":\""
                    << exhaustive->minimum_value << "\",\"exhaustive_set\":"
                    << json_array(exhaustive->minimizer)
                    << ",\"exhaustive_seconds\":"
                    << seconds(exhaustive->elapsed);
        }
        if (submodularity.has_value())
        {
          std::cout << ",\"submodularity_check_seconds\":"
                    << seconds(submodularity->elapsed);
        }
        std::cout << "}\n";
        return;
      }

      std::cout << "Case: " << name << " (n=" << oracle->size() << ")\n"
                << "  Backend:         " << arguments.backend << '\n'
                << "  Orlin minimizer: " << format_set(result.minimizer) << '\n'
                << "  Orlin value:     " << result.minimum_value << '\n'
                << std::fixed << std::setprecision(9)
                << "  Orlin time:      " << seconds(stats.elapsed) << " s\n"
                << "  iterations:      " << stats.iterations << '\n'
                << "  oracle calls:    " << stats.oracle_calls << " ("
                << stats.validation_oracle_calls << " validation)\n"
                << "  greedy / Reduce / gaps: " << stats.greedy_calls << " / "
                << stats.reduce_calls << " / " << stats.distance_gap_events
                << '\n';
      if (exhaustive.has_value())
      {
        std::cout << "  exact minimizer: " << format_set(exhaustive->minimizer)
                  << '\n'
                  << "  exact value:     " << exhaustive->minimum_value << '\n';
      }
      if (submodularity.has_value())
      {
        std::cout << "  submodularity:   exhaustive PASS in "
                  << seconds(submodularity->elapsed) << " s\n";
      }
      std::cout << '\n';
    };

    if (arguments.backend == "double")
    {
      solve_and_report(orlin::DoubleOrlinSFM(*oracle, options));
    }
    else
    {
      solve_and_report(orlin::OrlinSFM(*oracle, options));
    }
  }

} // namespace

int main(int argc, char **argv)
{
  try
  {
    const Arguments arguments = parse_arguments(argc, argv);
    for (const auto &name : arguments.cases)
      run_case(name, arguments);
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "error: " << error.what() << '\n';
    usage(argv[0]);
    return 1;
  }
}
