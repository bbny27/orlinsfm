#pragma once

#include "orlin/orlin_sfm.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace orlin::benchmarks
{

  struct Arc
  {
    std::size_t tail;
    std::size_t head;
    std::int64_t capacity;
  };

  struct Feature
  {
    std::int64_t weight;
    std::vector<std::size_t> items;
  };

  struct MatchingEdge
  {
    std::size_t left;
    std::size_t right;
  };

  class FileOracle final : public SubmodularOracle
  {
  public:
    enum class Kind
    {
      cut,
      coverage,
      concave,
      matching
    };

    FileOracle(Kind kind, std::vector<std::int64_t> unary, std::int64_t offset,
               std::vector<Arc> arcs, std::vector<Feature> features,
               std::string name, std::int64_t concavity_scale = 0,
               std::size_t matching_right_size = 0,
               std::vector<MatchingEdge> matching_edges = {})
        : kind_(kind),
          unary_(std::move(unary)),
          offset_(offset),
          arcs_(std::move(arcs)),
          features_(std::move(features)),
          name_(std::move(name)),
          concavity_scale_(concavity_scale),
          matching_right_size_(matching_right_size),
          matching_adjacency_(unary_.size())
    {
      if (concavity_scale_ < 0)
      {
        throw std::invalid_argument("concavity scale must be nonnegative");
      }
      for (const auto &edge : matching_edges)
      {
        if (edge.left >= unary_.size() || edge.right >= matching_right_size_)
        {
          throw std::invalid_argument("matching edge is outside its bipartition");
        }
        matching_adjacency_[edge.left].push_back(edge.right);
      }
    }

    std::size_t size() const override { return unary_.size(); }
    std::string name() const override { return name_; }

    Scalar evaluate(const Subset &subset) const override
    {
      if (subset.size() != size())
      {
        throw std::invalid_argument("benchmark subset has the wrong size");
      }
      boost::multiprecision::cpp_int value(offset_);
      for (std::size_t item = 0; item < size(); ++item)
      {
        if (subset[item])
          value += unary_[item];
      }
      if (kind_ == Kind::cut)
      {
        for (const auto &arc : arcs_)
        {
          if (subset[arc.tail] && !subset[arc.head])
          {
            value += arc.capacity;
          }
        }
      }
      else if (kind_ == Kind::coverage)
      {
        for (const auto &feature : features_)
        {
          bool covered = false;
          for (const auto item : feature.items)
            covered = covered || subset[item];
          if (covered)
            value += feature.weight;
        }
      }
      else if (kind_ == Kind::concave)
      {
        std::size_t cardinality = 0;
        for (const auto selected : subset)
          cardinality += selected != 0;
        const boost::multiprecision::cpp_int k(cardinality);
        value -= boost::multiprecision::cpp_int(concavity_scale_) *
                 k * (k - 1) / 2;
      }
      else
      {
        value += maximum_matching(subset);
      }
      return Scalar(value);
    }

    Kind kind() const { return kind_; }
    const std::vector<std::int64_t> &unary() const { return unary_; }
    std::int64_t offset() const { return offset_; }
    const std::vector<Arc> &arcs() const { return arcs_; }
    const std::vector<Feature> &features() const { return features_; }

  private:
    Kind kind_;
    std::vector<std::int64_t> unary_;
    std::int64_t offset_;
    std::vector<Arc> arcs_;
    std::vector<Feature> features_;
    std::string name_;
    std::int64_t concavity_scale_;
    std::size_t matching_right_size_;
    std::vector<std::vector<std::size_t>> matching_adjacency_;

    std::size_t maximum_matching(const Subset &subset) const
    {
      const std::size_t unmatched = unary_.size();
      std::vector<std::size_t> matched_left(matching_right_size_, unmatched);
      std::function<bool(std::size_t, std::vector<std::uint8_t> &)> augment =
          [&](std::size_t left, std::vector<std::uint8_t> &seen)
      {
        for (const auto right : matching_adjacency_[left])
        {
          if (seen[right])
            continue;
          seen[right] = 1;
          if (matched_left[right] == unmatched ||
              augment(matched_left[right], seen))
          {
            matched_left[right] = left;
            return true;
          }
        }
        return false;
      };
      std::size_t cardinality = 0;
      for (std::size_t left = 0; left < unary_.size(); ++left)
      {
        if (!subset[left])
          continue;
        std::vector<std::uint8_t> seen(matching_right_size_, 0);
        cardinality += augment(left, seen);
      }
      return cardinality;
    }
  };

  struct Instance
  {
    FileOracle oracle;
    std::optional<std::int64_t> known_optimum;
  };

  inline std::int64_t add_checked(std::int64_t first, std::int64_t second)
  {
    if ((second > 0 && first > std::numeric_limits<std::int64_t>::max() - second) ||
        (second < 0 && first < std::numeric_limits<std::int64_t>::min() - second))
    {
      throw std::overflow_error("benchmark capacity sum exceeds int64");
    }
    return first + second;
  }

  inline Instance load_sfm(const std::filesystem::path &path)
  {
    std::ifstream input(path);
    if (!input)
      throw std::runtime_error("cannot open instance: " + path.string());

    std::string line;
    std::string kind_name;
    std::size_t size = 0;
    bool has_header = false;
    std::int64_t offset = 0;
    std::int64_t concavity_scale = 0;
    bool has_concavity_scale = false;
    std::size_t matching_right_size = 0;
    std::optional<std::int64_t> optimum;
    std::vector<std::int64_t> unary;
    std::vector<Arc> arcs;
    std::vector<Feature> features;
    std::vector<MatchingEdge> matching_edges;

    while (std::getline(input, line))
    {
      std::istringstream row(line);
      std::string tag;
      if (!(row >> tag) || tag == "c" || tag.front() == '#')
        continue;
      if (tag == "p")
      {
        if (has_header || !(row >> kind_name >> size) ||
            (kind_name != "cut" && kind_name != "coverage" &&
             kind_name != "concave" && kind_name != "matching"))
        {
          throw std::runtime_error("invalid p line in " + path.string());
        }
        if (kind_name == "matching" &&
            (!(row >> matching_right_size) || matching_right_size == 0))
        {
          throw std::runtime_error("matching p line needs a positive right size");
        }
        has_header = true;
        unary.assign(size, 0);
      }
      else if (!has_header)
      {
        throw std::runtime_error("p line must precede data in " + path.string());
      }
      else if (tag == "offset")
      {
        if (!(row >> offset))
          throw std::runtime_error("invalid offset line");
      }
      else if (tag == "u")
      {
        std::size_t item;
        std::int64_t value;
        if (!(row >> item >> value) || item >= size)
        {
          throw std::runtime_error("invalid unary line");
        }
        unary[item] = add_checked(unary[item], value);
      }
      else if (tag == "a")
      {
        Arc arc{};
        if (!(row >> arc.tail >> arc.head >> arc.capacity) || arc.tail >= size ||
            arc.head >= size || arc.capacity < 0 || kind_name != "cut")
        {
          throw std::runtime_error("invalid arc line");
        }
        arcs.push_back(arc);
      }
      else if (tag == "f")
      {
        Feature feature{};
        if (!(row >> feature.weight) || feature.weight < 0 ||
            kind_name != "coverage")
        {
          throw std::runtime_error("invalid feature line");
        }
        std::size_t item;
        while (row >> item)
        {
          if (item >= size)
            throw std::runtime_error("invalid feature item");
          feature.items.push_back(item);
        }
        if (feature.items.empty())
          throw std::runtime_error("empty feature line");
        features.push_back(std::move(feature));
      }
      else if (tag == "q")
      {
        if (!(row >> concavity_scale) || concavity_scale < 0 ||
            kind_name != "concave" || has_concavity_scale)
        {
          throw std::runtime_error("invalid concavity line");
        }
        has_concavity_scale = true;
      }
      else if (tag == "e")
      {
        MatchingEdge edge{};
        if (!(row >> edge.left >> edge.right) || edge.left >= size ||
            edge.right >= matching_right_size || kind_name != "matching")
        {
          throw std::runtime_error("invalid matching edge line");
        }
        matching_edges.push_back(edge);
      }
      else if (tag == "o")
      {
        std::int64_t value;
        if (!(row >> value))
          throw std::runtime_error("invalid optimum line");
        optimum = value;
      }
      else
      {
        throw std::runtime_error("unknown instance tag: " + tag);
      }
    }
    if (!has_header)
      throw std::runtime_error("instance has no p line");
    if (kind_name == "concave" && !has_concavity_scale)
    {
      throw std::runtime_error("concave instance has no q line");
    }
    const auto kind = kind_name == "cut" ? FileOracle::Kind::cut : kind_name == "coverage" ? FileOracle::Kind::coverage
                                                               : kind_name == "concave"    ? FileOracle::Kind::concave
                                                                                           : FileOracle::Kind::matching;
    return {FileOracle(kind, std::move(unary), offset, std::move(arcs),
                       std::move(features), path.filename().string(),
                       concavity_scale, matching_right_size,
                       std::move(matching_edges)),
            optimum};
  }

  inline std::int64_t load_dimacs_solution(const std::filesystem::path &path)
  {
    std::ifstream input(path);
    if (!input)
      throw std::runtime_error("cannot open solution: " + path.string());
    std::string line;
    while (std::getline(input, line))
    {
      std::istringstream row(line);
      std::string tag;
      std::int64_t value;
      if (row >> tag >> value && tag == "s")
        return value;
    }
    throw std::runtime_error("DIMACS solution has no s line: " + path.string());
  }

  inline Instance load_dimacs_max(
      const std::filesystem::path &path,
      const std::optional<std::filesystem::path> &solution = std::nullopt)
  {
    std::ifstream input(path);
    if (!input)
      throw std::runtime_error("cannot open DIMACS file: " + path.string());

    std::size_t nodes = 0;
    std::size_t source = 0;
    std::size_t sink = 0;
    bool has_source = false;
    bool has_sink = false;
    std::vector<Arc> dimacs_arcs;
    std::string line;
    while (std::getline(input, line))
    {
      std::istringstream row(line);
      std::string tag;
      if (!(row >> tag) || tag == "c")
        continue;
      if (tag == "p")
      {
        std::string problem;
        std::size_t declared_arcs;
        if (!(row >> problem >> nodes >> declared_arcs) || problem != "max" ||
            nodes < 2)
        {
          throw std::runtime_error("invalid DIMACS p line");
        }
        dimacs_arcs.reserve(declared_arcs);
      }
      else if (tag == "n")
      {
        std::size_t node;
        char role;
        if (!(row >> node >> role) || node == 0 || node > nodes)
        {
          throw std::runtime_error("invalid DIMACS n line");
        }
        --node;
        if (role == 's')
        {
          source = node;
          has_source = true;
        }
        else if (role == 't')
        {
          sink = node;
          has_sink = true;
        }
      }
      else if (tag == "a")
      {
        std::size_t tail;
        std::size_t head;
        std::int64_t capacity;
        if (!(row >> tail >> head >> capacity) || tail == 0 || head == 0 ||
            tail > nodes || head > nodes || capacity < 0)
        {
          throw std::runtime_error("invalid DIMACS a line");
        }
        dimacs_arcs.push_back({tail - 1, head - 1, capacity});
      }
    }
    if (!has_source || !has_sink || source == sink)
    {
      throw std::runtime_error("DIMACS file needs distinct source and sink");
    }

    std::vector<std::size_t> index(nodes, nodes);
    std::size_t ground_size = 0;
    for (std::size_t node = 0; node < nodes; ++node)
    {
      if (node != source && node != sink)
        index[node] = ground_size++;
    }
    std::vector<std::int64_t> unary(ground_size, 0);
    std::vector<Arc> arcs;
    std::int64_t offset = 0;
    for (const auto &arc : dimacs_arcs)
    {
      if (arc.tail == source && arc.head == sink)
      {
        offset = add_checked(offset, arc.capacity);
      }
      else if (arc.tail == source && arc.head != source)
      {
        offset = add_checked(offset, arc.capacity);
        if (arc.head != sink)
        {
          unary[index[arc.head]] =
              add_checked(unary[index[arc.head]], -arc.capacity);
        }
      }
      else if (arc.head == sink && arc.tail != sink)
      {
        if (arc.tail != source)
        {
          unary[index[arc.tail]] =
              add_checked(unary[index[arc.tail]], arc.capacity);
        }
      }
      else if (arc.tail != sink && arc.head != source &&
               arc.tail != source && arc.head != sink)
      {
        arcs.push_back({index[arc.tail], index[arc.head], arc.capacity});
      }
    }
    std::optional<std::int64_t> optimum;
    if (solution)
      optimum = load_dimacs_solution(*solution);
    return {FileOracle(FileOracle::Kind::cut, std::move(unary), offset,
                       std::move(arcs), {}, path.filename().string()),
            optimum};
  }

} // namespace orlin::benchmarks
