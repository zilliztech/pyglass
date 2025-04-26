#pragma once

#include "glass/graph.hpp"

namespace glass {

struct Builder {
  virtual Graph<int32_t> Build(const float *data, int32_t N, int32_t dim) = 0;
  virtual double GetConstructionTime() const {
    // Unsupported
    return 0.0f;
  }
  virtual ~Builder() = default;
};

} // namespace glass