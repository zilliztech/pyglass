#pragma once

#include "glass/common.hpp"
#include "glass/quant/bf16_quant.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/types.hpp"

namespace glass {

template <Metric metric, typename Template = Quantizer<metric, 16, 8>>
struct E5M2Quantizer : Template {
  using type = E5M2Quantizer;
  using data_type = e5m2;

  E5M2Quantizer() = default;

  explicit E5M2Quantizer(int dim) : Template(dim) {}

  void train(const float *data, int32_t n) {
  }

  void add(const float *data, int32_t n) {
    this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
      encode(data + (int64_t)i * this->dim(), (data_type *)this->get_code(i));
    }
  }

  void encode(const float *from, data_type *to) const {
    for (int i = 0; i < this->dim(); ++i) {
      to[i] = data_type(from[i]);
    }
  }

  void decode(const data_type *from, float *to) const {
    for (int i = 0; i < this->dim(); ++i) {
      to[i] = float(from[i]);
    }
  }

  constexpr static auto dist_func = metric == Metric::L2 ? L2SqrE5M2 : IPE5M2;

  using ComputerType =
      ComputerImpl<Tensor, dist_func, float, float, float, e5m2>;

  auto get_computer(const float *query) const {
    return ComputerType(this->storage, query, MemCpyTag{});
  }
};

} // namespace glass
