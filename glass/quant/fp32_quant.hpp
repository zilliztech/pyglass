#pragma once

#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"

namespace glass {

template <Metric metric, typename Template = Quantizer<metric, 16, 32>>
struct FP32Quantizer : Template {
  using type = FP32Quantizer;
  using data_type = float;

  FP32Quantizer() = default;

  explicit FP32Quantizer(int dim) : Template(dim) {}

  void train(const float *, int32_t) {}

  void add(const float *data, int32_t n) {
    this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * this->dim(), (data_type *)this->get_code(i));
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

  constexpr static auto dist_func =
      metric == Metric::L2 ? helpa::l2_fp32_fp32 : helpa::dot_fp32_fp32;

  constexpr static auto dist_func_sym = dist_func;

  using ComputerType =
      ComputerImpl<Tensor, dist_func, float, float, float, float>;
  using SymComputerType = SymComputerImpl<Tensor, dist_func, float, float>;

  auto get_computer(const float *query) const {
    return ComputerType(this->storage, query, MemCpyTag{});
  }

  auto get_sym_computer() const { return SymComputerType(this->storage); }
};

} // namespace glass
