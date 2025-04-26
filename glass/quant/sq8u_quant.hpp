#pragma once

#include "glass/quant/bf16_quant.hpp"
#include "glass/quant/calibrator.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/quant/utils.hpp"

#include <cmath>
#include <thread>

namespace glass {

template <Metric metric, typename Template = Quantizer<metric, 64, 8>>
struct SQ8QuantizerUniform : Template {
  using type = SQ8QuantizerUniform;
  using data_type = int8_t;

  constexpr static float drop_ratio = 0.00f;

  using CalibratorType =
      std::conditional_t<metric == Metric::L2, AffineCalibrator<127>,
                         SymCalibrator<127>>;
  CalibratorType calibrator;

  SQ8QuantizerUniform() = default;

  explicit SQ8QuantizerUniform(int dim) : Template(dim) {}

  void train(const float *data, int32_t n) {
    calibrator.calibrate(data, (int64_t)n * this->dim(), drop_ratio);
  }

  void add(const float *data, int32_t n) {
    this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * this->dim(), (data_type *)this->get_code(i));
    }
  }

  void encode(const float *from, data_type *to) const {
    for (int j = 0; j < this->dim(); ++j) {
      to[j] = calibrator.transform(from[j]);
    }
  }

  void decode(const data_type *from, float *to) const {
    for (int j = 0; j < this->dim(); ++j) {
      to[j] = calibrator.transform_back(from[j]);
    }
  }

  constexpr static auto dist_func =
      metric == Metric::L2 ? helpa::l2_s7_s7 : helpa::dot_s8_s8;

  constexpr static auto dist_func_sym = dist_func;

  using ComputerType =
      ComputerImpl<Tensor, dist_func, int32_t, float, int8_t, int8_t>;
  using SymComputerType =
      SymComputerImpl<Tensor, dist_func_sym, int32_t, int8_t>;

  auto get_computer(const float *query) const {
    return ComputerType(this->storage, query,
                        [this](const float *from, data_type *&to) {
                          to = (data_type *)align_alloc(this->code_size());
                          this->encode(from, to);
                        });
  }

  auto get_sym_computer() const { return SymComputerType(this->storage); }
};

} // namespace glass
