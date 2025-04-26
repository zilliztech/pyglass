#pragma once

#include "glass/common.hpp"
#include "glass/quant/calibrator.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/quant/utils.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace glass {

template <Metric metric, typename CalibratorType = AffinePerDimCalibrator<255>,
          typename Template = Quantizer<metric, 16, 8>>
struct SQ8Quantizer : Template {
  using type = SQ8Quantizer;
  using data_type = uint8_t;

  constexpr static float drop_ratio = 0.00f;

  CalibratorType calibrator;

  SQ8Quantizer() = default;

  explicit SQ8Quantizer(int dim) : Template(dim), calibrator(dim) {}

  void train(const float *data, int32_t n) {
    calibrator.calibrate(data, n, this->dim(), drop_ratio);
    calibrator.mins.resize(this->dim_align());
    calibrator.difs.resize(this->dim_align());
  }

  void add(const float *data, int32_t n) {
    this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * this->dim(), (data_type *)this->get_code(i));
    }
  }

  void encode(const float *from, uint8_t *to) const {
    for (int j = 0; j < this->dim(); ++j) {
      to[j] = calibrator.transform(from[j], j);
    }
  }

  void decode(const uint8_t *from, float *to) const {
    for (int j = 0; j < this->dim(); ++j) {
      to[j] = calibrator.transform_back(from[j], j);
    }
  }

  constexpr static auto dist_func =
      metric == Metric::L2 ? L2SqrSQ8_ext : IPSQ8_ext;

  using ComputerType = ComputerImpl<Tensor, dist_func, float, float, float,
                                    uint8_t, const float *, const float *>;

  auto get_computer(const float *query) const {
    return ComputerType(this->storage, query, MemCpyTag{},
                        calibrator.mins.data(), calibrator.difs.data());
  }
};

} // namespace glass
