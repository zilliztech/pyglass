#pragma once

#include "glass/quant/calibrator.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/quant/utils.hpp"

#include <cmath>
#include <vector>

namespace glass {

template <Metric metric, typename CalibratorType = AffinePerDimCalibrator<63>,
          typename Template = Quantizer<metric, 8, 6>>
struct SQ6Quantizer : Template {
  using type = SQ6Quantizer;
  using data_type = uint8_t;

  constexpr static float drop_ratio = 0.01f;

  CalibratorType calibrator;

  SQ6Quantizer() = default;

  explicit SQ6Quantizer(int dim) : Template(dim), calibrator(dim) {}

  void train(const float *data, int32_t n) {
    calibrator.calibrate(data, n, this->dim(), drop_ratio);
    calibrator.mins.resize(this->dim_align());
    calibrator.difs.resize(this->dim_align());
  }

  void add(const float *data, int32_t n) {
    this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
      encode(data + (int64_t)i * this->dim(), (data_type *)this->get_code(i));
    }
  }

  static void encode_component(int32_t x, uint8_t *code, int i) {
    code += (i >> 2) * 3;
    switch (i & 3) {
    case 0:
      code[0] |= x;
      break;
    case 1:
      code[0] |= x << 6;
      code[1] |= x >> 2;
      break;
    case 2:
      code[1] |= x << 4;
      code[2] |= x >> 4;
      break;
    case 3:
      code[2] |= x << 2;
      break;
    }
  }

  void encode(const float *from, data_type *to) const {
    for (int32_t i = 0; i < this->dim(); ++i) {
      int32_t x = calibrator.transform(from[i], i);
      encode_component(x, to, i);
    }
  }

  constexpr static auto dist_func =
      metric == Metric::L2 ? L2SqrSQ6_ext : IPSQ6_ext;

  using ComputerType = ComputerImpl<Tensor, dist_func, float, float, float,
                                    uint8_t, const float *, const float *>;

  auto get_computer(const float *query) const {
    return ComputerType(this->storage, query, MemCpyTag{},
                        calibrator.mins.data(), calibrator.difs.data());
  };
};

} // namespace glass
