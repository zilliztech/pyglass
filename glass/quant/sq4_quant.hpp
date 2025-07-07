#pragma once

#include <cmath>
#include <vector>

#include "glass/quant/calibrator.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/quant/utils.hpp"

namespace glass {

template <Metric metric, typename CalibratorType = AffinePerDimCalibrator<15>,
          typename Template = Quantizer<metric, 16, 4>>
struct SQ4Quantizer : Template {
    using type = SQ4Quantizer;
    using data_type = uint8_t;

    constexpr static float drop_ratio = 0.01f;

    CalibratorType calibrator;

    SQ4Quantizer() = default;

    explicit SQ4Quantizer(int dim) : Template(dim), calibrator(dim) {}

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

    void encode(const float *from, data_type *to) const {
        memset(to, 0, (this->dim() + 1) / 2);
        for (int j = 0; j < this->dim(); ++j) {
            uint8_t y = calibrator.transform(from[j], j);
            if (j & 1) {
                to[j / 2] |= y << 4;
            } else {
                to[j / 2] |= y;
            }
        }
    }

    void decode(const data_type *from, float *to) const {
        for (int j = 0; j < this->dim(); ++j) {
            uint8_t y;
            if (j & 1) {
                y = from[j / 2] >> 4 & 15;
            } else {
                y = from[j / 2] & 15;
            }
            to[j] = calibrator.transform_back(y, j);
        }
    }

    constexpr static auto dist_func = metric == Metric::L2 ? L2SqrSQ4_ext : IPSQ4_ext;

    using ComputerType = ComputerImpl<Tensor, dist_func, float, float, float, uint8_t, const float *, const float *>;

    auto get_computer(const float *query) const {
        return ComputerType(this->storage, query, MemCpyTag{}, calibrator.mins.data(), calibrator.difs.data());
    };
};

}  // namespace glass
