#pragma once

#include "glass/quant/calibrator.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/simd/common.hpp"

namespace glass {

template <Metric metric, typename CalibratorType = AffineCalibrator<15>, typename Template = Quantizer<metric, 128, 4>>
struct SQ4QuantizerUniformAsym : Template {
    using type = SQ4QuantizerUniformAsym;
    using data_type = uint8_t;

    constexpr static float drop_ratio = 0.01f;

    CalibratorType calibrator;

    SQ4QuantizerUniformAsym() = default;

    explicit SQ4QuantizerUniformAsym(int dim) : Template(dim) {}

    void train(const float *data, int32_t n) { calibrator.calibrate(data, (int64_t)n * this->dim(), drop_ratio); }

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
            uint8_t y = calibrator.transform(from[j]);
            if (j & 1) {
                to[j / 2] |= y << 4;
            } else {
                to[j / 2] |= y;
            }
        }
    }

    constexpr static auto dist_func = L2SqrSQ8SQ4;

    using ComputerType = ComputerImpl<Tensor, dist_func, int32_t, float, uint8_t, uint8_t>;

    auto get_computer(const float *query) const {
        auto encode8 = [this](const float *from, data_type *&to) {
            to = (data_type *)align_alloc(this->dim_align() * sizeof(uint8_t));
            for (int j = 0; j < this->dim(); ++j) {
                to[j] = calibrator.transform(from[j], 127);
            }
        };
        return ComputerType(this->storage, query, encode8);
    }
};

}  // namespace glass
