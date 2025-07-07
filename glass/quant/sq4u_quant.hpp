#pragma once

#include <cmath>

#include "glass/quant/bf16_quant.hpp"
#include "glass/quant/calibrator.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/quant/utils.hpp"

namespace glass {

template <Metric metric, typename Template = Quantizer<metric, 128, 4>>
struct SQ4QuantizerUniform : Template {
    using type = SQ4QuantizerUniform;
    using data_type = uint8_t;

    constexpr static float drop_ratio = 0.01f;
    constexpr static float train_ratio = 0.1f;

    using CalibratorType = AffineCalibrator<15>;
    CalibratorType calibrator;

    SQ4QuantizerUniform() = default;

    explicit SQ4QuantizerUniform(int dim) : Template(dim) {}

    void train(const float *data, int32_t n) {
        calibrator.calibrate(data, (int64_t)n * this->dim() * train_ratio, drop_ratio);
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
            uint8_t y = calibrator.transform(from[j]);
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
            to[j] = calibrator.transform_back(y);
        }
    }

    constexpr static auto dist_func = helpa::l2a_u4_u4;

    constexpr static auto dist_func_sym = dist_func;

    using ComputerType = ComputerImpl<Tensor, dist_func, int32_t, float, uint8_t, uint8_t>;
    using SymComputerType = SymComputerImpl<Tensor, dist_func_sym, int32_t, uint8_t>;

    auto get_computer(const float *query) const {
        return ComputerType(this->storage, query, [this](const float *from, data_type *&to) {
            to = (data_type *)align_alloc(this->code_size());
            this->encode(from, to);
        });
    }

    auto get_sym_computer() const { return SymComputerType(this->storage); }
};

}  // namespace glass
