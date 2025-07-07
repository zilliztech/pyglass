#pragma once

#include "glass/quant/calibrator.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/quant/utils.hpp"
#include "helpa/l2.hpp"

namespace glass {

template <Metric metric, typename Template = Quantizer<metric, 256, 2>>
struct SQ2QuantizerUniform : Template {
    using type = SQ2QuantizerUniform;
    using data_type = uint8_t;

    constexpr static float drop_ratio = 0.05f;
    constexpr static float train_ratio = 0.1f;

    using CalibratorType = AffineCalibrator<3>;
    CalibratorType calibrator;

    SQ2QuantizerUniform() = default;

    explicit SQ2QuantizerUniform(int dim) : Template(dim) {}

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
        memset(to, 0, (this->dim() + 3) / 4);
        for (int j = 0; j < this->dim(); ++j) {
            uint8_t y = calibrator.transform(from[j]);
            if (j % 4 == 0) {
                to[j / 4] |= y;
            } else if (j % 4 == 1) {
                to[j / 4] |= y << 2;
            } else if (j % 4 == 2) {
                to[j / 4] |= y << 4;
            } else {
                to[j / 4] |= y << 6;
            }
        }
    }

    constexpr static auto dist_func = helpa::l2a_u2_u2;

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
