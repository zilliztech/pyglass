#pragma once

#include <cstdint>

#include "glass/quant/utils.hpp"

namespace glass {

template <int32_t multiplier>
struct Calibrator {
    constexpr static int32_t mul = multiplier;
};

template <int32_t multiplier>
struct AffineCalibrator : Calibrator<multiplier> {
    float min = 0.0f;
    float dif = 0.0f;

    AffineCalibrator() = default;

    void calibrate(const float *data, int64_t nitems, float drop_ratio = 0.0f) {
        float max;
        std::tie(this->min, max) = find_minmax(data, nitems, drop_ratio);
        this->dif = max - this->min;
        printf("AffineCalibrator calibration done, min = %f, max = %f, dif = %f\n", this->min, max, this->dif);
    }

    int32_t transform(float x) const {
        x = (x - min) / dif;
        x = limit_range(x);
        return std::round(x * this->mul);
    }

    float transform(float x, int32_t mul) const {
        x = (x - min) / dif;
        x = limit_range(x);
        return std::round(x * mul);
    }

    float transform_back(int32_t x) const { return float(x) / this->mul * dif + min; }

    float transform_back(int32_t x, int32_t mul) const { return float(x) / mul * dif + min; }
};

template <int32_t multiplier>
struct SymCalibrator : Calibrator<multiplier> {
    float max = 0.0f;

    SymCalibrator() = default;

    void calibrate(const float *data, int64_t nitems, float drop_ratio = 0.0f) {
        max = find_absmax(data, nitems, drop_ratio);
        printf("SymCalibrator calibration done, max = %f\n", this->max);
    }

    int32_t transform(float x) const {
        x = x / max;
        x = limit_range_sym(x);
        return std::round(x * this->mul);
    }

    float transform_back(int32_t x) const { return float(x) / this->mul * max; }
};

template <int32_t multiplier>
struct AffinePerDimCalibrator : Calibrator<multiplier> {
    int32_t d = 0;
    std::vector<float> mins;
    std::vector<float> difs;

    AffinePerDimCalibrator() = default;

    explicit AffinePerDimCalibrator(int32_t dim) : d(dim), mins(d), difs(d) {}

    void calibrate(const float *data, int32_t n, int32_t d, float drop_ratio = 0.0f) {
        std::vector<float> maxs;
        find_minmax_perdim(mins, maxs, data, n, d, drop_ratio);
        for (int32_t i = 0; i < d; ++i) {
            difs[i] = maxs[i] - mins[i];
        }
        printf("AffinePerDimCalibrator calibration done\n");
    }

    int32_t transform(float x, int32_t dim) const {
        x = (x - mins[dim]) / difs[dim];
        x = limit_range(x);
        return std::round(x * this->mul);
    }

    float transform_back(int32_t x, int32_t dim) const { return x / this->mul * difs[dim] + mins[dim]; }
};

}  // namespace glass
