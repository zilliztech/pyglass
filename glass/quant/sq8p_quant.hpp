#pragma once

#include <atomic>
#include <cmath>

#include "glass/quant/bf16_quant.hpp"
#include "glass/quant/calibrator.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/quant/utils.hpp"
#include "helpa/dot.hpp"

namespace glass {

template <Metric metric, typename Template = Quantizer<metric, 64, 8>>
struct SQ8Quantizer2 : Template {
    using type = SQ8Quantizer2;
    using data_type = int8_t;

    constexpr static int32_t ncount = 127;

    std::vector<bf16> maxs;
    std::vector<bf16> norms;

    SQ8Quantizer2() = default;

    explicit SQ8Quantizer2(int dim) : Template(dim) {}

    void train(const float *, int32_t) {}

    void add(const float *data, int32_t n) {
        this->storage.init(n);
        maxs.resize(n);
        if constexpr (metric == Metric::L2) {
            norms.resize(n);
        }
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < n; ++i) {
            const float *vec = data + (int64_t)i * this->dim();
            maxs[i] = bf16(encode(vec, (data_type *)this->get_code(i)));
            if constexpr (metric == Metric::L2) {
                norms[i] = bf16(-helpa::dot_fp32_fp32(vec, vec, this->dim()));
            }
        }
    }

    float encode(const float *from, data_type *to) const {
        float mx = 0.0f;
        for (int j = 0; j < this->dim(); ++j) {
            mx = std::max(mx, std::abs(from[j]));
        }
        for (int j = 0; j < this->dim(); ++j) {
            float x = from[j] / mx;
            x = limit_range_sym(x);
            int8_t y = x * ncount;
            to[j] = y;
        }
        return mx;
    }

    struct ComputerType : Computer<Tensor> {
        using dist_type = float;
        const bf16 *maxs = nullptr, *norms = nullptr;
        int8_t *q = nullptr;
        float mxq = 0.0f;

        ComputerType(const Tensor &tensor, const float *query, const auto &encoder, const bf16 *maxs, const bf16 *norms)
            : Computer<Tensor>(tensor), maxs(maxs), norms(norms) {
            encoder(query, q);
            for (int i = 0; i < tensor.dim(); ++i) {
                mxq = std::max(mxq, std::abs(query[i]));
            }
        }

        ~ComputerType() { free(q); }

        GLASS_INLINE float operator()(int32_t u) const {
            dist_type sum;
            auto dot = helpa::dot_s8_s8(q, (const int8_t *)this->tensor.get(u), this->tensor.dim_align()) *
                       float(maxs[u]) * mxq / ncount / ncount;
            if constexpr (metric == Metric::L2) {
                sum = float(norms[u]) + 2 * dot;
            } else {
                sum = dot;
            }
            return sum;
        }
    };

    auto get_computer(const float *query) const {
        return ComputerType(
            this->storage, query,
            [this](const float *from, data_type *&to) {
                to = (data_type *)align_alloc(this->code_size());
                this->encode(from, to);
            },
            maxs.data(), norms.data());
    }
};

}  // namespace glass
