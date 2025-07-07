#pragma once

#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/types.hpp"

namespace glass {

#if defined(USE_AVX512FP16)
constexpr int32_t fp16_align = 32;
#else
constexpr int32_t fp16_align = 16;
#endif

template <Metric metric, typename Template = Quantizer<metric, fp16_align, 16>>
struct FP16Quantizer : Template {
    using type = FP16Quantizer;
    using data_type = fp16;

    FP16Quantizer() = default;

    explicit FP16Quantizer(int dim) : Template(dim) {}

    void train(const float *data, int32_t n) {}

    void add(const float *data, int32_t n) {
        this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < n; ++i) {
            encode(data + i * this->dim(), (data_type *)this->get_code(i));
        }
    }

    void encode(const float *from, data_type *to) const { helpa::fp32_to_fp16(from, to, this->dim()); }

    void decode(const data_type *from, float *to) const {
        for (int i = 0; i < this->dim(); ++i) {
            to[i] = float(from[i]);
        }
    }

    constexpr static auto dist_func = metric == Metric::L2 ? helpa::l2_fp32_fp16 : helpa::dot_fp32_fp16;

    constexpr static auto dist_func_sym =
        metric == Metric::L2 ? helpa::l2_fp16_fp16 : [](const fp16 *x, const fp16 *y, const int32_t d) {
            return helpa::dot_fp16_fp16(x, y, d);
        };

    using ComputerType = ComputerImpl<Tensor, dist_func, float, float, float, fp16>;
    using SymComputerType = SymComputerImpl<Tensor, dist_func_sym, float, fp16>;

    auto get_computer(const float *query) const { return ComputerType(this->storage, query, MemCpyTag{}); }

    auto get_sym_computer() const { return SymComputerType(this->storage); }
};

}  // namespace glass
