#pragma once

#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/types.hpp"
#include "helpa/core.hpp"

namespace glass {

#if defined(USE_AVX512BF16) && defined(__AVX512BF16__)
constexpr int32_t bf16_align = 32;
#else
constexpr int32_t bf16_align = 16;
#endif

template <Metric metric, typename Template = Quantizer<metric, bf16_align, 16>>
struct BF16Quantizer : Template {
    using type = BF16Quantizer;
    using data_type = bf16;

    BF16Quantizer() = default;

    explicit BF16Quantizer(int dim) : Template(dim) {}

    void train(const float *data, int32_t n) {}

    void add(const float *data, int32_t n) {
        this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < n; ++i) {
            encode(data + (int64_t)i * this->dim(), (data_type *)this->get_code(i));
        }
    }

    void encode(const float *from, data_type *to) const {
        for (int i = 0; i < this->dim(); ++i) {
            to[i] = data_type(from[i]);
        }
    }

    void decode(const data_type *from, float *to) const {
        for (int i = 0; i < this->dim(); ++i) {
            to[i] = float(from[i]);
        }
    }

#if defined(USE_AVX512BF16) && defined(__AVX512BF16__)

    using ComputerType = SymComputer<typename Template::storage_type, metric, float, float, BF16>;

    auto get_computer(const float *query) const { return ComputerType(this->storage, query, *this); }

    constexpr static auto dist_func = metric == Metric::L2 ? helpa::l2_bf16_bf16 : helpa::dot_bf16_bf16;

    constexpr static auto dist_func_sym = dist_func;

    using ComputerType = ComputerImpl<dist_func, float, float, BF16, BF16>;
    using SymComputerType = SymComputerImpl<dist_func_sym, float, BF16>;

    auto get_computer(const float *query) const {
        return ComputerType(this->storage, query, [this](const float *from, data_type *&to) {
            to = (data_type *)align_alloc(this->code_size());
            this->encode(from, to);
        });
    }

    auto get_sym_computer() const { return SymComputerType(this->storage); }

#else

    constexpr static auto dist_func =
        metric == Metric::L2 ? helpa::l2_fp32_bf16 : [](const float *x, const bf16 *y, const int32_t d) {
            return helpa::dot_fp32_bf16(x, y, d);
        };

    constexpr static auto dist_func_sym =
        metric == Metric::L2 ? helpa::l2_bf16_bf16 : [](const bf16 *x, const bf16 *y, const int32_t d) {
            return helpa::dot_bf16_bf16(x, y, d);
        };

    using ComputerType = ComputerImpl<Tensor, dist_func, float, float, float, bf16>;
    using SymComputerType = SymComputerImpl<Tensor, dist_func_sym, float, bf16>;

    auto get_computer(const float *query) const { return ComputerType(this->storage, query, MemCpyTag{}); }

    auto get_sym_computer() const { return SymComputerType(this->storage); }

#endif
};

}  // namespace glass
