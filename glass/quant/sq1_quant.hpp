#pragma once

#include "glass/quant/bf16_quant.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/quant/utils.hpp"

#include <bit>
#include <cmath>
#include <immintrin.h>
#include <thread>

namespace glass {

template <Metric metric, typename Template = Quantizer<metric, 512, 1>>
struct SQ1Quantizer : Template {
  using type = SQ1Quantizer;
  using data_type = uint8_t;

  SQ1Quantizer() = default;

  explicit SQ1Quantizer(int dim) : Template(dim) {}

  void train(const float *data, int32_t n) {}

  void add(const float *data, int32_t n) {
    this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * this->dim(), (data_type *)this->get_code(i));
    }
  }

  void encode(const float *from, data_type *to) const {
    for (int j = 0; j < this->dim(); ++j) {
      if (from[j] > 0) {
        to[j / 8] |= 1 << (j % 8);
      }
    }
  }

  static int32_t distf(const uint8_t *x, const uint8_t *y, int32_t d) {
#if defined(__AVX512F__)
    __m512i sum = _mm512_setzero_si512();
    for (int i = 0; i < d; i += 512) {
      auto xx = _mm512_loadu_si512(x + i / 8);
      auto yy = _mm512_loadu_si512(y + i / 8);
      __m512i x = _mm512_xor_si512(xx, yy);
      sum = _mm512_add_epi32(sum, _mm512_popcnt_epi32(x));
    }
    return _mm512_reduce_add_epi32(sum);
#else
    int32_t sum = 0;
    const uint64_t *x64 = (const uint64_t *)x;
    const uint64_t *y64 = (const uint64_t *)y;
    for (int i = 0; i < d / 64; ++i) {
      sum += std::popcount(x64[i] ^ y64[i]);
    }
    return sum;
#endif
  }

  constexpr static auto dist_func = distf;

  constexpr static auto dist_func_sym = dist_func;

  using ComputerType =
      ComputerImpl<Tensor, dist_func, int32_t, float, uint8_t, uint8_t>;
  using SymComputerType =
      SymComputerImpl<Tensor, dist_func_sym, int32_t, uint8_t>;

  auto get_computer(const float *query) const {
    return ComputerType(this->storage, query,
                        [this](const float *from, data_type *&to) {
                          to = (data_type *)align_alloc(this->code_size());
                          this->encode(from, to);
                        });
  }

  auto get_sym_computer() const { return SymComputerType(this->storage); }
};

} // namespace glass
