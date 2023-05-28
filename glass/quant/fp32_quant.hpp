#pragma once

#include "glass/common.hpp"
#include "glass/memory.hpp"
#include "glass/simd/distance.hpp"

namespace glass {

template <Metric metric, int DIM = 0> struct FP32Quantizer {
  using data_type = float;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;

  FP32Quantizer() = default;

  explicit FP32Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align * 4) {}

  ~FP32Quantizer() { free(codes); }

  void train(const float *data, int64_t n) {
    codes = (char *)alloc2M(n * code_size);
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  void encode(const float *from, char *to) { std::memcpy(to, from, d * 4); }

  char *get_data(int u) const { return codes + u * code_size; }

  template <typename Pool>
  void reorder(const Pool &pool, const float *, int *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = float;
    constexpr static auto dist_func = metric == Metric::L2 ? L2Sqr : IP;
    const FP32Quantizer &quant;
    float *q = nullptr;
    Computer(const FP32Quantizer &quant, const float *query)
        : quant(quant), q((float *)alloc64B(quant.d_align * 4)) {
      std::memcpy(q, query, quant.d * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace glass
