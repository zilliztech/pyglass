#pragma once

#include "glass/common.hpp"
#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/simd/distance.hpp"

#include <cmath>

namespace glass {

template <Metric metric, typename Reorderer = FP32Quantizer<metric>,
          int DIM = 0>
struct SQ4Quantizer {
  using data_type = uint8_t;
  constexpr static int kAlign = 128;
  // todo：mx,mi和dif可以删掉，后续不会用
  float mx = -HUGE_VALF, mi = HUGE_VALF, dif;
  int d, d_align;
  int64_t code_size;
  data_type *codes = nullptr;
  // todo：增加成员变量float *centroid, 存储每一维的平均值

  Reorderer reorderer;

  SQ4Quantizer() = default;

  explicit SQ4Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align / 2),  // todo：由于mx和mi存储在一起，code_size应该把mx和mi的大小也考虑进来
        reorderer(dim) {}

  ~SQ4Quantizer() { free(codes); }

  void train(const float *data, int n) {
    // todo：初始化centroid
    // todo：不使用全局mx和mi，仿照LVQ，每个向量有自己的mx和mi
    for (int64_t i = 0; i < n * d; ++i) {
      mx = std::max(mx, data[i]);
      mi = std::min(mi, data[i]);
    }
    dif = mx - mi;
    codes = (data_type *)alloc2M(n * code_size);
    for (int i = 0; i < n; ++i) {
      // todo：每个向量根据自己的mx和mi进行量化，并且mx和mi与每个向量的code存储在一起
      encode(data + i * d, get_data(i));
    }
    reorderer.train(data, n);
  }

  char *get_data(int u) const { return (char *)codes + u * code_size; }

  // todo：修改这个函数，void encode(const float *from, char *to, float u, float l) const，根据u和l来量化向量，并将u(mx)和l(mi)与code存储在一起
  void encode(const float *from, char *to) const {
    for (int j = 0; j < d; ++j) {
      float x = (from[j] - mi) / dif;
      if (x < 0.0) {
        x = 0.0;
      }
      if (x > 0.999) {
        x = 0.999;
      }
      uint8_t y = 16 * x;
      if (j & 1) {
        to[j / 2] |= y << 4;
      } else {
        to[j / 2] |= y;
      }
    }
  }

  template <typename Pool>
  void reorder(const Pool &pool, const float *q, int *dst, int k) const {
    int cap = pool.capacity();
    auto computer = reorderer.get_computer(q);
    searcher::MaxHeap<typename Reorderer::template Computer<0>::dist_type> heap(
        k);
    for (int i = 0; i < cap; ++i) {
      if (i + 1 < cap) {
        computer.prefetch(pool.id(i + 1), 1);
      }
      int id = pool.id(i);
      float dist = computer(id);
      heap.push(id, dist);
    }
    for (int i = 0; i < k; ++i) {
      dst[i] = heap.pop();
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = int32_t;
    // todo：另外使用avx256实现一个距离计算函数，该函数做了两件事：1、根据code、mx和mi将code恢复为原向量，原向量元素类型为float；2、将恢复后的向量与query算L2距离
    constexpr static auto dist_func = L2SqrSQ4;
    const SQ4Quantizer &quant;
    uint8_t *q;  // todo：这里改成float *q;
    Computer(const SQ4Quantizer &quant, const float *query)
        : quant(quant), q((uint8_t *)alloc64B(quant.code_size)) {
      // todo：不需要量化query。原逻辑是返回量化的query和code的距离，现在改为了原query与原向量的距离
      quant.encode(query, (char *)q);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align);
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
