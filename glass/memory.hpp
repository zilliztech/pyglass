#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

#include "glass/common.hpp"

namespace glass {

constexpr size_t size_64B = 64;
constexpr size_t size_2M = 2 * 1024 * 1024;
constexpr size_t size_1G = 1 * 1024 * 1024 * 1024;

template <size_t alignment>
inline void *align_alloc_memory(size_t nbytes, bool set = true, uint8_t x = 0) {
  size_t len = (nbytes + alignment - 1) / alignment * alignment;
  if (alignment == size_1G) {
    printf("Allocating %.2fG memory for %.2fG data\n", double(len) / size_1G,
           double(nbytes) / size_1G);
  }
  auto p = std::aligned_alloc(alignment, len);
  if constexpr (alignment >= size_2M) {
    madvise(p, len, MADV_HUGEPAGE);
  }
  if (set) {
    std::memset(p, x, len);
  }
  return p;
}

inline void *align_alloc(size_t nbytes, bool set = true, uint8_t x = 0) {
  if (nbytes >= size_1G) {
    return align_alloc_memory<size_1G>(nbytes, set, x);
  } else if (nbytes >= size_2M) {
    return align_alloc_memory<size_2M>(nbytes, set, x);
  } else {
    return align_alloc_memory<size_64B>(nbytes, set, x);
  }
}

} // namespace glass
