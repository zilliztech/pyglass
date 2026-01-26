#pragma once

#include <sys/mman.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace glass {

constexpr size_t size_64B = 64;
constexpr size_t size_2M = 2 * 1024 * 1024;
constexpr size_t size_1G = 1 * 1024 * 1024 * 1024;

inline void *align_alloc_memory(size_t alignment, size_t nbytes, bool set = true, uint8_t x = 0) {
    size_t len = (nbytes + alignment - 1) / alignment * alignment;
    auto p = std::aligned_alloc(alignment, len);
    if (alignment >= size_2M) {
        printf("Allocate %.2fMB for %.2fMB data\n", double(len) / 1024 / 1024, double(nbytes) / 1024 / 1024);
        madvise(p, len, MADV_HUGEPAGE);
    }
    if (set) {
        std::memset(p, x, len);
    }
    return p;
}

inline void *align_alloc(size_t nbytes, bool set = true, uint8_t x = 0) {
    if (nbytes >= size_1G / 2) {
        return align_alloc_memory(size_1G, nbytes, set, x);
    } else if (nbytes >= size_2M) {
        return align_alloc_memory(size_2M, nbytes, set, x);
    } else {
        return align_alloc_memory(size_64B, nbytes, set, x);
    }
}

}  // namespace glass
