#pragma once

#include "glass/common.hpp"

#if defined(__SSE2__)
#include <immintrin.h>
#endif

namespace glass {

GLASS_INLINE inline void prefetch_L1(const void *address) {
#if defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T0);
#else
    __builtin_prefetch(address, 0, 3);
#endif
}

GLASS_INLINE inline void prefetch_L2(const void *address) {
#if defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T1);
#else
    __builtin_prefetch(address, 0, 2);
#endif
}

GLASS_INLINE inline void prefetch_L3(const void *address) {
#if defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T2);
#else
    __builtin_prefetch(address, 0, 1);
#endif
}

inline void mem_prefetch(const char *ptr, const int num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 28:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 27:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 26:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 25:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 24:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 23:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 22:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 21:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 20:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}

}  // namespace glass
