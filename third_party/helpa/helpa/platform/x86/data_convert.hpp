#pragma once

#include <emmintrin.h>

#include "helpa/common.hpp"

#if defined(USE_X86)

#include <immintrin.h>

#include "helpa/data_convert.hpp"
#include "helpa/ref/data_convert_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline void
fp32_to_fp16(const float* from, fp16* to, size_t size) {
    size_t sa = size / 16 * 16;
    fp32_to_fp16_align(from, to, size);
    fp32_to_fp16_ref(from + sa, to + sa, size - sa);
}

inline void
fp32_to_fp16_align(const float* from, fp16* to, size_t size) {
#if defined(USE_AVX512)
    for (size_t i = 0; i < size; i += 16) {
        auto xx = _mm512_loadu_ps(from + i);
        auto yy = _mm512_cvtps_ph(xx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256((__m256i*)(to + i), yy);
    }
#else
    for (size_t i = 0; i < size; i += 8) {
        auto xx = _mm256_loadu_ps(from + i);
        auto yy = _mm256_cvtps_ph(xx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(to + i), yy);
    }
#endif
}

}  // namespace helpa

#endif
