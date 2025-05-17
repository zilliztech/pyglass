#pragma once

namespace helpa {

#if defined(__AVX2__)
#define USE_X86
#endif

#if defined(__AVX512F__)
#define USE_AVX512
#endif

#if defined(__AVX512VNNI__)
#define USE_VNNI
#endif

#if defined(__aarch64__)
#define USE_ARM
#endif

#if defined(__ARM_FEATURE_SVE)
#define USE_SVE
#define USE_SVE_BF16
#endif

#if defined(__F16C__)
#define USE_F16C
#endif

#if defined(__clang__)

#define HELPA_FAST_BEGIN
#define HELPA_FAST_END
#define HELPA_INLINE __attribute__((always_inline))

#elif defined(__GNUC__)

#define HELPA_FAST_BEGIN \
    _Pragma("GCC push_options") _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define HELPA_FAST_END _Pragma("GCC pop_options")
#define HELPA_INLINE [[gnu::always_inline]]

#endif

}  // namespace helpa
