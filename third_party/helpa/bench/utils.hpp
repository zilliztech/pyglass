#include <cassert>
#include <cmath>
#include <cstdio>

#define HELPA_ASSERT_FMT(X, FMT, ...)                                          \
    do {                                                                       \
        if (!(X)) {                                                            \
            fprintf(stderr,                                                    \
                    "Helpa assertion '%s' failed in %s "                       \
                    "at %s:%d; details: " FMT "\n",                            \
                    #X, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); \
            abort();                                                           \
        }                                                                      \
    } while (false)

void
check_float_equal(float x, float y) {
    HELPA_ASSERT_FMT((std::abs(x - y) / (std::abs(y + 1e-12))) < 1e-4, "float point %f and %f not euqal\n", x, y);
}

void
check_int32_equal(int32_t x, int32_t y) {
    HELPA_ASSERT_FMT(x == y, "int %d and %d not euqal\n", x, y);
}