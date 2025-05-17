#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>

#include "helpa/core.hpp"
#include "helpa/ref/l2_ref.hpp"
#include "helpa/types.hpp"

template <typename T>
auto
gen_random_fvec(int32_t dim) -> std::vector<T> {
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = T((double)rand() / (double)RAND_MAX);
    }
    return res;
}

template <typename T>
auto
gen_random_ivec(int32_t dim, int32_t lo, int32_t hi) -> std::vector<T> {
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = rand() % (hi - lo + 1) + lo;
    }
    return res;
}

void
check_float_near(float x, float y) {
    EXPECT_NEAR(x, y, std::min(std::abs(x), std::abs(y)) * 0.01);
}

void
check_int_equal(int32_t x, int32_t y) {
    EXPECT_EQ(x, y);
}

static void
bench_l2_fp32_fp32(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<float>(dim);
    auto y = gen_random_fvec<float>(dim);
    check_float_near(helpa::l2_fp32_fp32(x.data(), y.data(), dim), helpa::l2_fp32_fp32_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_fp32_fp32(x.data(), y.data(), dim));
    }
}

static void
bench_l2_fp32_fp16(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<float>(dim);
    auto y = gen_random_fvec<helpa::fp16>(dim);
    check_float_near(helpa::l2_fp32_fp16(x.data(), y.data(), dim), helpa::l2_fp32_fp16_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_fp32_fp16(x.data(), y.data(), dim));
    }
}

static void
bench_l2_fp16_fp16(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<helpa::fp16>(dim);
    auto y = gen_random_fvec<helpa::fp16>(dim);
    check_float_near(helpa::l2_fp16_fp16(x.data(), y.data(), dim), helpa::l2_fp16_fp16_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_fp16_fp16(x.data(), y.data(), dim));
    }
}

static void
bench_l2_fp32_bf16(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<float>(dim);
    auto y = gen_random_fvec<helpa::bf16>(dim);
    check_float_near(helpa::l2_fp32_bf16(x.data(), y.data(), dim), helpa::l2_fp32_bf16_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_fp32_bf16(x.data(), y.data(), dim));
    }
}

static void
bench_l2_bf16_bf16(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<helpa::bf16>(dim);
    auto y = gen_random_fvec<helpa::bf16>(dim);
    check_float_near(helpa::l2_bf16_bf16(x.data(), y.data(), dim), helpa::l2_bf16_bf16_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_bf16_bf16(x.data(), y.data(), dim));
    }
}

static void
bench_l2_s7_s7(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_ivec<int8_t>(dim, 0, 127);
    auto y = gen_random_ivec<int8_t>(dim, 0, 127);
    check_int_equal(helpa::l2_s7_s7(x.data(), y.data(), dim), helpa::l2_s7_s7_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_s7_s7(x.data(), y.data(), dim));
    }
}

static void
bench_l2a_u4_u4(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_ivec<uint8_t>(dim / 2, 0, 255);
    auto y = gen_random_ivec<uint8_t>(dim / 2, 0, 255);
    check_int_equal(helpa::l2a_u4_u4(x.data(), y.data(), dim), helpa::l2_u4_u4_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2a_u4_u4(x.data(), y.data(), dim));
    }
}

static void
bench_dot_fp32_fp32(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<float>(dim);
    auto y = gen_random_fvec<float>(dim);
    check_float_near(helpa::dot_fp32_fp32(x.data(), y.data(), dim), helpa::dot_fp32_fp32_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dot_fp32_fp32(x.data(), y.data(), dim));
    }
}

static void
bench_dot_fp32_fp16(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<float>(dim);
    auto y = gen_random_fvec<helpa::fp16>(dim);
    check_float_near(helpa::dot_fp32_fp16(x.data(), y.data(), dim), helpa::dot_fp32_fp16_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dot_fp32_fp16(x.data(), y.data(), dim));
    }
}

static void
bench_dot_fp16_fp16(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<helpa::fp16>(dim);
    auto y = gen_random_fvec<helpa::fp16>(dim);
    check_float_near(helpa::dot_fp16_fp16(x.data(), y.data(), dim), helpa::dot_fp16_fp16_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dot_fp16_fp16(x.data(), y.data(), dim));
    }
}

static void
bench_dot_fp32_bf16(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<float>(dim);
    auto y = gen_random_fvec<helpa::bf16>(dim);
    check_float_near(helpa::dot_fp32_bf16(x.data(), y.data(), dim), helpa::dot_fp32_bf16_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dot_fp32_bf16(x.data(), y.data(), dim));
    }
}

static void
bench_dot_bf16_bf16(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_fvec<helpa::bf16>(dim);
    auto y = gen_random_fvec<helpa::bf16>(dim);
    check_float_near(helpa::dot_bf16_bf16(x.data(), y.data(), dim), helpa::dot_bf16_bf16_ref(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dot_bf16_bf16(x.data(), y.data(), dim));
    }
}

static void
bench_dot_s8_s8(benchmark::State& s) {
    int32_t dim = s.range(0);
    auto x = gen_random_ivec<int8_t>(dim, -99, 99);
    auto y = gen_random_ivec<int8_t>(dim, -99, 99);
    check_int_equal(helpa::dot_s8_s8(x.data(), y.data(), dim), helpa::dot_s8_s8_ref(x.data(), y.data(), dim));

    if (helpa::dot_s8_s8(x.data(), y.data(), dim) != helpa::dot_s8_s8_ref(x.data(), y.data(), dim)) {
        exit(0);
    }
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dot_s8_s8(x.data(), y.data(), dim));
    }
}

#define BENCHIT(bench) \
    BENCHMARK(bench)->ArgNames({"dim"})->ArgsProduct({{128, 256, 512, 1024, 2048}})->Unit(benchmark::kNanosecond)->DisplayAggregatesOnly();

BENCHIT(bench_l2_fp32_fp32)
BENCHIT(bench_l2_fp32_fp16)
BENCHIT(bench_l2_fp16_fp16)
BENCHIT(bench_l2_fp32_bf16)
BENCHIT(bench_l2_bf16_bf16)
BENCHIT(bench_l2_s7_s7)
BENCHIT(bench_l2a_u4_u4)

BENCHIT(bench_dot_fp32_fp32)
BENCHIT(bench_dot_fp32_fp16)
BENCHIT(bench_dot_fp16_fp16)
BENCHIT(bench_dot_fp32_bf16)
BENCHIT(bench_dot_bf16_bf16)
BENCHIT(bench_dot_s8_s8)

BENCHMARK_MAIN();
