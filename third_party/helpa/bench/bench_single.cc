#include <benchmark/benchmark.h>

#include <climits>
#include <random>

#include "helpa/core.hpp"
#include "helpa/ref/l2_ref.hpp"
#include "utils.hpp"

constexpr int32_t dim = 1536;

using random_bytes_engine = std::independent_bits_engine<std::default_random_engine, CHAR_BIT, unsigned char>;

static void
bench_l2_fp32_fp32(benchmark::State& s) {
    std::vector<float> x(dim);
    std::vector<float> y(dim);
    std::mt19937 rng;
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::generate(x.begin(), x.end(), [&] { return dis(rng); });
    std::generate(y.begin(), y.end(), [&] { return dis(rng); });
    check_float_equal(helpa::l2_fp32_fp32_ref(x.data(), y.data(), dim), helpa::l2_fp32_fp32(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_fp32_fp32(x.data(), y.data(), dim));
    }
}

static void
bench_l2_u8_s8(benchmark::State& s) {
    std::vector<uint8_t> x(dim);
    std::vector<int8_t> y(dim);
    random_bytes_engine rng;
    std::generate(x.begin(), x.end(), rng);
    std::generate(y.begin(), y.end(), rng);
#if !defined(__AVX512VNNI__)  // Results are not accurate is vnni is not enabled
    for (auto& t : x) t &= (1 << 7) - 1;
    for (auto& t : y) t &= (1 << 7) - 1;
#endif
    check_int32_equal(helpa::l2_u8_s8_ref(x.data(), y.data(), dim), helpa::l2_u8_s8(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_u8_s8(x.data(), y.data(), dim));
    }
}

static void
bench_l2_s8_s8(benchmark::State& s) {
    std::vector<int8_t> x(dim);
    std::vector<int8_t> y(dim);
    random_bytes_engine rng;
    std::generate(x.begin(), x.end(), rng);
    std::generate(y.begin(), y.end(), rng);
    check_int32_equal(helpa::l2_s8_s8_ref(x.data(), y.data(), dim), helpa::l2_s8_s8(x.data(), y.data(), dim));

    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_s8_s8(x.data(), y.data(), dim));
    }
}

static void
bench_l2a_u4_u4(benchmark::State& s) {
    std::vector<uint8_t> x(dim / 2);
    std::vector<uint8_t> y(dim / 2);
    random_bytes_engine rng;
    std::generate(x.begin(), x.end(), rng);
    std::generate(y.begin(), y.end(), rng);
    check_int32_equal(helpa::l2_u4_u4_ref(x.data(), y.data(), dim), helpa::l2a_u4_u4(x.data(), y.data(), dim));

    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2a_u4_u4(x.data(), y.data(), dim));
    }
}

static void
bench_dot_fp32_fp32(benchmark::State& s) {
    std::vector<float> x(dim);
    std::vector<float> y(dim);
    std::mt19937 rng;
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::generate(x.begin(), x.end(), [&] { return dis(rng); });
    std::generate(y.begin(), y.end(), [&] { return dis(rng); });
    check_float_equal(helpa::dot_fp32_fp32_ref(x.data(), y.data(), dim), helpa::dot_fp32_fp32(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dot_fp32_fp32(x.data(), y.data(), dim));
    }
}

static void
bench_dot_u8_s8(benchmark::State& s) {
    std::vector<uint8_t> x(dim);
    std::vector<int8_t> y(dim);
    random_bytes_engine rng;
    std::generate(x.begin(), x.end(), rng);
    std::generate(y.begin(), y.end(), rng);
#if !defined(__AVX512VNNI__)  // Results are not accurate is vnni is not enabled
    for (auto& t : x) t &= (1 << 7) - 1;
    for (auto& t : y) t &= (1 << 7) - 1;
#endif
    check_int32_equal(helpa::l2_u8_s8_ref(x.data(), y.data(), dim), helpa::l2_u8_s8(x.data(), y.data(), dim));
    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::l2_u8_s8(x.data(), y.data(), dim));
    }
}

static void
bench_dot_s8_s8(benchmark::State& s) {
    std::vector<int8_t> x(dim);
    std::vector<int8_t> y(dim);
    random_bytes_engine rng;
    std::generate(x.begin(), x.end(), rng);
    std::generate(y.begin(), y.end(), rng);

#if !defined(__AVX512VNNI__)  // Results are not accurate is vnni is not enabled
    for (auto& t : x) t &= (1 << 3) - 1;
    for (auto& t : y) t &= (1 << 3) - 1;
#endif
    check_int32_equal(helpa::dot_s8_s8_ref(x.data(), y.data(), dim), helpa::dot_s8_s8(x.data(), y.data(), dim));

    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dot_s8_s8(x.data(), y.data(), dim));
    }
}

static void
bench_dota_u4_u4(benchmark::State& s) {
    std::vector<uint8_t> x(dim / 2);
    std::vector<uint8_t> y(dim / 2);
    random_bytes_engine rng;
    std::generate(x.begin(), x.end(), rng);
    std::generate(y.begin(), y.end(), rng);
    check_int32_equal(helpa::dot_u4_u4_ref(x.data(), y.data(), dim), helpa::dota_u4_u4(x.data(), y.data(), dim));

    for (auto _ : s) {
        benchmark::DoNotOptimize(helpa::dota_u4_u4(x.data(), y.data(), dim));
    }
}

BENCHMARK(bench_l2_fp32_fp32);
BENCHMARK(bench_l2_s8_s8);
BENCHMARK(bench_l2_u8_s8);
BENCHMARK(bench_l2a_u4_u4);
BENCHMARK(bench_dot_fp32_fp32);
BENCHMARK(bench_dot_s8_s8);
BENCHMARK(bench_dot_u8_s8);
BENCHMARK(bench_dota_u4_u4);
