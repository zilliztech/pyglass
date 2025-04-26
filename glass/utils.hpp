#pragma once

#include <algorithm>
#include <atomic>
#include <mutex>
#include <random>
#include <unordered_set>

namespace glass {

using LockGuard = std::lock_guard<std::mutex>;

inline void GenRandom(std::mt19937 &rng, int *addr, const int size,
                      const int N) {
  for (int i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (int i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  int off = rng() % N;
  for (int i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

struct RandomGenerator {
  std::mt19937 mt;

  explicit RandomGenerator(int64_t seed = 1234) : mt((unsigned int)seed) {}

  /// random positive integer
  int rand_int() { return mt() & 0x7fffffff; }

  /// random int64_t
  int64_t rand_int64() {
    return int64_t(rand_int()) | int64_t(rand_int()) << 31;
  }

  /// generate random integer between 0 and max-1
  int rand_int(int max) { return mt() % max; }

  /// between 0 and 1
  float rand_float() { return mt() / float(mt.max()); }

  double rand_double() { return mt() / double(mt.max()); }
};

struct Timer {
#define CUR_TIME std::chrono::high_resolution_clock::now()
  Timer(const std::string &msg) : msg(msg), start(CUR_TIME) {}

  ~Timer() {
    auto ed = CUR_TIME;
    auto ela = std::chrono::duration<double>(ed - start).count();
    printf("Timer [%s] time %lfs\n", msg.c_str(), ela);
  }
  std::string msg;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

using timestamp_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
struct running_stats_printer_t {
  std::size_t total{};
  std::atomic<std::size_t> progress{};
  std::size_t last_printed_progress{};
  timestamp_t last_printed_time{};
  timestamp_t start_time{};

  running_stats_printer_t(std::size_t n, char const *msg) {
    std::printf("%s. %zu items\n", msg, n);
    total = n;
    last_printed_time = start_time = std::chrono::high_resolution_clock::now();
  }

  ~running_stats_printer_t() {
    std::size_t count = progress.load();
    timestamp_t time = std::chrono::high_resolution_clock::now();
    std::size_t duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(time - start_time)
            .count();
    float vectors_per_second = count * 1e9 / duration;
    std::printf("\r\33[2K100 %% completed, %.0f vectors/s, elapsed %.2fs\n",
                vectors_per_second, duration / 1e9);
  }

  void refresh(std::size_t step = 10000) {
    std::size_t new_progress = progress.load();
    if (new_progress - last_printed_progress < step)
      return;
    print(new_progress, total);
  }

  void print(std::size_t progress, std::size_t total) {

    constexpr char bars_k[] =
        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
    constexpr std::size_t bars_len_k = 60;

    float percentage = progress * 1.f / total;
    int left_pad = (int)(percentage * bars_len_k);
    int right_pad = bars_len_k - left_pad;

    std::size_t count_new = progress - last_printed_progress;
    timestamp_t time_new = std::chrono::high_resolution_clock::now();
    std::size_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               time_new - last_printed_time)
                               .count();
    float vectors_per_second = count_new * 1e9 / duration;
    size_t rem_time = (total - progress) / vectors_per_second;

    std::printf(
        "\r%3.3f%% [%.*s%*s] %.0f vectors/s, finished %zu/%zu, %zum%zus",
        percentage * 100.f, left_pad, bars_k, right_pad, "", vectors_per_second,
        progress, total, rem_time / 60, rem_time % 60);
    std::fflush(stdout);

    last_printed_progress = progress;
    last_printed_time = time_new;
    this->total = total;
  }
};

} // namespace glass
