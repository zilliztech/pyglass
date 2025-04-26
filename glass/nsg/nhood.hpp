#pragma once

#include "glass/neighbor.hpp"
#include "glass/utils.hpp"
#include <functional>
#include <mutex>
#include <random>
#include <vector>

namespace glass {

struct Nhood {
  std::vector<FlagNeighbor> pool; // candidate pool (a max heap)
  int M;
  std::mutex lock;
  std::vector<int> nn_new, nn_old;
  std::vector<int> rnn_new, rnn_old;

  Nhood(int s, int64_t N) {
    M = s;
    nn_new.resize(s * 2);
    for (auto &x : nn_new) {
      x = rand() % N;
    }
  }

  Nhood &operator=(const Nhood &other) {
    M = other.M;
    std::copy(other.nn_new.begin(), other.nn_new.end(),
              std::back_inserter(nn_new));
    nn_new.reserve(other.nn_new.capacity());
    pool.reserve(other.pool.capacity());
    return *this;
  }

  Nhood(const Nhood &other) {
    M = other.M;
    std::copy(other.nn_new.begin(), other.nn_new.end(),
              std::back_inserter(nn_new));
    nn_new.reserve(other.nn_new.capacity());
    pool.reserve(other.pool.capacity());
  }

  void insert(int id, float dist) {
    std::scoped_lock guard(lock);
    if (dist > pool.front().distance)
      return;
    for (int i = 0; i < (int)pool.size(); i++) {
      if (id == pool[i].id)
        return;
    }
    if (pool.size() < pool.capacity()) {
      pool.push_back(FlagNeighbor(id, dist, true));
      std::push_heap(pool.begin(), pool.end());
    } else {
      std::pop_heap(pool.begin(), pool.end());
      pool[pool.size() - 1] = FlagNeighbor(id, dist, true);
      std::push_heap(pool.begin(), pool.end());
    }
  }

  template <typename C> void join(const auto &computer, C callback) const {
    static_assert(
        std::is_convertible_v<C, std::function<void(int32_t, int32_t)>>);
    int32_t nsz = nn_new.size(), osz = nn_old.size();
    for (int32_t i = 0; i < nsz; ++i) {
      int32_t u = nn_new[i];
      for (int32_t j = 0; j < nsz; ++j) {
        if (j + 1 < nsz && u < nn_new[j + 1]) {
          computer.prefetch(nn_new[j + 1], 1);
        }
        int32_t v = nn_new[j];
        if (u < v) {
          callback(u, v);
        }
      }
      for (int32_t j = 0; j < osz; ++j) {
        if (j + 1 < osz) {
          computer.prefetch(nn_old[j + 1], 1);
        }
        int32_t v = nn_old[j];
        callback(u, v);
      }
    }
  }
};
} // namespace glass
