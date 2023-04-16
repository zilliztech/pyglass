#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <omp.h>
#include <string>
#include <vector>

#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/simd/distance.hpp"
#include "glass/utils.hpp"

namespace glass {

struct NNDescent {
  struct Nhood {
    std::vector<Neighbor> pool; // candidate pool (a max heap)
    int M;
    std::mutex lock;
    std::vector<int> nn_new, nn_old;
    std::vector<int> rnn_new, rnn_old;

    Nhood(std::mt19937 &rng, int s, int64_t N) {
      M = s;
      nn_new.resize(s * 2);
      GenRandom(rng, nn_new.data(), (int)nn_new.size(), N);
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
        pool.push_back(Neighbor(id, dist, true));
        std::push_heap(pool.begin(), pool.end());
      } else {
        std::pop_heap(pool.begin(), pool.end());
        pool[pool.size() - 1] = Neighbor(id, dist, true);
        std::push_heap(pool.begin(), pool.end());
      }
    }

    template <typename C> void join(C callback) const {
      for (int const i : nn_new) {
        for (int const j : nn_new) {
          if (i < j) {
            callback(i, j);
          }
        }
        for (int j : nn_old) {
          callback(i, j);
        }
      }
    }
  };

  std::vector<Nhood> graph;
  Graph<int> final_graph;
  int64_t d;
  int64_t nb;
  const float *data;
  int K;
  int S = 10;
  int R = 100;
  int iters = 10;
  int random_seed = 347;
  int L;
  Dist<float, float, float> dist_func;

  NNDescent(int64_t dim, const std::string &metric) : d(dim) {
    if (metric == "L2") {
      dist_func = L2SqrRef;
    } else if (metric == "IP") {
      dist_func = IPRef;
    }
  }

  void Build(const float *data, int n, int K) {
    this->data = data;
    this->nb = n;
    this->K = K;
    this->L = K + 50;
    Init();
    Descent();
    final_graph.init(n, K);
    for (int i = 0; i < nb; i++) {
      std::sort(graph[i].pool.begin(), graph[i].pool.end());
      for (int j = 0; j < K; j++) {
        final_graph.at(i, j) = graph[i].pool[j].id;
      }
    }
    std::vector<Nhood>().swap(graph);
  }

  void Init() {
    graph.reserve(nb);
    {
      std::mt19937 rng(random_seed * 6007);
      for (int i = 0; i < nb; ++i) {
        graph.emplace_back(rng, S, nb);
      }
    }
#pragma omp parallel
    {
      std::mt19937 rng(random_seed * 7741 + omp_get_thread_num());
#pragma omp for
      for (int i = 0; i < nb; ++i) {
        std::vector<int> tmp(S);
        GenRandom(rng, tmp.data(), S, nb);
        for (int j = 0; j < S; j++) {
          int id = tmp[j];
          if (id == i)
            continue;
          float dist = dist_func(data + i * d, data + id * d, d);
          graph[i].pool.push_back(Neighbor(id, dist, true));
        }
        std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
        graph[i].pool.reserve(L);
      }
    }
  }

  void Descent() {
    int num_eval = std::min((int64_t)100, nb);
    std::vector<int> eval_points(num_eval);
    std::vector<std::vector<int>> eval_gt(num_eval);
    std::mt19937 rng(random_seed * 6577 + omp_get_thread_num());
    GenRandom(rng, eval_points.data(), num_eval, nb);
    GenEvalGt(eval_points, eval_gt);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 1; iter <= iters; ++iter) {
      Join();
      Update();
      float recall = EvalRecall(eval_points, eval_gt);
      printf("NNDescent iter: [%d/%d], recall: %f\n", iter, iters, recall);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(t2 - t1).count();
    printf("NNDescent cost: %.2lfs\n", ela);
  }

  void Join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
    for (int u = 0; u < nb; u++) {
      graph[u].join([&](int i, int j) {
        if (i != j) {
          float dist = dist_func(data + i * d, data + j * d, d);
          graph[i].insert(j, dist);
          graph[j].insert(i, dist);
        }
      });
    }
  }

  void Update() {
#pragma omp parallel for
    for (int i = 0; i < nb; i++) {
      std::vector<int>().swap(graph[i].nn_new);
      std::vector<int>().swap(graph[i].nn_old);
    }
#pragma omp parallel for
    for (int n = 0; n < nb; ++n) {
      auto &nn = graph[n];
      std::sort(nn.pool.begin(), nn.pool.end());
      if ((int)nn.pool.size() > L) {
        nn.pool.resize(L);
      }
      nn.pool.reserve(L);
      int maxl = std::min(nn.M + S, (int)nn.pool.size());
      int c = 0;
      int l = 0;
      while ((l < maxl) && (c < S)) {
        if (nn.pool[l].flag)
          ++c;
        ++l;
      }
      nn.M = l;
    }
#pragma omp parallel
    {
      std::mt19937 rng(random_seed * 5081 + omp_get_thread_num());
#pragma omp for
      for (int n = 0; n < nb; ++n) {
        auto &node = graph[n];
        auto &nn_new = node.nn_new;
        auto &nn_old = node.nn_old;
        for (int l = 0; l < node.M; ++l) {
          auto &nn = node.pool[l];
          auto &other = graph[nn.id];
          if (nn.flag) {
            nn_new.push_back(nn.id);
            if (nn.distance > other.pool.back().distance) {
              std::scoped_lock guard(other.lock);
              if ((int)other.rnn_new.size() < R) {
                other.rnn_new.push_back(n);
              } else {
                int pos = rng() % R;
                other.rnn_new[pos] = n;
              }
            }
            nn.flag = false;
          } else {
            nn_old.push_back(nn.id);
            if (nn.distance > other.pool.back().distance) {
              std::scoped_lock guard(other.lock);
              if ((int)other.rnn_old.size() < R) {
                other.rnn_old.push_back(n);
              } else {
                int pos = rng() % R;
                other.rnn_old[pos] = n;
              }
            }
          }
        }
        std::make_heap(node.pool.begin(), node.pool.end());
      }
    }
#pragma omp parallel for
    for (int i = 0; i < nb; ++i) {
      auto &nn_new = graph[i].nn_new;
      auto &nn_old = graph[i].nn_old;
      auto &rnn_new = graph[i].rnn_new;
      auto &rnn_old = graph[i].rnn_old;
      nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
      nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
      if ((int)nn_old.size() > R * 2) {
        nn_old.resize(R * 2);
        nn_old.reserve(R * 2);
      }
      std::vector<int>().swap(graph[i].rnn_new);
      std::vector<int>().swap(graph[i].rnn_old);
    }
  }

  void GenEvalGt(const std::vector<int> &eval_set,
                 std::vector<std::vector<int>> &eval_gt) {
#pragma omp parallel for
    for (int i = 0; i < (int)eval_set.size(); i++) {
      std::vector<Neighbor> tmp;
      for (int j = 0; j < nb; j++) {
        if (eval_set[i] == j)
          continue;
        float dist = dist_func(data + eval_set[i] * d, data + j * d, d);
        tmp.push_back(Neighbor(j, dist, true));
      }
      std::partial_sort(tmp.begin(), tmp.begin() + K, tmp.end());
      for (int j = 0; j < K; j++) {
        eval_gt[i].push_back(tmp[j].id);
      }
    }
  }

  float EvalRecall(const std::vector<int> &eval_set,
                   const std::vector<std::vector<int>> &eval_gt) {
    float mean_acc = 0.0f;
    for (int i = 0; i < (int)eval_set.size(); i++) {
      float acc = 0;
      std::vector<Neighbor> &g = graph[eval_set[i]].pool;
      const std::vector<int> &v = eval_gt[i];
      for (int j = 0; j < (int)g.size(); j++) {
        for (int k = 0; k < (int)v.size(); k++) {
          if (g[j].id == v[k]) {
            acc++;
            break;
          }
        }
      }
      mean_acc += acc / v.size();
    }
    return mean_acc / eval_set.size();
  }
};

} // namespace glass