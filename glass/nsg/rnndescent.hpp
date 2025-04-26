#pragma once

#include "glass/builder.hpp"
#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/nsg/nhood.hpp"
#include "glass/quant/quant.hpp"
#include "glass/utils.hpp"
#include <chrono>
#include <cstdio>
#include <mutex>
#include <omp.h>
#include <ostream>
#include <random>
#include <vector>

namespace glass {

template <typename ComputerType> struct RNNDescent {
  using storage_idx_t = int;

  using KNNGraph = std::vector<std::vector<FlagNeighbor>>;

  int32_t nb;
  KNNGraph graph;
  std::vector<std::mutex> locks;
  int32_t random_seed = 2021;

  int32_t T1 = 4;
  int32_t T2 = 15;
  int32_t degree_bound = 96;
  int32_t init_degree = 20;
  int32_t out_degree = 32;
  const ComputerType &computer;

  Graph<int32_t> out_graph;
  std::vector<int32_t> final_graph;
  std::vector<int32_t> offsets;

  explicit RNNDescent(int32_t nb, int32_t R, const ComputerType &computer)
      : nb(nb), out_degree(R), computer(computer) {}

  ~RNNDescent() = default;

  void Build() {
    auto st = std::chrono::high_resolution_clock::now();
    Init();
    for (int t1 = 0; t1 < T1; ++t1) {
      printf("Iter [%d/%d]:\n  ", t1 + 1, T1);
      fflush(stdout);
      for (int t2 = 0; t2 < T2; ++t2) {
        auto stt = std::chrono::high_resolution_clock::now();
        UpdateNeighbors();
        auto ed = std::chrono::high_resolution_clock::now();
        auto cost = std::chrono::duration<double>(ed - stt).count();
        printf("\titer [%2d/%2d] done, cost %.2fs\n", t2 + 1, T2, cost);
      }

      if (t1 != T1 - 1) {
        auto stt = std::chrono::high_resolution_clock::now();
        AddReverseEdges();
        auto ed = std::chrono::high_resolution_clock::now();
        auto cost = std::chrono::duration<double>(ed - stt).count();
        auto total_cost = std::chrono::duration<double>(ed - st).count();
        printf("\tAddReverseEdges done, cost %.2fs, total cost %.2fs\n", cost,
               total_cost);
      }
    }
    PostProcess();
  }

  void Init() {
    graph.resize(nb);
    locks = std::vector<std::mutex>(nb);

    std::mt19937 rng(random_seed);
#pragma omp parallel for
    for (int i = 0; i < nb; ++i) {
      std::vector<int> tmp(init_degree);
      GenRandom(rng, tmp.data(), init_degree, nb);
      for (int j = 0; j < init_degree; j++) {
        int id = tmp[j];
        if (id == i) {
          continue;
        }
        auto dist = computer(i, id);
        graph[i].emplace_back(id, dist, true);
      }
      std::sort(graph[i].begin(), graph[i].end());
    }
  }

  void insert_nn(int id, int nn_id, float distance, bool flag) {
    auto &pool = graph[id];
    {
      std::scoped_lock lk(locks[id]);
      pool.emplace_back(nn_id, distance, flag);
    }
  }

  void UpdateNeighbors() {

#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < nb; ++u) {
      auto &pool = graph[u];
      auto &mtx = locks[u];
      std::vector<FlagNeighbor> new_pool;
      std::vector<FlagNeighbor> old_pool;
      {
        std::scoped_lock lk(mtx);
        old_pool = pool;
        pool.clear();
      }
      std::sort(old_pool.begin(), old_pool.end());
      old_pool.erase(std::unique(old_pool.begin(), old_pool.end(),
                                 [](FlagNeighbor &a, FlagNeighbor &b) {
                                   return a.id == b.id;
                                 }),
                     old_pool.end());

      for (const auto &nn : old_pool) {
        bool ok = true;
        for (const auto &other_nn : new_pool) {
          if (!nn.flag && !other_nn.flag) {
            continue;
          }
          if (nn.id == other_nn.id) {
            ok = false;
            break;
          }
          float distance = computer(nn.id, other_nn.id);
          if (distance < nn.distance) {
            ok = false;
            insert_nn(other_nn.id, nn.id, distance, true);
            break;
          }
        }
        if (ok) {
          new_pool.emplace_back(nn);
        }
      }

      for (auto &&nn : new_pool) {
        nn.flag = false;
      }
      {
        std::scoped_lock lk(mtx);
        pool.insert(pool.end(), new_pool.begin(), new_pool.end());
      }
    }
  }

  void AddReverseEdges() {
    std::vector<std::vector<FlagNeighbor>> reverse_pools(nb);

#pragma omp parallel for
    for (int u = 0; u < nb; ++u) {
      for (const auto &nn : graph[u]) {
        std::scoped_lock lk(locks[nn.id]);
        reverse_pools[nn.id].emplace_back(u, nn.distance, nn.flag);
      }
    }

#pragma omp parallel for
    for (int u = 0; u < nb; ++u) {
      auto &pool = graph[u];
      for (auto &&nn : pool) {
        nn.flag = true;
      }
      auto &rpool = reverse_pools[u];
      rpool.insert(rpool.end(), pool.begin(), pool.end());
      pool.clear();
      std::sort(rpool.begin(), rpool.end());
      rpool.erase(std::unique(rpool.begin(), rpool.end(),
                              [](FlagNeighbor &a, FlagNeighbor &b) {
                                return a.id == b.id;
                              }),
                  rpool.end());
      if (rpool.size() > degree_bound) {
        rpool.resize(degree_bound);
      }
    }

#pragma omp parallel for
    for (int u = 0; u < nb; ++u) {
      for (const auto &nn : reverse_pools[u]) {
        std::scoped_lock lk(locks[nn.id]);
        graph[nn.id].emplace_back(u, nn.distance, nn.flag);
      }
    }

#pragma omp parallel for
    for (int u = 0; u < nb; ++u) {
      auto &pool = graph[u];
      std::sort(pool.begin(), pool.end());
      if (pool.size() > degree_bound) {
        pool.resize(degree_bound);
      }
    }
  }

  void PostProcess() {
#pragma omp parallel for
    for (int u = 0; u < nb; ++u) {
      auto &pool = graph[u];
      std::sort(pool.begin(), pool.end());
      pool.erase(std::unique(pool.begin(), pool.end(),
                             [](FlagNeighbor &a, FlagNeighbor &b) {
                               return a.id == b.id;
                             }),
                 pool.end());
    }

    offsets.resize(nb + 1);
    offsets[0] = 0;
    for (int u = 0; u < nb; ++u) {
      offsets[u + 1] = offsets[u] + graph[u].size();
    }

    out_graph.init(nb, out_degree);
    out_graph.eps = {rand() % nb};
#pragma omp parallel for
    for (int32_t u = 0; u < nb; ++u) {
      const auto &pool = graph[u];
      for (int32_t i = 0; i < std::min((int32_t)pool.size(), out_degree); ++i) {
        out_graph.at(u, i) = pool[i].id;
      }
    }

    KNNGraph().swap(graph);
  }

  Graph<int32_t> ToGraph() { return std::move(out_graph); }
};

template <SymComputableQuantConcept QuantType>
struct RNNDescentBuilder : public Builder {

  int32_t R;

  double construction_time;

  RNNDescentBuilder(int32_t R) : R(R) {}

  Graph<int32_t> Build(const float *data, int32_t N, int32_t dim) override {
    QuantType quant(dim);
    quant.train(data, N);
    quant.add(data, N);
    auto computer = quant.get_sym_computer();
    RNNDescent rnndescent(N, R, computer);
    {
      Timer timer("RNNDescent build");
      rnndescent.Build();
      construction_time =
          std::chrono::duration<double>(
              std::chrono::high_resolution_clock::now() - timer.start)
              .count();
    }
    return rnndescent.ToGraph();
  }

  double GetConstructionTime() const override { return construction_time; }
};

inline std::unique_ptr<Builder>
create_rnndescent(const std::string &quantizer = "SQ4U", int32_t R = 32) {
  auto qua = quantizer_map[quantizer];
  if (qua == QuantizerType::FP32) {
    return std::make_unique<RNNDescentBuilder<FP32Quantizer<Metric::L2>>>(R);
  }
  if (qua == QuantizerType::BF16) {
    return std::make_unique<RNNDescentBuilder<BF16Quantizer<Metric::L2>>>(R);
  }
  if (qua == QuantizerType::FP16) {
    return std::make_unique<RNNDescentBuilder<FP16Quantizer<Metric::L2>>>(R);
  }
  if (qua == QuantizerType::SQ8U) {
    return std::make_unique<RNNDescentBuilder<SQ8QuantizerUniform<Metric::L2>>>(
        R);
  }
  if (qua == QuantizerType::SQ4U) {
    return std::make_unique<RNNDescentBuilder<SQ4QuantizerUniform<Metric::L2>>>(
        R);
  }
  if (qua == QuantizerType::SQ2U) {
    return std::make_unique<RNNDescentBuilder<SQ2QuantizerUniform<Metric::L2>>>(
        R);
  }
  printf("Quantizer type %s not supported\n", quantizer.c_str());
  return nullptr;
}

} // namespace glass
