#pragma once

#include <atomic>
#include <random>
#include <stack>

#include "glass/builder.hpp"
#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/utils.hpp"
#include "nndescent.hpp"

namespace glass {

struct NSG : public Builder {
  int d;
  std::string metric;
  int R;
  int L;
  int C;
  int nb;
  float *data;
  int ep;
  Graph<int> final_graph;
  RandomGenerator rng; ///< random generator
  Dist<float, float, float> dist_func;
  int GK;
  int nndescent_S;
  int nndescent_R;
  int nndescent_L;
  int nndescent_iter;

  explicit NSG(int dim, const std::string &metric, int R = 32, int L = 200)
      : d(dim), metric(metric), R(R), L(L), rng(0x0903) {
    this->C = R + 100;
    srand(0x1998);
    if (metric == "L2") {
      dist_func = L2SqrRef;
    } else if (metric == "IP") {
      dist_func = IPRef;
    }
    this->GK = 64;
    this->nndescent_S = 10;
    this->nndescent_R = 100;
    this->nndescent_L = this->GK + 50;
    this->nndescent_iter = 10;
  }

  void Build(float *data, int n) override {
    this->nb = n;
    this->data = data;
    NNDescent nnd(d, metric);
    nnd.S = nndescent_S;
    nnd.R = nndescent_R;
    nnd.L = nndescent_L;
    nnd.iters = nndescent_iter;
    nnd.Build(data, n, GK);
    const auto &knng = nnd.final_graph;
    Init(knng);
    std::vector<int> degrees(n, 0);
    {
      Graph<Node> tmp_graph(n, R);
      link(knng, tmp_graph);
      final_graph.init(n, R);
      std::fill_n(final_graph.data, n * R, EMPTY_ID);
      final_graph.eps = {ep};
#pragma omp parallel for
      for (int i = 0; i < n; i++) {
        int cnt = 0;
        for (int j = 0; j < R; j++) {
          int id = tmp_graph.at(i, j).id;
          if (id != EMPTY_ID) {
            final_graph.at(i, cnt) = id;
            cnt += 1;
          }
          degrees[i] = cnt;
        }
      }
    }
    [[maybe_unused]] int num_attached = tree_grow(degrees);
    int max = 0, min = 1e6;
    double avg = 0;
    for (int i = 0; i < n; i++) {
      int size = 0;
      while (size < R && final_graph.at(i, size) != EMPTY_ID) {
        size += 1;
      }
      max = std::max(size, max);
      min = std::min(size, min);
      avg += size;
    }
    avg = avg / n;
    printf("Degree Statistics: Max = %d, Min = %d, Avg = %lf\n", max, min, avg);
  }

  Graph<int> GetGraph() override { return final_graph; }

  void Init(const Graph<int> &knng) {
    std::vector<float> center(d);
    for (int i = 0; i < d; ++i) {
      center[i] = 0.0;
    }
    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < d; j++) {
        center[j] += data[i * d + j];
      }
    }
    for (int i = 0; i < d; i++) {
      center[i] /= nb;
    }
    int ep_init = rng.rand_int(nb);
    std::vector<Neighbor> retset;
    std::vector<Node> tmpset;
    std::vector<bool> vis(nb);
    search_on_graph<false>(center.data(), knng, vis, ep_init, L, retset,
                           tmpset);
    // set enterpoint
    this->ep = retset[0].id;
  }

  template <bool collect_fullset>
  void search_on_graph(const float *q, const Graph<int> &graph,
                       std::vector<bool> &vis, int ep, int pool_size,
                       std::vector<Neighbor> &retset,
                       std::vector<Node> &fullset) const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);
    std::vector<int> init_ids(pool_size);
    int num_ids = 0;
    for (int i = 0; i < (int)init_ids.size() && i < graph.K; i++) {
      int id = (int)graph.at(ep, i);
      if (id < 0 || id >= nb) {
        continue;
      }
      init_ids[i] = id;
      vis[id] = true;
      num_ids += 1;
    }
    while (num_ids < pool_size) {
      int id = gen.rand_int(nb);
      if (vis[id]) {
        continue;
      }
      init_ids[num_ids] = id;
      num_ids++;
      vis[id] = true;
    }
    for (int i = 0; i < (int)init_ids.size(); i++) {
      int id = init_ids[i];
      float dist = dist_func(q, data + id * d, d);
      retset[i] = Neighbor(id, dist, true);
      if (collect_fullset) {
        fullset.emplace_back(retset[i].id, retset[i].distance);
      }
    }
    std::sort(retset.begin(), retset.begin() + pool_size);
    int k = 0;
    while (k < pool_size) {
      int updated_pos = pool_size;
      if (retset[k].flag) {
        retset[k].flag = false;
        int n = retset[k].id;
        for (int m = 0; m < graph.K; m++) {
          int id = (int)graph.at(n, m);
          if (id < 0 || id > nb || vis[id]) {
            continue;
          }
          vis[id] = true;
          float dist = dist_func(q, data + id * d, d);
          Neighbor nn(id, dist, true);
          if (collect_fullset) {
            fullset.emplace_back(id, dist);
          }
          if (dist >= retset[pool_size - 1].distance) {
            continue;
          }
          int r = insert_into_pool(retset.data(), pool_size, nn);
          updated_pos = std::min(updated_pos, r);
        }
      }
      k = (updated_pos <= k) ? updated_pos : (k + 1);
    }
  }

  void link(const Graph<int> &knng, Graph<Node> &graph) {
    auto st = std::chrono::high_resolution_clock::now();
    std::atomic<int> cnt{0};
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nb; i++) {
      std::vector<Node> pool;
      std::vector<Neighbor> tmp;
      std::vector<bool> vis(nb);
      search_on_graph<true>(data + i * d, knng, vis, ep, L, tmp, pool);
      sync_prune(i, pool, vis, knng, graph);
      pool.clear();
      tmp.clear();
      int cur = cnt += 1;
      if (cur % 10000 == 0) {
        printf("NSG building progress: [%d/%d]\n", cur, nb);
      }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    printf("NSG building cost: %.2lfs\n", ela);

    std::vector<std::mutex> locks(nb);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nb; ++i) {
      add_reverse_links(i, locks, graph);
    }
  }

  void sync_prune(int q, std::vector<Node> &pool, std::vector<bool> &vis,
                  const Graph<int> &knng, Graph<Node> &graph) {
    for (int i = 0; i < knng.K; i++) {
      int id = knng.at(q, i);
      if (id < 0 || id >= nb || vis[id]) {
        continue;
      }

      float dist = dist_func(data + q * d, data + id * d, d);
      pool.emplace_back(id, dist);
    }

    std::sort(pool.begin(), pool.end());

    std::vector<Node> result;

    int start = 0;
    if (pool[start].id == q) {
      start++;
    }
    result.push_back(pool[start]);

    while ((int)result.size() < R && (++start) < (int)pool.size() &&
           start < C) {
      auto &p = pool[start];
      bool occlude = false;
      for (int t = 0; t < (int)result.size(); t++) {
        if (p.id == result[t].id) {
          occlude = true;
          break;
        }

        float djk = dist_func(data + result[t].id * d, data + p.id * d, d);
        if (djk < p.distance /* dik */) {
          occlude = true;
          break;
        }
      }
      if (!occlude) {
        result.push_back(p);
      }
    }

    for (int i = 0; i < R; i++) {
      if (i < (int)result.size()) {
        graph.at(q, i).id = result[i].id;
        graph.at(q, i).distance = result[i].distance;
      } else {
        graph.at(q, i).id = EMPTY_ID;
      }
    }
  }

  void add_reverse_links(int q, std::vector<std::mutex> &locks,
                         Graph<Node> &graph) {
    for (int i = 0; i < R; i++) {
      if (graph.at(q, i).id == EMPTY_ID) {
        break;
      }

      Node sn(q, graph.at(q, i).distance);
      int des = graph.at(q, i).id;

      std::vector<Node> tmp_pool;
      int dup = 0;
      {
        LockGuard guard(locks[des]);
        for (int j = 0; j < R; j++) {
          if (graph.at(des, j).id == EMPTY_ID) {
            break;
          }
          if (q == graph.at(des, j).id) {
            dup = 1;
            break;
          }
          tmp_pool.push_back(graph.at(des, j));
        }
      }

      if (dup) {
        continue;
      }

      tmp_pool.push_back(sn);
      if ((int)tmp_pool.size() > R) {
        std::vector<Node> result;
        int start = 0;
        std::sort(tmp_pool.begin(), tmp_pool.end());
        result.push_back(tmp_pool[start]);

        while ((int)result.size() < R && (++start) < (int)tmp_pool.size()) {
          auto &p = tmp_pool[start];
          bool occlude = false;
          for (int t = 0; t < (int)result.size(); t++) {
            if (p.id == result[t].id) {
              occlude = true;
              break;
            }
            float djk = dist_func(data + result[t].id * d, data + p.id * d, d);
            if (djk < p.distance /* dik */) {
              occlude = true;
              break;
            }
          }
          if (!occlude) {
            result.push_back(p);
          }
        }

        {
          LockGuard guard(locks[des]);
          for (int t = 0; t < (int)result.size(); t++) {
            graph.at(des, t) = result[t];
          }
        }

      } else {
        LockGuard guard(locks[des]);
        for (int t = 0; t < R; t++) {
          if (graph.at(des, t).id == EMPTY_ID) {
            graph.at(des, t) = sn;
            break;
          }
        }
      }
    }
  }

  int tree_grow(std::vector<int> &degrees) {
    int root = ep;
    std::vector<bool> vis(nb);
    int num_attached = 0;
    int cnt = 0;
    while (true) {
      cnt = dfs(vis, root, cnt);
      if (cnt >= nb) {
        break;
      }
      std::vector<bool> vis2(nb);
      root = attach_unlinked(vis, vis2, degrees);
      num_attached += 1;
    }
    return num_attached;
  }

  int dfs(std::vector<bool> &vis, int root, int cnt) const {
    int node = root;
    std::stack<int> stack;
    stack.push(root);
    if (vis[root]) {
      cnt++;
    }
    vis[root] = true;
    while (!stack.empty()) {
      int next = EMPTY_ID;
      for (int i = 0; i < R; i++) {
        int id = final_graph.at(node, i);
        if (id != EMPTY_ID && !vis[id]) {
          next = id;
          break;
        }
      }
      if (next == EMPTY_ID) {
        stack.pop();
        if (stack.empty()) {
          break;
        }
        node = stack.top();
        continue;
      }
      node = next;
      vis[node] = true;
      stack.push(node);
      cnt++;
    }
    return cnt;
  }

  int attach_unlinked(std::vector<bool> &vis, std::vector<bool> &vis2,
                      std::vector<int> &degrees) {
    int id = EMPTY_ID;
    for (int i = 0; i < nb; i++) {
      if (vis[i]) {
        id = i;
        break;
      }
    }
    if (id == EMPTY_ID) {
      return EMPTY_ID;
    }
    std::vector<Neighbor> tmp;
    std::vector<Node> pool;
    search_on_graph<true>(data + id * d, final_graph, vis2, ep, L, tmp, pool);
    std::sort(pool.begin(), pool.end());
    int node;
    bool found = false;
    for (int i = 0; i < (int)pool.size(); i++) {
      node = pool[i].id;
      if (degrees[node] < R && node != id) {
        found = true;
        break;
      }
    }
    if (!found) {
      do {
        node = rng.rand_int(nb);
        if (vis[node] && degrees[node] < R && node != id) {
          found = true;
        }
      } while (!found);
    }
    int pos = degrees[node];
    final_graph.at(node, pos) = id;
    degrees[node] += 1;
    return node;
  }
};

} // namespace glass