#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "glass/hnsw/HNSWInitializer.hpp"
#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/computer.hpp"
#include "glass/simd/prefetch.hpp"

namespace glass {

template <typename GraphType>
concept GraphConcept = requires(GraphType graph, int32_t i, int32_t j) {
  graph.size();
  graph.range();
  graph.at(i, j);
  graph.edges(i);
  graph.degree(i);
};

template <typename node_t> struct Graph {
  int32_t N;
  int32_t K;

  constexpr static int EMPTY_ID = -1;

  node_t *data = nullptr;

  std::unique_ptr<HNSWInitializer> initializer = nullptr;

  std::vector<int> eps;

  Graph() = default;

  Graph(node_t *edges, int32_t N, int32_t K) : N(N), K(K), data(edges) {}

  Graph(int32_t N, int32_t K)
      : N(N), K(K),
        data((node_t *)align_alloc((size_t)N * K * sizeof(node_t), true, -1)) {}

  Graph(const Graph &g) = delete;

  Graph(Graph &&g) { swap(*this, g); }

  Graph &operator=(const Graph &rhs) = delete;

  Graph &operator=(Graph &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  friend void swap(Graph &lhs, Graph &rhs) {
    using std::swap;
    swap(lhs.N, rhs.N);
    swap(lhs.K, rhs.K);
    swap(lhs.data, rhs.data);
    swap(lhs.initializer, rhs.initializer);
    swap(lhs.eps, rhs.eps);
  }

  void init(int32_t N, int K) {
    data = (node_t *)align_alloc((size_t)N * K * sizeof(node_t), true, -1);
    this->K = K;
    this->N = N;
  }

  ~Graph() { free(data); }

  int32_t size() const { return N; }

  int32_t range() const { return K; }

  const int *edges(int32_t u) const { return data + (int64_t)K * u; }

  int *edges(int32_t u) { return data + (int64_t)K * u; }

  node_t at(int32_t i, int32_t j) const { return data[(int64_t)i * K + j]; }

  node_t &at(int32_t i, int32_t j) { return data[(int64_t)i * K + j]; }

  int32_t degree(int32_t i) const {
    int32_t deg = 0;
    while (deg < range() && at(i, deg) != EMPTY_ID) {
      ++deg;
    }
    return deg;
  }

  void prefetch(int32_t u, int32_t lines) const {
    mem_prefetch((char *)edges(u), lines);
  }

  void initialize_search(NeighborPoolConcept auto &pool,
                         const ComputerConcept auto &computer) const {
    if (initializer) {
      initializer->initialize(pool, computer);
    } else {
      for (auto ep : eps) {
        pool.insert(ep, computer(ep));
        pool.set_visited(ep);
      }
    }
  }

  void save(const std::string &filename) const {
    static_assert(std::is_same_v<node_t, int32_t>);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    int nep = eps.size();
    writer.write((char *)&nep, 4);
    writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, (int64_t)N * K * 4);
    if (initializer) {
      initializer->save(writer);
    }
    printf("Graph Saving done\n");
  }

  void load(const std::string &filename, const std::string &format) {
    if (format == "glass") {
      load(filename);
    } else if (format == "diskann") {
      load_diskann(filename);
    } else if (format == "hnswlib") {
      load_hnswlib(filename);
    } else if (format == "nsg") {
      load_nsg(filename);
    } else {
      printf("Unknown graph format\n");
      exit(-1);
    }
  }

  void load(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    int nep;
    reader.read((char *)&nep, 4);
    eps.resize(nep);
    reader.read((char *)eps.data(), nep * 4);
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    data = (node_t *)align_alloc((int64_t)N * K * 4, true, -1);
    reader.read((char *)data, N * K * 4);
    if (reader.peek() != EOF) {
      initializer = std::make_unique<HNSWInitializer>(N);
      initializer->load(reader);
    }
    printf("Graph Loding done\n");
  }

  void load_diskann(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    size_t size;
    reader.read((char *)&size, 8);
    reader.read((char *)&K, 4);
    eps.resize(1);

    reader.read((char *)&eps[0], 4);
    size_t x;
    reader.read((char *)&x, 8);
    N = 0;
    while (reader.tellg() < size) {
      N++;
      int32_t cur_k;
      reader.read((char *)&cur_k, 4);
      reader.seekg(cur_k * 4, reader.cur);
    }
    reader.seekg(24, reader.beg);
    data = (node_t *)align_alloc((int64_t)N * K * 4, true, -1);
    memset(data, -1, (int64_t)N * K * 4);
    for (int i = 0; i < N; ++i) {
      int cur_k;
      reader.read((char *)&cur_k, 4);
      reader.read((char *)edges(i), 4 * cur_k);
    }
  }

  void load_hnswlib(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    reader.seekg(8, std::ios::cur);
    size_t max_elements;
    reader.read((char *)&max_elements, 8);
    N = max_elements;

    reader.seekg(8, std::ios::cur);
    size_t size_per_element;
    reader.read((char *)&size_per_element, 8);
    reader.seekg(16, std::ios::cur);
    int32_t max_level;
    reader.read((char *)&max_level, 4);
    if (max_level > 1) {
      printf("Not supported\n"); // TODO: support multilevel hnsw
      exit(-1);
    }
    eps.resize(1);
    reader.read((char *)&eps[0], 4);
    reader.seekg(8, std::ios::cur);
    size_t maxM0;
    reader.read((char *)&maxM0, 8);
    K = maxM0;
    reader.seekg(24, std::ios::cur);
    data = (node_t *)align_alloc((int64_t)N * K * 4, true, -1);
    for (int i = 0; i < N; ++i) {
      std::vector<char> buf(size_per_element);
      reader.read(buf.data(), size_per_element);
      int *lst = (int *)buf.data();
      int k = lst[0];
      memcpy(edges(i), lst + 1, k * 4);
    }
  }

  void load_nsg(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    reader.seekg(0, reader.end);
    size_t size = reader.tellg();
    reader.seekg(0, reader.beg);
    reader.read((char *)&K, 4);
    eps.resize(1);
    reader.read((char *)&eps[0], 4);
    N = 0;
    while (reader.tellg() < size) {
      N++;
      int32_t cur_k;
      reader.read((char *)&cur_k, 4);
      reader.seekg(cur_k * 4, reader.cur);
    }
    data = (node_t *)align_alloc((int64_t)N * K * 4, true, -1);
    reader.seekg(8, reader.beg);
    memset(data, -1, (int64_t)N * K * 4);
    for (int i = 0; i < N; ++i) {
      int cur_k;
      reader.read((char *)&cur_k, 4);
      reader.read((char *)edges(i), 4 * cur_k);
    }
  }
};

} // namespace glass
