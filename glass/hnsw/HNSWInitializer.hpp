#pragma once

#include <cstdlib>
#include <fstream>
#include <vector>

#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/computer.hpp"

namespace glass {

struct HNSWInitializer {
    int32_t N, K;
    int ep;
    std::vector<int> levels;
    std::vector<int *> lists;
    HNSWInitializer() = default;

    explicit HNSWInitializer(int32_t n, int32_t K = 0) : N(n), K(K), levels(n), lists(n) {}

    HNSWInitializer(const HNSWInitializer &rhs) = delete;

    HNSWInitializer(HNSWInitializer &&rhs) = default;

    ~HNSWInitializer() {
        for (auto &p : lists) {
            free(p);
            p = nullptr;
        }
    }

    int at(int32_t level, int32_t u, int32_t i) const { return lists[u][(level - 1) * K + i]; }

    int &at(int32_t level, int32_t u, int32_t i) { return lists[u][(level - 1) * K + i]; }

    const int *edges(int32_t level, int32_t u) const { return lists[u] + (level - 1) * K; }

    int *edges(int32_t level, int32_t u) { return lists[u] + (level - 1) * K; }

    void initialize(NeighborPoolConcept auto &pool, const ComputerConcept auto &computer) const {
        int u = ep;
        auto cur_dist = computer(u);
        for (int32_t level = levels[u]; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const int32_t *list = edges(level, u);
                for (int32_t i = 0; i < K && list[i] != -1; ++i) {
                    int32_t v = list[i];
                    auto dist = computer(v);
                    if (dist < cur_dist) {
                        cur_dist = dist;
                        u = v;
                        changed = true;
                    }
                }
            }
        }
        pool.insert(u, cur_dist);
        pool.set_visited(u);
    }

    void load(std::ifstream &reader) {
        reader.read((char *)&N, 4);
        reader.read((char *)&K, 4);
        reader.read((char *)&ep, 4);
        for (int i = 0; i < N; ++i) {
            int cur;
            reader.read((char *)&cur, 4);
            levels[i] = cur / K;
            lists[i] = (int *)align_alloc(cur * 4, true, -1);
            reader.read((char *)lists[i], cur * 4);
        }
    }

    void save(std::ofstream &writer) const {
        writer.write((char *)&N, 4);
        writer.write((char *)&K, 4);
        writer.write((char *)&ep, 4);
        for (int i = 0; i < N; ++i) {
            int cur = levels[i] * K;
            writer.write((char *)&cur, 4);
            writer.write((char *)lists[i], cur * 4);
        }
    }
};

}  // namespace glass