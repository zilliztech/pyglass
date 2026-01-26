#pragma once

#include <cstdint>

#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/computer.hpp"

namespace glass {

struct PrefetchParams {
    size_t po = 1;
    size_t pl = 1;
};

void GraphSearchImpl(const GraphConcept auto &graph, NeighborPoolConcept auto &pool,
                     const ComputerConcept auto &computer, PrefetchParams prefetch_params) {
    alignas(64) int32_t edge_buf[graph.K];
    while (pool.has_next()) {
        auto u = pool.pop();
        size_t edge_size = 0;
        for (int32_t i = 0; i < graph.K; ++i) {
            int32_t v = graph.at(u, i);
            if (v == -1) {
                break;
            }
            if (pool.check_visited(v)) {
                continue;
            }
            pool.set_visited(v);
            edge_buf[edge_size++] = v;
        }
        for (size_t i = 0; i < std::min(prefetch_params.po, edge_size); ++i) {
            computer.prefetch(edge_buf[i], prefetch_params.pl);
        }
        for (size_t i = 0; i < edge_size; ++i) {
            if (i + prefetch_params.po < edge_size) {
                computer.prefetch(edge_buf[i + prefetch_params.po], prefetch_params.pl);
            }
            auto v = edge_buf[i];
            auto cur_dist = computer(v);
            pool.insert(v, cur_dist);
        }
    }
}

}  // namespace glass
