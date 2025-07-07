#pragma once

#include "glass/graph.hpp"

namespace glass {

inline void print_degree_statistic(const GraphConcept auto &graph) {
    int n = graph.size();
    int r = graph.range();
    int max = 0, min = 1e6;
    double avg = 0;
    for (int i = 0; i < n; i++) {
        int size = 0;
        while (size < r && graph.at(i, size) != graph.EMPTY_ID) {
            size += 1;
        }
        max = std::max(size, max);
        min = std::min(size, min);
        avg += size;
    }
    avg = avg / n;
    printf("Degree Statistics: Range = %d, Max = %d, Min = %d, Avg = %lf\n", r, max, min, avg);
}

}  // namespace glass
