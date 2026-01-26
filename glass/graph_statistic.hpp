#pragma once

#include <map>

#include "glass/graph.hpp"

namespace glass {

inline void print_degree_statistic(const GraphConcept auto& graph) {
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

inline void print_distance_statistics(const GraphConcept auto& graph) {
    int n = graph.size();
    int r = graph.range();
    auto bfs = [&](int s) -> std::vector<int> {
        std::vector<int> dist(n, -1);
        std::queue<int> q;
        q.push(s);
        dist[s] = 0;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int i = 0; i < r; ++i) {
                int v = graph.at(u, i);
                if (v == -1) {
                    break;
                }
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        return dist;
    };
    std::vector<int> dist = bfs(graph.eps[0]);
    std::map<int, int> cnt;
    for (int i = 0; i < n; ++i) {
        cnt[dist[i]] += 1;
    }
    printf("Graph Distance Statistics:\n");
    for (auto [d, c] : cnt) {
        printf("\t%d: %d\n", d, c);
    }
}

inline void print_graph_statistic(const GraphConcept auto& graph) {
    print_degree_statistic(graph);
    print_distance_statistics(graph);
}

}  // namespace glass
