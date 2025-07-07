#pragma once

#include <cmath>
#include <queue>

#include "glass/neighbor.hpp"
#include "glass/utils.hpp"

namespace glass {

inline float limit_range(float x) {
    if (x < 0.0f) {
        x = 0.0f;
    }
    if (x > 1.0f) {
        x = 1.0f;
    }
    return x;
}

inline float limit_range_sym(float x) {
    if (x < -1.0f) {
        x = -1.0f;
    }
    if (x > 1.0f) {
        x = 1.0f;
    }
    return x;
}

inline std::pair<float, float> find_minmax(const float *data, int64_t nitems, float ratio = 0.0f) {
    size_t top = int64_t(nitems * ratio) + 1;
    MaxHeap<float> max_heap(top), min_heap(top);
    for (int64_t i = 0; i < nitems; ++i) {
        float x = data[i];
        min_heap.push(x);
        max_heap.push(-x);
    }
    return {min_heap.top(), -max_heap.top()};
}

inline float find_absmax(const float *data, int64_t nitems, float ratio = 0.0f) {
    size_t top = int64_t(nitems * ratio) + 1;
    MaxHeap<float> heap(top);
    for (int64_t i = 0; i < nitems; ++i) {
        float x = std::abs(data[i]);
        heap.push(-x);
    }
    return -heap.top();
}

inline void find_minmax_perdim(std::vector<float> &mins, std::vector<float> &maxs, const float *data, int32_t n,
                               int32_t d, float ratio = 0.0f) {
    int64_t top = (int64_t)n * ratio + 1;
    std::vector<MaxHeap<float>> max_heaps(d, MaxHeap<float>(top));
    std::vector<MaxHeap<float>> min_heaps(d, MaxHeap<float>(top));
    for (int64_t i = 0; i < (int64_t)n * d; ++i) {
        auto &max_heap = max_heaps[i / n];
        auto &min_heap = min_heaps[i / n];
        float x = data[i];
        max_heap.push(-x);
        min_heap.push(x);
    }
    mins.resize(d);
    maxs.resize(d);
    for (int32_t i = 0; i < d; ++i) {
        mins[i] = min_heaps[i].top();
        maxs[i] = -max_heaps[i].top();
    }
}

}  // namespace glass