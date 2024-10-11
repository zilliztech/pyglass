#pragma once

#include "glass/common.hpp"
#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/simd/distance.hpp"

#include <cmath>

namespace glass {

    template <Metric metric, typename Reorderer = FP32Quantizer<metric>,
            int DIM = 0>
    struct TOQuantizer {
        using data_type = uint8_t;
        constexpr static int kAlign = 16;
        int d, d_align;
        int64_t code_size;
        data_type *codes = nullptr;
        float* centroid = nullptr;

        Reorderer reorderer;

        TOQuantizer() = default;

        explicit TOQuantizer(int dim)
                : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align / 2 + 2 * sizeof(float)),
                  reorderer(dim) {}

        ~TOQuantizer() { free(codes); free(centroid); }

        void train(const float *data, int n) {
            centroid = (float *)alloc64B(d_align * sizeof(float));

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < d; ++j) {
                    centroid[j] += data[i * d + j];
                }
            }
            for (int j = 0; j < d; ++j) {
                centroid[j] /= n;
            }
            codes = (data_type *)alloc2M(n * code_size);
            for (int i = 0; i < n; ++i) {
                float mx = -HUGE_VALF, mi = HUGE_VALF;
                for (int j = 0; j < d; ++j) {
                    float val = data[i * d + j] - centroid[j];
                    mx = std::max(mx, val);
                    mi = std::min(mi, val);
                }
                encode(data + i * d, get_data(i), mx, mi);
            }

            reorderer.train(data, n);
        }

        char *get_data(int u) const { return (char *)codes + u * code_size; }

        void encode(const float *from, char *to, float mx, float mi) const {
            float dif = mx - mi;
            *(float *)to = mi;
            *(float *)(to + sizeof(float)) = dif;
            to += 2 * sizeof(float);

            for (int j = 0; j < d; ++j) {
                float x = (from[j] - (centroid[j] + mi)) / dif;
                if (x < 0.0) {
                    x = 0.0;
                }
                if (x > 0.999) {
                    x = 0.999;
                }
                uint8_t y = 16 * x;

                int group_index = j / 16;
                int local_index = j % 16;

                if (local_index < 8) {
                    to[group_index * 8 + local_index] |= y;
                } else {
                    to[group_index * 8 + local_index - 8] |= y << 4;
                }
            }
        }

        template <typename Pool>
        void reorder(const Pool &pool, const float *q, int *dst, int k) const {
            int cap = pool.capacity();
            auto computer = reorderer.get_computer(q);
            searcher::MaxHeap<typename Reorderer::template Computer<0>::dist_type> heap(
                    k);
            for (int i = 0; i < cap; ++i) {
                if (i + 1 < cap) {
                    computer.prefetch(pool.id(i + 1), 1);
                }
                int id = pool.id(i);
                float dist = computer(id);
                heap.push(id, dist);
            }
            for (int i = 0; i < k; ++i) {
                dst[i] = heap.pop();
            }
        }

        template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
            using dist_type = float;
            constexpr static auto dist_func = L2SqrTO_ext;
            const TOQuantizer &quant;
            float *q;
            Computer(const TOQuantizer &quant, const float *query)
                    : quant(quant), q((float *)alloc64B(quant.d_align * 4)) {
                std::memcpy(q, query, quant.d * 4);
            }
            ~Computer() { free(q); }
            dist_type operator()(int u) const {
                return dist_func(q, (data_type *)quant.get_data(u), quant.centroid, quant.d_align);
            }

            void prefetch(int u, int lines) const {
                mem_prefetch(quant.get_data(u), lines);
            }
        };

        auto get_computer(const float *query) const {
            return Computer<0>(*this, query);
        }
    };

} // namespace glass
