//
// Created by weijian on 5/13/24.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <string>
#include <cstring>
#include <set>

#include "glass/searcher.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/hnsw/hnsw.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;

///// @brief Reading binary data vectors. Raw data store as a (N x dim)
///// @param file_path file path of binary data
///// @param data returned 2D data vectors
//template<class T>
//void ReadBin(const std::string &file_path,
//             std::vector<std::vector<T>> &data) {
//    std::cout << "Reading Data: " << file_path << std::endl;
//    std::ifstream ifs;
//    ifs.open(file_path, std::ios::binary);
//    assert(ifs.is_open());
//    unsigned N;  // num of points
//    unsigned num_dimensions;
//    ifs.read((char *)&N, sizeof(unsigned));
//    ifs.read((char *)&num_dimensions, sizeof(unsigned));
//    data.resize(N);
//    std::cout << "# of points: " << N << std::endl;
//    std::cout << "# of dimensions: " << num_dimensions << std::endl;
//    std::vector<T> buff(num_dimensions);
//    int counter = 0;
//    while (ifs.read((char *)buff.data(), num_dimensions * sizeof(T))) {
//        std::vector<T> row(num_dimensions);
//        for (int d = 0; d < num_dimensions; d++) {
//            row[d] = static_cast<T>(buff[d]);
//        }
//        data[counter++] = std::move(row);
//    }
//    ifs.close();
//    std::cout << "Finish Reading Data" << endl;
//}

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
template<class T>
void ReadBin(const std::string &file_path,
             const int num_dimensions,
             std::vector<std::vector<T>> &data) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    unsigned N;  // num of points
    ifs.read((char *)&N, sizeof(unsigned));
    data.resize(N);
    std::cout << "# of points: " << N << std::endl;
    std::vector<T> buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char *)buff.data(), num_dimensions * sizeof(T))) {
        std::vector<T> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++) {
            row[d] = static_cast<T>(buff[d]);
        }
        data[counter++] = std::move(row);
    }
    ifs.close();
    std::cout << "Finish Reading Data" << endl;
}

int main() {
//    string data_file = "/data/deep1b/base.1B.fbin.crop_nb_10000000";
//    string query_file = "/data/deep1b/query.public.10K.fbin";
//    string gt_file = "/data/deep1b/deep-10M";
    const string data_file = "/dataset/sigmod2024/medium/contest-data-release-1m.bin",
            query_file = "/dataset/sigmod2024/medium/contest-queries-release-1m.bin",
            gt_file = "/dataset/sigmod2024/medium/contest-gt-release-1m.bin";
//    const string data_file = "/dataset/sigmod2024/large/contest-data-release-10m.bin",
//            query_file = "/dataset/sigmod2024/large/contest-queries-release-10m.bin",
//            gt_file = "/dataset/sigmod2024/large/contest-gt-release-10m.bin";

    vector<vector<float>> data{}, queries{};
    vector<vector<int>> GT{};

    vector<float> data_buf{};

    ReadBin(data_file, 102, data);
    ReadBin(query_file, 104, queries);
    ReadBin(gt_file, 100, GT);

    int nb = data.size(), d = 100;
    int nq = queries.size();
    int k = 100;
    vector<vector<int>> output(nq, vector<int>(k, 0));

    bool update_index = true;

    cout << nb << ", " << d << endl;
    cout << nq << endl;
    cout << nq << ", " << queries.front().size() << endl;
    cout << output.size() << ", " << output.front().size() << endl;
    data_buf.resize(nb * d);
#pragma omp parallel for
    for(int i = 0; i < nb; i++) {
        std::memcpy(data_buf.data() + i * 100, data[i].data() + 2, d * 4);
    }

    glass::HNSW index(d, "L2");
    if(update_index) {
        index.Build(data_buf.data(), nb);
        index.final_graph.save("hnsw_index_glass");
    } else {
        index.final_graph.load("hnsw_index_glass");
    }
    glass::Searcher<glass::FP32Quantizer<glass::Metric::L2>> searcher(index.final_graph);
    searcher.SetData(data_buf.data(), nb, d);
    searcher.SetEf(500);
    searcher.Optimize(96);
    cout << "11111" << endl;

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < nq; i++) {
        if(queries[i][0] == 0) {
            searcher.Search(queries[i].data() + 4, k, output[i].data());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    cout << "search time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    std::atomic<int> total_coselection{0};
    std::atomic<int> total_num{0};
#pragma omp parallel for
//    int total_coselection = 0;
//    int total_num = 0;
    for (int i = 0; i < nq; i++) {
        if(queries[i][0] == 0) {
            int cur_coselection = 0;
            std::set gt(GT[i].begin(), GT[i].end());
            std::set res(output[i].begin(), output[i].end());
            for (auto item: res) {
                if (gt.find(item) != gt.end()) {
                    cur_coselection++;
                }
            }
            total_num += 1;
            total_coselection += cur_coselection;
        }
    }

    std::cout << "recall = " << (double) total_coselection * 100 / (total_num * 100) << " %" << std::endl;
}