//
// Created by weijian on 10/8/24.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <string>
#include <cstring>
#include <set>
#include <format>
#include <filesystem>


#include "glass/searcher.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/quant/sq8_quant.hpp"
#include "glass/quant/sq4_quant.hpp"
#include "glass/nsg/nsg.hpp"
#include "glass/hnsw/hnsw.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;
namespace fs = std::filesystem;


/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
template<class T>
void ReadBin(const std::string &file_path,
             std::vector<std::vector<T>> &data) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    unsigned N;  // num of points
    unsigned num_dimensions;
    ifs.read((char *)&N, sizeof(unsigned));
    ifs.read((char *)&num_dimensions, sizeof(unsigned));
    data.resize(N);
    std::cout << "# of points: " << N << std::endl;
    std::cout << "# of dimensions: " << num_dimensions << std::endl;
    std::vector<T> buff(num_dimensions);
    int counter = 0;
    for(int i = 0; i < N; i++) {
        ifs.read((char *)buff.data(), num_dimensions * sizeof(T));
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
    cout << "Explore whether it is feasible to use sq4 for image search and then rerank?" << endl;
    string dataset = "sift1m";
    string data_file = std::format("/data/raid0/{}/{}_base.fbin", dataset, dataset);
    string query_file = std::format("/data/raid0/{}/{}_query.fbin", dataset, dataset);
    string gt_file = std::format("/data/raid0/{}/{}_gt", dataset, dataset);
    int R = 64, L = 128;
    int nb = 0, d = 0;
    int nq = 0;
    int k = 10;
    string index_path = std::format("/data/raid0/{}/{}_R{}_L{}", dataset, dataset, R, L);

    bool update_index = !fs::exists(index_path);

    vector<float> data{};
    vector<vector<float>> queries{};
    vector<vector<int>> GT{};
    {
        vector<vector<float>> tmp_data{};
        ReadBin(query_file, queries);
        ReadBin(gt_file, GT);
        ReadBin(data_file, tmp_data);
        nb = tmp_data.size(); d = tmp_data.front().size();
        nq = queries.size();
        data.resize(nb * d);
#pragma omp parallel for
        for (int i = 0; i < nb; i++) {
            std::memcpy(data.data() + i * d, tmp_data[i].data(), d * 4);
        }
    }
    glass::HNSW index(d, "L2", R, L);
    if (update_index) {
        index.Build(data.data(), nb);
        index.final_graph.save(index_path);
    } else {
        index.final_graph.load(index_path);
    }

    glass::Searcher<glass::SQ8Quantizer<glass::Metric::L2>> searcher(index.final_graph);
    searcher.SetData(data.data(), nb, d);
    glass::FP32Quantizer<glass::Metric::L2> raw_data(d);
    raw_data.train(data.data(), nb);

//    std::vector<int> efs{10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500};
    std::vector<int> efs{150};
    for (auto ef : efs) {
        searcher.SetEf(ef);
        searcher.Optimize(8);
        vector<vector<int>> output(nq, vector<int>(k, 0));

        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(8)
        for (int i = 0; i < nq; i++) {
            searcher.Search(queries[i].data(), k, output[i].data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        cout << "search time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

        std::atomic<int> total_coselection{0};
        std::atomic<int> total_num{0};
        std::vector<double> recalls{};
//#pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            int cur_coselection = 0;
            std::set gt(GT[i].begin(), GT[i].begin() + k);
            std::set res(output[i].begin(), output[i].begin() + k);
            for (auto item: res) {
                if (gt.find(static_cast<int64_t>(item)) != gt.end()) {
                    cur_coselection++;
                }
            }
            total_num += 1;
            total_coselection += cur_coselection;
            recalls.push_back((double)cur_coselection * 100 / k);
        }

        std::cout << "R = " << R << ", L = " << L << ", ef = " << ef << ", recall = " << (double) total_coselection * 100 / (total_num * k) << " %" << std::endl;
    }
}