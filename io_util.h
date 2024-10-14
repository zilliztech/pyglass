//
// Created by weijian on 10/12/24.
//
#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <format>
#include <fstream>
#include <cassert>
#include <cstring>

#include "Eigen/Dense"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using namespace Eigen;

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

void read_dataset(std::string dataset, vector<float>& data, int& nb, vector<vector<float>>& queries, int& nq, int& d, vector<vector<int>>& GT) {
    string data_file = std::format("/data/raid0/{}/{}_base.fbin", dataset, dataset);
    string query_file = std::format("/data/raid0/{}/{}_query.fbin", dataset, dataset);
    string gt_file = std::format("/data/raid0/{}/{}_gt", dataset, dataset);
    {
        vector<vector<float>> tmp_data{};
        ReadBin(query_file, queries);
        ReadBin(gt_file, GT);
        ReadBin(data_file, tmp_data);
        nb = tmp_data.size();
        // load数据时直接按16维对齐，免的后面量化时再对齐
        int d_unalign = tmp_data.front().size();
        d = (d_unalign + 15) / 16 * 16;  // align
        nq = queries.size();
        data.resize(nb * d, 0.0f);
#pragma omp parallel for
        for (int i = 0; i < nb; i++) {
            std::memcpy(data.data() + i * d, tmp_data[i].data(), d_unalign * 4);
        }
    }
}


void read_dataset(std::string dataset, vector<float>& data, int& nb, vector<float>& queries, int& nq, int& d, vector<vector<int>>& GT) {
    string data_file = std::format("/data/raid0/{}/{}_base.fbin", dataset, dataset);
    string query_file = std::format("/data/raid0/{}/{}_query.fbin", dataset, dataset);
    string gt_file = std::format("/data/raid0/{}/{}_gt", dataset, dataset);
    {
        vector<vector<float>> tmp_queries{};
        vector<vector<float>> tmp_data{};
        ReadBin(query_file, tmp_queries);
        ReadBin(gt_file, GT);
        ReadBin(data_file, tmp_data);
        nb = tmp_data.size();
        // load数据时直接按16维对齐，免的后面量化时再对齐
        int d_unalign = tmp_data.front().size();
        d = (d_unalign + 15) / 16 * 16;  // align
        nq = tmp_queries.size();
        data.resize(nb * d, 0.0f);
        queries.resize(nq * d, 0.0f);
#pragma omp parallel for
        for (int i = 0; i < nb; i++) {
            std::memcpy(data.data() + i * d, tmp_data[i].data(), d_unalign * 4);
        }
#pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            std::memcpy(queries.data() + i * d, tmp_queries[i].data(), d_unalign * 4);
        }
    }
}

void read_dataset_rotated(std::string dataset, vector<float>& data, int& nb, vector<float>& queries, int& nq, int& d, vector<vector<int>>& GT) {//    {
    string data_file = std::format("/data/raid0/{}/{}_base.fbin", dataset, dataset);
    string query_file = std::format("/data/raid0/{}/{}_query.fbin", dataset, dataset);
    string gt_file = std::format("/data/raid0/{}/{}_gt", dataset, dataset);
    vector<vector<float>> tmp_data{};
    vector<vector<float>> tmp_queries{};
    ReadBin(query_file, tmp_queries);
    ReadBin(gt_file, GT);
    ReadBin(data_file, tmp_data);

    nb = tmp_data.size();
    int d_unalign = tmp_data.front().size();
    d = (d_unalign + 15) / 16 * 16;  // 对齐
    nq = tmp_queries.size();

    data.resize(nb * d, 0.0f);     // 初始化为0，避免未定义行为
    queries.resize(nq * d, 0.0f);  // 初始化为0，避免未定义行为

#pragma omp parallel for
    for (int i = 0; i < nb; i++) {
        std::memcpy(data.data() + i * d, tmp_data[i].data(), d_unalign * sizeof(float));
    }

#pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        std::memcpy(queries.data() + i * d, tmp_queries[i].data(), d_unalign * sizeof(float));
    }

    // 将data和queries转换为Eigen矩阵
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> eigenData(data.data(), nb, d);
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> eigenQueries(queries.data(), nq, d);

    // 生成一个正交矩阵
    MatrixXf randomMatrix = MatrixXf::Random(d, d);
    HouseholderQR<MatrixXf> qr(randomMatrix);
    MatrixXf orthogonalMatrix = qr.householderQ();

    // 对data和queries进行乘法
    eigenData = eigenData * orthogonalMatrix;
    eigenQueries = eigenQueries * orthogonalMatrix;

    // 结果已经保存在data和queries中，无需再进行memcpy
}