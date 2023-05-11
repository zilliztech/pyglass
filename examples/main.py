import numpy as np
from time import time
from sklearn import preprocessing
import matplotlib.pyplot as plt
import psutil
import os
import json

import glassppy as glass
from datasets import dataset_dict


class Glass:
    def __init__(self, name, level, metric, method_param):
        self.metric = metric
        self.R = method_param['R']
        self.L = method_param['L']
        self.index_type = method_param['index_type']
        self.optimize = method_param['optimize']
        self.batch = method_param['batch']
        self.name = 'glass_(%s)' % (method_param)
        self.dir = 'indices'
        self.path = f'{name}_{self.index_type}_R_{self.R}_L_{self.L}.glass'
        self.level = level

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[1]
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if self.path not in os.listdir(self.dir):
            p = glass.Index(self.index_type, dim=self.d,
                            metric=self.metric, R=self.R, L=self.L)
            g = p.build(X)
            g.save(os.path.join(self.dir, self.path))
            del p
            del g
        g = glass.Graph(os.path.join(self.dir, self.path))
        self.searcher = glass.Searcher(g, X, self.metric, self.level)
        if self.optimize:
            if batch:
                self.searcher.optimize()
            else:
                self.searcher.optimize(1)

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)
        self.ef = ef

    def prepare_query(self, q, n):
        if self.metric == 'IP':
            q = q / np.linalg.norm(q)
        self.q = q
        self.n = n

    def run_prepared_query(self):
        self.res = self.searcher.search(
            self.q, self.n)

    def get_prepared_query_results(self):
        return self.res

    def prepare_batch_query(self, X, n):
        if self.metric == 'angular':
            X = preprocessing.normalize(X, axis=1, norm='l2')
        self.queries = X
        self.n = n
        self.nq = len(X)

    def run_batch_query(self):
        self.result = self.searcher.batch_search(
            self.queries, self.n)

    def get_batch_results(self):
        return self.result.reshape(self.nq, -1)

    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024 / 1024

    def freeIndex(self):
        del self.searcher


if __name__ == '__main__':
    plt.style.use('seaborn-whitegrid')

    with open('examples/config.json', 'r') as f:
        config = json.load(f)

    index_types = config['index_types']
    levels = config['levels']
    efs = config['efs']
    R = config['R']
    L = config['L']
    topk = config['topk']
    runs = config['runs']
    batch = config['batch']
    optimize = config['optimize']
    dataset_names = config['datasets']

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    for dataset_name in dataset_names:
        plt.figure(figsize=(10, 6), dpi=120)
        dataset = dataset_dict[dataset_name]()
        base = dataset.get_base()
        query = dataset.get_queries()
        gt = dataset.get_groundtruth(topk)
        name = dataset.name
        metric = dataset.metric
        nq = dataset.nq
        for it, index_type in enumerate(index_types):
            for level in levels:
                p = Glass(name, level, metric, {
                    'index_type': index_type, 'R': R, 'L': L, 'optimize': optimize, 'batch': batch})
                t = time()
                p.fit(base)
                ela = time() - t
                print(f"Building time of index: {ela}s")
                print(f"Memory usage of index: {p.get_memory_usage()}MB")
                qpss = []
                recalls = []
                print(
                    f"dataset: {name}, index: {index_type}, level: {level}")
                for ef in efs:
                    print(f"  ef: {ef}")
                    p.set_query_arguments(ef)
                    mx_qps = 0.0
                    for run in range(runs):
                        T = 0
                        res = np.zeros_like(gt)
                        if batch:
                            p.prepare_batch_query(query, topk)
                            t = time()
                            p.run_batch_query()
                            T = time() - t
                            res = p.get_batch_results()
                        else:
                            for i in range(nq):
                                p.prepare_query(query[i], topk)
                                t = time()
                                p.run_prepared_query()
                                T += time() - t
                                res[i] = p.get_prepared_query_results()
                        cnt = 0
                        for i in range(nq):
                            cnt += np.intersect1d(res[i], gt[i]).size
                        recall = cnt / nq / 10
                        qps = nq / T
                        print(
                            f"    runs [{run + 1}/{runs}], recall: {recall:.4f}, qps: {qps:.2f}")
                        mx_qps = max(mx_qps, qps)
                    qpss.append(mx_qps)
                    recalls.append(recall)
                plt.plot(recalls, qpss,
                         label=f'{index_type} + level{level}', marker=['x', 'o'][it])
        plt.xlabel(f'Recall@{topk}')
        plt.ylabel("QPS")
        plt.legend()
        plt.title(f'{name}')
        plt.savefig(os.path.join(
            results_dir, f'{name}_R{R}_L{L}_batch{batch}_optimize{optimize}_top{topk}.png'))
