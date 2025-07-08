import numpy as np
from time import time
import matplotlib.pyplot as plt
import psutil
import os
import yaml
import sys

import glass
from ann_dataset import dataset_dict


class Glass:
    def __init__(
        self,
        name,
        build_quant,
        search_quant,
        refine_quant,
        metric,
        rebuild,
        method_param,
    ):
        self.metric = metric
        self.R = method_param["R"]
        self.L = method_param["L"]
        self.index_type = method_param["index_type"]
        self.batch = method_param["batch"]
        self.name = "glass_(%s)" % (method_param)
        self.dir = "indices"
        self.build_quant = build_quant
        self.search_quant = search_quant
        self.refine_quant = refine_quant
        self.rebuild = rebuild
        self.path = (
            f"{name}_{self.index_type}_{self.build_quant}_R{self.R}_L{self.L}.glass"
        )

    def fit(self, X):
        self.d = X.shape[1]
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if rebuild or self.path not in os.listdir(self.dir):
            glass.build_graph(
                self.index_type,
                X,
                os.path.join(self.dir, self.path),
                metric=self.metric,
                quant=self.build_quant,
                R=self.R,
                L=self.L,
            )
        g = glass.Graph(os.path.join(self.dir, self.path))
        self.searcher = glass.Searcher(
            g, X, self.metric, self.search_quant, self.refine_quant
        )
        self.searcher.enable_stats(True)
        self.searcher.optimize()

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)
        self.ef = ef

    def prepare_query(self, q, n):
        if self.metric == "IP":
            q = q / np.linalg.norm(q)
        self.q = q
        self.n = n

    def run_prepared_query(self):
        self.res, self.dis = self.searcher.search(self.q, self.n)

    def get_prepared_query_results(self):
        return self.res

    def prepare_batch_query(self, X, n):
        self.queries = X
        self.n = n
        self.nq = len(X)

    def run_batch_query(self):
        self.result, self.dis = self.searcher.batch_search(self.queries, self.n)

    def get_batch_results(self):
        return self.result

    def get_stats(self):
        return self.searcher.get_stats()

    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024 / 1024

    def freeIndex(self):
        del self.searcher
        del self.g


if __name__ == "__main__":
    if len(sys.argv) < 2:
        config_path = "examples/config.yaml"
    else:
        config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    index_params = config["index_params"]
    rebuild = config["rebuild"]
    build_quant = config["build_quant"]
    search_quants = config["search_quants"]
    refine_quant = config.get("refine_quant", "")
    efs = config["efs"]
    topks = config["topks"]
    runs = config["runs"]
    batch = config["batch"]
    dataset_names = config["datasets"]

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    for dataset_name in dataset_names:
        dataset = dataset_dict[dataset_name]()
        base = dataset.get_database()
        query = dataset.get_queries()
        for is_batch in batch:
            for topk in topks:
                plt.figure(figsize=(10, 6), dpi=120)
                gt = dataset.get_groundtruth(topk)
                name = dataset.name
                metric = dataset.metric
                nq, _ = query.shape
                for index_param in index_params:
                    index_type = index_param["index_type"]
                    index_args = index_param["index_args"]
                    R = index_args["R"]
                    L = index_args["L"] if "L" in index_args else 0
                    for search_quant in search_quants:
                        p = Glass(
                            name,
                            build_quant,
                            search_quant,
                            refine_quant,
                            metric,
                            rebuild,
                            {
                                "index_type": index_type,
                                "R": R,
                                "L": L,
                                "batch": is_batch,
                            },
                        )
                        t = time()
                        p.fit(base)
                        ela = time() - t
                        print(f"Building time of index: {ela}s")
                        print(f"Memory usage of index: {p.get_memory_usage()}MB")
                        qpss = []
                        recalls = []
                        print(
                            f"dataset: {name}, index: {index_type}, quantizer: {search_quant}, top{topk}"
                        )
                        print(
                            f"{'ef':>4} | {'Recall (%)':>12} | {'Max QPS':>12} | {'P99 Latency (ms)':>20} | {'Avg Dist Comps':>18}"
                        )
                        print("-" * 75)
                        for ef in efs:
                            p.set_query_arguments(ef)
                            mx_qps = 0.0
                            for run in range(runs):
                                T = 0
                                res = np.zeros_like(gt)
                                if is_batch:
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
                                if run == 0:
                                    cnt = 0
                                    for i in range(nq):
                                        cnt += np.intersect1d(res[i], gt[i]).size
                                    recall = cnt / nq / topk
                                qps = nq / T
                                mx_qps = max(mx_qps, qps)
                            stats = p.get_stats()
                            print(
                                f"{ef:4d} | {recall*100.0:12.2f} | {mx_qps:12.2f} | {stats['p99_latency_ms']:20.3f} | {stats['avg_dist_comps']:18.2f}"
                            )
                            qpss.append(mx_qps)
                            recalls.append(recall)
                        plt.plot(
                            recalls,
                            qpss,
                            label=f"{index_type} + {search_quant} R={R} L={L}",
                            marker="x",
                        )
                plt.xlabel(f"Recall@{topk}")
                plt.ylabel("QPS")
                plt.legend()
                title = f"{name}{'_batch' if is_batch else ''}_top{topk}"
                plt.title(title)
                plt.savefig(os.path.join(results_dir, f"{title}.png"))
