from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import psutil
import yaml
from sklearn.cluster import KMeans

import glass
from ann_dataset import dataset_dict


@dataclass
class GraphIndexConfig:
    """Configuration for graph-based index."""

    index_type: str
    R: int
    L: int = 0
    build_quant: str = "SQ8U"


@dataclass
class IvfIndexConfig:
    """Configuration for IVF index."""

    index_type: str
    nlist: int


IndexConfig = Union[GraphIndexConfig, IvfIndexConfig]


def create_index_config(data: Dict[str, Any]) -> IndexConfig:
    """Factory function to create index configuration."""
    index_type = data["index_type"]
    args = data["index_args"]
    if index_type == "IVF":
        return IvfIndexConfig(index_type=index_type, nlist=args["nlist"])
    else:  # For HNSW and other graph-based indexes
        return GraphIndexConfig(
            index_type=index_type,
            R=args["R"],
            L=args.get("L", 0),
            build_quant=args.get("build_quant", "SQ8U"),
        )


@dataclass
class SearchConfig:
    """Configuration for search parameters."""

    search_quant: str
    refine_quant: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchConfig:
        """Create SearchConfig from dictionary."""
        return cls(
            search_quant=data["search_quant"], refine_quant=data.get("refine_quant", "")
        )


class Glass:
    """Glass index wrapper for ANN search."""

    def __init__(
        self,
        dataset_name: str,
        search_config: SearchConfig,
        index_config: IndexConfig,
        metric: str,
        rebuild: bool = False,
    ):
        self.dataset_name = dataset_name
        self.search_config = search_config
        self.index_config = index_config
        self.metric = metric
        self.rebuild = rebuild
        self.dir = Path("indices")
        self.searcher = None

    @property
    def name(self) -> str:
        """Generate descriptive name for the index."""
        return f"glass_({self.index_config})"

    @property
    def path(self) -> Path:
        """Generate index file path."""
        if isinstance(self.index_config, IvfIndexConfig):
            filename = (
                f"{self.dataset_name}_{self.index_config.index_type}_"
                f"nlist{self.index_config.nlist}.glass"
            )
        else:
            filename = (
                f"{self.dataset_name}_{self.index_config.index_type}_"
                f"{self.index_config.build_quant}_R{self.index_config.R}_L{self.index_config.L}.glass"
            )
        return self.dir / filename

    def _build_graph_index(self, X: np.ndarray) -> None:
        """Build graph-based index."""
        assert isinstance(self.index_config, GraphIndexConfig)
        self.dir.mkdir(exist_ok=True)

        if self.rebuild or not self.path.exists():
            glass.build_graph(
                self.index_config.index_type,
                X,
                str(self.path),
                metric=self.metric,
                quant=self.index_config.build_quant,
                R=self.index_config.R,
                L=self.index_config.L,
            )

        g = glass.Graph(str(self.path))
        self.searcher = glass.Searcher(
            g,
            X,
            self.metric,
            self.search_config.search_quant,
            self.search_config.refine_quant,
        )
        self.searcher.enable_stats(True)
        self.searcher.optimize()

    def _build_ivf_index(self, X: np.ndarray) -> None:
        """Build IVF index using KMeans clustering."""
        assert isinstance(self.index_config, IvfIndexConfig)
        print("Fitting KMeans")
        kmeans = KMeans(n_clusters=self.index_config.nlist, random_state=0).fit(
            X[:10000]
        )
        print("KMeans done")

        centroids = kmeans.cluster_centers_
        ivf_map = kmeans.predict(X)

        self.searcher = glass.Searcher(
            X,
            centroids,
            ivf_map,
            self.metric,
            self.search_config.search_quant,
            self.search_config.refine_quant,
        )

    def fit(self, X: np.ndarray) -> None:
        """Fit the index on data."""
        if isinstance(self.index_config, IvfIndexConfig):
            self._build_ivf_index(X)
        elif isinstance(self.index_config, GraphIndexConfig):
            self._build_graph_index(X)
        else:
            raise TypeError(f"Unsupported index config type: {type(self.index_config)}")

    def search(
        self, queries: np.ndarray, topk: int, ef: int, nthreads: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors."""
        self.searcher.set_ef(ef)
        return self.searcher.batch_search(queries, topk, nthreads)

    def get_stats(self) -> Dict[str, float]:
        """Get search statistics."""
        return self.searcher.get_stats()

    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        return psutil.Process().memory_info().rss / 1024 / 1024


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_concurrency_values(config: Dict[str, Any]) -> List[int]:
    """Generate concurrency values from min to max, doubling each time."""
    min_conc = config.get("min", 1)
    max_conc = config.get("max", psutil.cpu_count())

    values = []
    current = min_conc
    while current <= max_conc:
        values.append(current)
        current *= 2

    return values


def calculate_recall(results: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate recall rate."""
    nq = results.shape[0]
    topk = ground_truth.shape[1]

    correct = sum(np.intersect1d(results[i], ground_truth[i]).size for i in range(nq))

    return correct / (nq * topk)


def run_benchmark(
    glass_index: Glass,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef: int,
    concurrency: int,
    topk: int,
    runs: int = 3,
) -> Dict[str, float]:
    """Run benchmark for specific parameters."""
    max_qps = 0.0
    recall = 0.0

    for run in range(runs):
        start_time = time()
        results, distances = glass_index.search(queries, topk, ef, concurrency)
        elapsed = time() - start_time

        if run == 0:
            recall = calculate_recall(results, ground_truth)

        qps = len(queries) / elapsed
        max_qps = max(max_qps, qps)

    stats = glass_index.get_stats()

    return {
        "recall": recall,
        "max_qps": max_qps,
        "p99_latency_ms": stats["p99_latency_ms"],
        "avg_dist_comps": stats["avg_dist_comps"],
    }


def print_benchmark_header(
    dataset_name: str, index_type: str, search_quant: str, topk: int
) -> None:
    """Print benchmark header."""
    print(
        f"\ndataset: {dataset_name}, index: {index_type}, quantizer: {search_quant}, top{topk}"
    )
    print(
        f"{'ef':>4} | {'concurrency':>12} | {'Recall (%)':>12} | {'Max QPS':>12} | "
        f"{'P99 Latency (ms)':>20} | {'Avg Dist Comps':>18}"
    )
    print("-" * 90)


def print_benchmark_results(
    ef: int, concurrency: int, results: Dict[str, float]
) -> None:
    """Print benchmark results."""
    print(
        f"{ef:4d} | {concurrency:12d} | {results['recall'] * 100.0:12.2f} | "
        f"{results['max_qps']:12.2f} | {results['p99_latency_ms']:20.3f} | "
        f"{results['avg_dist_comps']:18.2f}"
    )


def main() -> None:
    """Main function to run benchmarks."""
    # Load configuration
    config_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/config.yaml")
    )
    config = load_config(config_path)

    # Parse configuration
    dataset_names = config["datasets"]
    index_params = config["index_params"]
    rebuild = config["rebuild"]
    search_quants = config["search_quants"]
    efs = config["efs"]
    topks = config["topks"]
    runs = config["runs"]
    concurrency_values = generate_concurrency_values(config["concurrency"])

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Run benchmarks for each dataset
    for dataset_name in dataset_names:
        dataset = dataset_dict[dataset_name]()
        base_data = dataset.get_database()
        queries = dataset.get_queries()

        for topk in topks:
            ground_truth = dataset.get_groundtruth(topk)

            for index_param in index_params:
                index_config = create_index_config(index_param)

                for search_param in search_quants:
                    # Create search configuration
                    search_config = SearchConfig.from_dict(search_param)

                    # Initialize Glass index
                    glass_index = Glass(
                        dataset_name=dataset.name,
                        search_config=search_config,
                        index_config=index_config,
                        metric=dataset.metric,
                        rebuild=rebuild,
                    )

                    # Build index
                    start_time = time()
                    glass_index.fit(base_data)
                    build_time = time() - start_time

                    print(f"\nBuilding time of index: {build_time:.2f}s")
                    print(
                        f"Memory usage of index: {glass_index.get_memory_usage():.2f}MB"
                    )

                    # Print benchmark header
                    print_benchmark_header(
                        dataset.name,
                        index_config.index_type,
                        search_config.search_quant,
                        topk,
                    )

                    # Run benchmarks for different parameters
                    for ef in efs:
                        for concurrency in concurrency_values:
                            results = run_benchmark(
                                glass_index,
                                queries,
                                ground_truth,
                                ef,
                                concurrency,
                                topk,
                                runs,
                            )
                            print_benchmark_results(ef, concurrency, results)
                        print("-" * 90)


if __name__ == "__main__":
    main()
