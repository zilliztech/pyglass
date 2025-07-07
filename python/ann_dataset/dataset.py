import os
import h5py
import numpy as np
from sklearn import preprocessing
import requests
from pathlib import Path

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")


class Dataset:
    def __init__(self, name, metric="L2", normalize=False, url="", data_dir="datasets"):
        self.name = name
        self.metric = metric
        self._normalize = normalize
        self.file_path = Path(data_dir) / f"{self.name}.hdf5"
        self._file = None
        self._download_url = url

        self._load_or_download_file()

    def _load_or_download_file(self):
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._download_file()
        self._file = h5py.File(self.file_path, "r")

    def _download_file(self):
        url = (
            f"{HF_ENDPOINT}/datasets/{self._download_url}/resolve/main/{self.name}.hdf5"
        )
        print(f"Downloading {self.name} from {url} to {self.file_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(self.file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    def _preprocess(self, data):
        if self._normalize:
            return preprocessing.normalize(data, norm="l2", axis=1)
        return data

    def get_database(self):
        return self._preprocess(np.array(self._file["train"]))

    def get_queries(self):
        return self._preprocess(np.array(self._file["test"]))

    def get_groundtruth(self, k):
        return np.array(self._file["neighbors"])[:, :k]

    def evaluate(self, pred, k=None):
        nq, topk = pred.shape
        if k is not None:
            topk = k
        gt = self.get_groundtruth(topk)
        cnt = 0
        for i in range(nq):
            cnt += np.intersect1d(pred[i], gt[i]).size
        return cnt / nq / k


hhy3_url = "hhy3/ann-datasets"
vibe_url = "vector-index-bench/vibe"

dataset_configs = {
    "sift-128-euclidean": {"metric": "L2", "normalize": False, "url": hhy3_url},
    "fashion-mnist-784-euclidean": {
        "metric": "L2",
        "normalize": False,
        "url": hhy3_url,
    },
    "nytimes-256-angular": {"metric": "IP", "normalize": True, "url": hhy3_url},
    "glove-100-angular": {"metric": "IP", "normalize": False, "url": hhy3_url},
    "glove-25-angular": {"metric": "IP", "normalize": True, "url": hhy3_url},
    "lastfm-64-dot": {"metric": "IP", "normalize": True, "url": hhy3_url},
    "gist-960-euclidean": {"metric": "L2", "normalize": False, "url": hhy3_url},
    "cohere-768-angular": {"metric": "IP", "normalize": True, "url": hhy3_url},
    "llama-128-ip": {"metric": "IP", "normalize": False, "url": vibe_url},
    "yi-128-ip": {"metric": "IP", "normalize": False, "url": vibe_url},
    "agnews-mxbai-1024-euclidean": {
        "metric": "L2",
        "normalize": False,
        "url": vibe_url,
    },
}

dataset_dict = {
    name: lambda n=name, cfg=config: Dataset(n, **cfg)
    for name, config in dataset_configs.items()
}
