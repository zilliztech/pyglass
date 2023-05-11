import os
import h5py
import numpy as np
from sklearn import preprocessing


def hdf5_read(fname, metric):
    file = h5py.File(fname, 'r')
    base = np.array(file['train'])
    query = np.array(file['test'])
    gt = np.array(file['neighbors'])
    if metric == "IP":
        base = preprocessing.normalize(base, norm='l2', axis=1)
        query = preprocessing.normalize(query, norm='l2', axis=1)
    return base, query, gt


class Dataset:
    def __init__(self):
        self.name = ""
        self.metric = "L2"
        self.d = -1
        self.nb = -1
        self.nq = -1
        self.base = None
        self.query = None
        self.gt = None

    def evaluate(self, pred, k):
        gt = self.get_groundtruth(k)
        cnt = 0
        for i in range(self.nq):
            cnt += np.intersect1d(pred[i], gt[i]).size
        return cnt / self.nq / k

    def get_base(self):
        return self.base

    def get_queries(self):
        return self.query

    def get_groundtruth(self, k):
        return self.gt[:, :k]

    def get_fname(self):
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
        return f'datasets/{self.name}.hdf5'

def download(name):
    url = f'https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/{name}.hdf5' 
    return url


class DatasetSIFT1M(Dataset):
    name = "sift-128-euclidean"
    metric = "L2"

    def __init__(self):
        path = self.get_fname()
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.base, self.query, self.gt = hdf5_read(path, self.metric)
        self.nb, self.d = self.base.shape
        self.nq, self.d = self.query.shape


class DatasetFashionMnist(Dataset):
    name = "fashion-mnist-784-euclidean"
    metric = "L2"

    def __init__(self):
        path = self.get_fname()
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.base, self.query, self.gt = hdf5_read(path, self.metric)
        self.nb, self.d = self.base.shape
        self.nq, self.d = self.query.shape


class DatasetNYTimes(Dataset):
    name = "nytimes-256-angular"
    metric = "IP"

    def __init__(self):
        path = self.get_fname()
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.base, self.query, self.gt = hdf5_read(path, self.metric)
        self.nb, self.d = self.base.shape
        self.nq, self.d = self.query.shape


class DatasetGlove100(Dataset):
    name = "glove-100-angular"
    metric = "IP"

    def __init__(self):
        path = self.get_fname()
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.base, self.query, self.gt = hdf5_read(path, self.metric)
        self.nb, self.d = self.base.shape
        self.nq, self.d = self.query.shape


class DatasetGlove25(Dataset):
    name = "glove-25-angular"
    metric = "IP"

    def __init__(self):
        path = self.get_fname()
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.base, self.query, self.gt = hdf5_read(path, self.metric)
        self.nb, self.d = self.base.shape
        self.nq, self.d = self.query.shape


class DatasetLastFM64(Dataset):
    name = "lastfm-64-dot"
    metric = "IP"

    def __init__(self):
        path = f"datasets/{self.name}.hdf5"
        path = self.get_fname()
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.base, self.query, self.gt = hdf5_read(path, self.metric)
        self.nb, self.d = self.base.shape
        self.nq, self.d = self.query.shape


class DatasetGIST960(Dataset):
    name = "gist-960-euclidean"
    metric = "L2"

    def __init__(self):
        path = f"datasets/{self.name}.hdf5"
        path = self.get_fname()
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.base, self.query, self.gt = hdf5_read(path, self.metric)
        self.nb, self.d = self.base.shape
        self.nq, self.d = self.query.shape


dataset_dict = {'sift-128-euclidean': DatasetSIFT1M, 'fashion-mnist-784-euclidean': DatasetFashionMnist,
                'nytimes-256-angular': DatasetNYTimes, 'glove-100-angular': DatasetGlove100, 'glove-25-angular': DatasetGlove25, 'lastfm-64-dot': DatasetLastFM64, 'gist-960-euclidean': DatasetGIST960}

