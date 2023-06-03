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
        self.file = None

    def evaluate(self, pred, k=None):
        nq, topk = pred.shape
        if k is not None:
            topk = k
        gt = self.get_groundtruth(topk)
        cnt = 0
        for i in range(nq):
            cnt += np.intersect1d(pred[i], gt[i]).size
        return cnt / nq / k

    def get_database(self):
        ret = np.array(self.file['train'])
        if self.metric == "IP":
            ret = preprocessing.normalize(ret)
        return ret

    def get_queries(self):
        ret = np.array(self.file['test'])
        if self.metric == "IP":
            ret = preprocessing.normalize(ret)
        return ret

    def get_groundtruth(self, k):
        ret = np.array(self.file['neighbors'])
        return ret[:, :k]

    def get_fname(self, dir):
        if dir is None:
            dir = "datasets"
        if not os.path.exists(dir):
            os.mkdir(dir)
        return f'{dir}/{self.name}.hdf5'


def download(name):
    url = f'https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/{name}.hdf5'
    return url


class DatasetSIFT1M(Dataset):
    name = "sift-128-euclidean"
    metric = "L2"

    def __init__(self, dir=None):
        path = self.get_fname(dir)
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.file = h5py.File(path)


class DatasetFashionMnist(Dataset):
    name = "fashion-mnist-784-euclidean"
    metric = "L2"

    def __init__(self, dir=None):
        path = self.get_fname(dir)
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.file = h5py.File(path)


class DatasetNYTimes(Dataset):
    name = "nytimes-256-angular"
    metric = "IP"

    def __init__(self, dir=None):
        path = self.get_fname(dir)
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.file = h5py.File(path)


class DatasetGlove100(Dataset):
    name = "glove-100-angular"
    metric = "IP"

    def __init__(self, dir=None):
        path = self.get_fname(dir)
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.file = h5py.File(path)


class DatasetGlove25(Dataset):
    name = "glove-25-angular"
    metric = "IP"

    def __init__(self, dir=None):
        path = self.get_fname(dir)
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.file = h5py.File(path)


class DatasetLastFM64(Dataset):
    name = "lastfm-64-dot"
    metric = "IP"

    def __init__(self, dir=None):
        path = self.get_fname(dir)
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.file = h5py.File(path)


class DatasetGIST960(Dataset):
    name = "gist-960-euclidean"
    metric = "L2"

    def __init__(self, dir=None):
        path = self.get_fname(dir)
        if not os.path.exists(path):
            os.system(f'wget --output-document {path} {download(self.name)}')
        self.file = h5py.File(path)


dataset_dict = {'sift-128-euclidean': DatasetSIFT1M, 'fashion-mnist-784-euclidean': DatasetFashionMnist,
                'nytimes-256-angular': DatasetNYTimes, 'glove-100-angular': DatasetGlove100, 'glove-25-angular': DatasetGlove25, 'lastfm-64-dot': DatasetLastFM64, 'gist-960-euclidean': DatasetGIST960}
