import tempfile
import bloscpack
import blosc
import pickle
from bisect import bisect
import os
from os import mkdir
from os.path import exists, isdir, join
import pandas as pd
import numpy as np
from pandas import msgpack
import shutil


def escape(text):
    return str(text)


def _safe_mkdir(path):
    if not exists(path):
        mkdir(path)


class Castra(object):
    def __init__(self, path=None, template=None):
        # check if we should create a random path
        if path is None:
            self.path = tempfile.mkdtemp(prefix='castra-')
            self._explicitly_given_path = False
        else:
            self.path = path
            self._explicitly_given_path = True

        # check if the given path exists already and create it if it doesn't
        if not exists(self.path):
            mkdir(self.path)
        # raise an Exception if it isn't a directory
        elif not isdir(self.path):
            raise ValueError("'path': %s must be a directory")

        self.meta_path = self.dirname('meta')

        # either we have a meta directory
        if exists(self.meta_path) and isdir(self.meta_path):
            if template is not None:
                raise ValueError(
                    "'template' must be 'None' when opening a Castra")
            self.load_meta()
            self.load_partition_list()
        # or we don't, in which case we need a template
        elif template is not None:
            mkdir(self.meta_path)
            self.columns, self.dtypes, self.index_dtype = \
                list(template.columns), template.dtypes, template.index.dtype
            self.partition_list = list()
            self.flush_meta()
            self.save_partition_list()
        else:
            raise ValueError(
                "must specify a 'template' when creating a new Castra")

    def load_meta(self, loads=pickle.loads):
        meta = []
        for name in ['columns', 'dtypes', 'index_dtype']:
            with open(join(self.meta_path, name), 'r') as f:
                meta.append(loads(f.read()))
        self.columns, self.dtype, self.index_dtype = meta

    def flush_meta(self, dumps=pickle.dumps):
        for name in ['columns', 'dtypes', 'index_dtype']:
            with open(join(self.meta_path, name), 'w') as f:
                f.write(dumps(getattr(self, name)))

    def load_partition_list(self, loads=pickle.loads):
        with open(join(self.meta_path, 'plist'), 'r') as f:
            self.partition_list = pickle.loads(f.read())

    def save_partition_list(self, dumps=pickle.dumps):
        with open(join(self.meta_path, 'plist'), 'w') as f:
            f.write(dumps(self.partition_list))

    def extend(self, df):
        # TODO: Ensure that df is consistent with existing data
        index = df.index.values
        partition_name = '--'.join([escape(index.min()), escape(index.max())])

        mkdir(self.dirname(partition_name))

        # Store columns
        for col in df.columns:
            fn = self.dirname(partition_name, col)
            x = df[col].values
            pack_file(x, fn)

        # Store index
        fn = self.dirname(partition_name, '.index')
        x = df.index.values
        bloscpack.pack_ndarray_file(x, fn)

        self.partition_list.append((index.max(), partition_name))

    def dirname(self, *args):
        return os.path.join(self.path, *args)

    def load_partition(self, name):
        columns = [unpack_file(self.dirname(name, col))
                   for col in self.columns]
        index = unpack_file(self.dirname(name, '.index'))

        return pd.DataFrame(dict(zip(self.columns, columns)),
                            columns=self.columns,
                            index=pd.Index(index, dtype=self.index_dtype))

    def __getitem__(self, key):
        assert isinstance(key, slice)
        i, j = select_partitions(self.partition_list, key)
        start, stop = key.start, key.stop
        data_frames = [self.load_partition(name)
                       for _, name in self.partition_list[slice(i, j)]]
        data_frames[0] = data_frames[0].loc[start:]
        data_frames[-1] = data_frames[-1].loc[:stop]
        return pd.concat(data_frames)

    def drop(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def flush(self):
        self.save_partition_list()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self._explicitly_given_path:
            self.drop()
        else:
            self.save_partition_list()

    def __del__(self):
        if not self._explicitly_given_path:
            self.drop()

    def __getstate__(self):
        self.save_partition_list()
        return (self.path, self._explicitly_given_path)

    def __setstate__(self, state):
        self.path = state[0]
        self._explicitly_given_path = state[1]
        self.meta_path = self.dirname('meta')
        self.load_meta()
        self.load_partition_list()


def pack_file(x, fn):
    """ Pack numpy array into filename

    Supports binary data with bloscpack and text data with msgpack+blosc

    >>> pack_file(np.array([1, 2, 3]), 'foo.blp')  # doctest: +SKIP

    See also:
        unpack_file
    """
    if x.dtype != 'O':
        bloscpack.pack_ndarray_file(x, fn)
    else:
        bytes = blosc.compress(msgpack.packb(x.tolist()), 1)
        with open(fn, 'wb') as f:
            f.write(bytes)


def unpack_file(fn):
    """ Unpack numpy array from filename

    Supports binary data with bloscpack and text data with msgpack+blosc

    >>> unpack_file('foo.blp')  # doctest: +SKIP
    array([1, 2, 3])

    See also:
        pack_file
    """
    try:
        return bloscpack.unpack_ndarray_file(fn)
    except ValueError:
        with open(fn, 'rb') as f:
            bytes = f.read()
        return np.array(msgpack.unpackb(blosc.decompress(bytes)))


def select_partitions(partition_list, key):
    """ Select partitions from partition list given slice

    >>> pl = [(0, 'a'), (10, 'b'), (20, 'c'), (30, 'd'), (40, 'e')]
    >>> select_partitions(pl, slice(3, 25))
    (1, 4)
    """
    assert key.step is None
    start, stop = key.start, key.stop
    i = bisect(partition_list, (start, None)) if start is not None else None
    j = bisect(partition_list, (stop, None)) + 1 if stop is not None else None
    return i, j
