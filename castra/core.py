import tempfile
import bloscpack
import blosc
import pickle
import os
from os import mkdir
from os.path import exists, isdir, join
import pandas as pd
import numpy as np
from pandas import msgpack
import shutil
from pandas.core.indexing import convert_to_index_sliceable
import numpy as np


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
            self.load_partitions()
        # or we don't, in which case we need a template
        elif template is not None:
            mkdir(self.meta_path)
            self.columns, self.dtypes, self.index_dtype = \
                list(template.columns), template.dtypes, template.index.dtype
            self.partitions = pd.Series([], dtype='O',
                                        index=template.index.__class__([]))
            self.flush_meta()
            self.save_partitions()
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

    def load_partitions(self, loads=pickle.loads):
        with open(join(self.meta_path, 'plist'), 'r') as f:
            self.partitions = pickle.loads(f.read())

    def save_partitions(self, dumps=pickle.dumps):
        with open(join(self.meta_path, 'plist'), 'w') as f:
            f.write(dumps(self.partitions))

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

        self.partitions[index.max()] = partition_name

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
        start, stop = key.start, key.stop
        names = select_partitions(self.partitions, key)

        data_frames = [self.load_partition(name) for name in names]

        data_frames[0] = data_frames[0].loc[start:]
        data_frames[-1] = data_frames[-1].loc[:stop]
        return pd.concat(data_frames)

    def drop(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def flush(self):
        self.save_partitions()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self._explicitly_given_path:
            self.drop()
        else:
            self.save_partitions()

    def __del__(self):
        if not self._explicitly_given_path:
            self.drop()

    def __getstate__(self):
        self.save_partitions()
        return (self.path, self._explicitly_given_path)

    def __setstate__(self, state):
        self.path = state[0]
        self._explicitly_given_path = state[1]
        self.meta_path = self.dirname('meta')
        self.load_meta()
        self.load_partitions()


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


def coerce_index(dt, o):
    if np.issubdtype(dt, np.datetime64):
        return pd.Timestamp(o)
    return o


def select_partitions(partitions, key):
    """ Select partitions from partition list given slice

    >>> p = pd.Series(['a', 'b', 'c', 'd', 'e'], index=[0, 10, 20, 30, 40])
    >>> select_partitions(p, slice(3, 25))
    ['b', 'c', 'd']
    """
    assert key.step is None
    start, stop = key.start, key.stop
    names = list(partitions.loc[start:stop])

    last = partitions.searchsorted(names[-1])[0]

    stop2 = coerce_index(partitions.index.dtype, stop)
    if partitions.index[last] < stop2 and len(partitions) > last + 1:
        names.append(partitions.iloc[last + 1])

    return names
