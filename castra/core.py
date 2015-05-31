import tempfile
import bloscpack
import pickle
from bisect import bisect
import os
import pandas as pd
import shutil


def escape(text):
    return str(text)


class Castra(object):
    def __init__(self, columns, dtypes, index_dtype, path=None):
        if path is None:
            path = tempfile.mkdtemp(prefix='castra-')
            self._explicitly_given_path = False
        else:
            self._explicitly_given_path = True

        self.path = path
        self.meta_path = self.dirname('meta')
        self.columns = list(columns)
        self.dtypes = dtypes
        self.index_dtype = index_dtype

        self.partition_list = list()
        self.flush_meta()

    def flush_meta(self, dumps=pickle.dumps):
        if not os.path.exists(self.meta_path):
            os.mkdir(self.meta_path)
        for name in ['columns', 'dtypes', 'index_dtype']:
            with open(os.path.join(self.meta_path, name), 'w') as f:
                f.write(dumps(getattr(self, name)))

    def extend(self, df):
        # TODO: Ensure that df is consistent with existing data
        index = df.index.values
        partition_name = '--'.join([escape(index.min()), escape(index.max())])

        os.mkdir(self.dirname(partition_name))

        # Store columns
        for col in df.columns:
            fn = self.dirname(partition_name, col)
            x = df[col].values
            bloscpack.pack_ndarray_file(x, fn)

        # Store index
        fn = self.dirname(partition_name, '.index')
        x = df.index.values
        bloscpack.pack_ndarray_file(x, fn)

        self.partition_list.append((index.max(), partition_name))

    def dirname(self, *args):
        return os.path.join(self.path, *args)

    def load_partition(self, name):
        columns = [bloscpack.unpack_ndarray_file(self.dirname(name, col))
                   for col in self.columns]
        index = bloscpack.unpack_ndarray_file(self.dirname(name, '.index'))

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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.drop()

    def __del__(self):
        if not self._explicitly_given_path:
            self.drop()


def select_partitions(partition_list, key):
    """ Select partitions from partition list given slice

    >>> pl = [(0, 'a'), (10, 'b'), (20, 'c'), (30, 'd'), (40, 'e')]
    >>> select_partitions(pl, slice(3, 25))
    (1, 4)
    """
    assert key.step is None
    start, stop = key.start, key.stop
    i = bisect(partition_list, (start, None)) if start is not None else None
    j = bisect(partition_list, (stop, None)) +1 if stop is not None else None
    return i, j
