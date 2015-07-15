from collections import Iterator

import os

from os.path import exists, isdir

try:
    import cPickle as pickle
except ImportError:
    import pickle

import shutil
import tempfile

from functools import partial

import blosc
import bloscpack

import numpy as np
import pandas as pd

from pandas import msgpack


def escape(text):
    return str(text)


def mkdir(path):
    if not exists(path):
        os.makedirs(path)


class Castra(object):
    meta_fields = ['columns', 'dtypes', 'index_dtype', 'axis_names']

    def __init__(self, path=None, template=None, categories=None):
        # check if we should create a random path
        self._explicitly_given_path = path is not None

        if not self._explicitly_given_path:
            self.path = tempfile.mkdtemp(prefix='castra-')
        else:
            self.path = path

        # check if the given path exists already and create it if it doesn't
        mkdir(self.path)

        # raise an Exception if it isn't a directory
        if not isdir(self.path):
            raise ValueError("'path': %s must be a directory")

        # either we have a meta directory
        if isdir(self.dirname('meta')):
            if template is not None:
                raise ValueError(
                    "'template' must be 'None' when opening a Castra")
            self.load_meta()
            self.load_partitions()
            self.load_categories()

        # or we don't, in which case we need a template
        elif template is not None:
            self.columns, self.dtypes, self.index_dtype = \
                list(template.columns), template.dtypes, template.index.dtype
            self.axis_names = [template.index.name, template.columns.name]
            self.partitions = pd.Series([], dtype='O',
                                        index=template.index.__class__([]))
            self.minimum = None
            if isinstance(categories, (list, tuple)):
                self.categories = dict((col, []) for col in categories)
            elif categories is True:
                self.categories = dict((col, [])
                                       for col in template.columns
                                       if template.dtypes[col] == 'object')
            else:
                self.categories = dict()

            if self.categories:
                categories = set(self.categories)
                template_categories = set(template.dtypes.index.values)
                if categories.difference(template_categories):
                    raise ValueError('passed in categories %s are not all '
                                     'contained in template dataframe columns '
                                     '%s' % (categories, template_categories))
                for c in self.categories:
                    self.dtypes[c] = pd.core.categorical.CategoricalDtype()

            mkdir(self.dirname('meta', 'categories'))
            self.flush_meta()
            self.save_partitions()
        else:
            raise ValueError(
                "must specify a 'template' when creating a new Castra")

    def load_meta(self, loads=pickle.loads):
        for name in self.meta_fields:
            with open(self.dirname('meta', name), 'rb') as f:
                setattr(self, name, loads(f.read()))

    def flush_meta(self, dumps=partial(pickle.dumps, protocol=2)):
        for name in self.meta_fields:
            with open(self.dirname('meta', name), 'wb') as f:
                f.write(dumps(getattr(self, name)))

    def load_partitions(self, loads=pickle.loads):
        with open(self.dirname('meta', 'plist'), 'rb') as f:
            self.partitions = loads(f.read())
        with open(self.dirname('meta', 'minimum'), 'rb') as f:
            self.minimum = loads(f.read())

    def save_partitions(self, dumps=partial(pickle.dumps, protocol=2)):
        with open(self.dirname('meta', 'minimum'), 'wb') as f:
            f.write(dumps(self.minimum))
        with open(self.dirname('meta', 'plist'), 'wb') as f:
            f.write(dumps(self.partitions))

    def append_categories(self, new, dumps=partial(pickle.dumps, protocol=2)):
        separator = b'-sep-'
        for col, cat in new.items():
            if cat:
                with open(self.dirname('meta', 'categories', col), 'ab') as f:
                    f.write(separator.join(map(dumps, cat)))
                    f.write(separator)

    def load_categories(self, loads=pickle.loads):
        separator = b'-sep-'
        self.categories = dict()
        for col in self.columns:
            fn = self.dirname('meta', 'categories', col)
            if os.path.exists(fn):
                with open(fn, 'rb') as f:
                    text = f.read()
                self.categories[col] = [loads(x)
                                        for x in text.split(separator)[:-1]]

    def extend(self, df):
        # TODO: Ensure that df is consistent with existing data
        if not df.index.is_monotonic_increasing:
            df = df.sort_index(inplace=False)
        index = df.index.values
        partition_name = '--'.join([escape(index.min()), escape(index.max())])

        mkdir(self.dirname(partition_name))

        new_categories, self.categories, df = _decategorize(self.categories, df)
        self.append_categories(new_categories)

        # Store columns
        for col in df.columns:
            pack_file(df[col].values, self.dirname(partition_name, col))

        # Store index
        fn = self.dirname(partition_name, '.index')
        x = df.index.values
        bloscpack.pack_ndarray_file(x, fn)

        if not len(self.partitions):
            self.minimum = index.min()
        self.partitions[index.max()] = partition_name
        self.flush()

    def dirname(self, *args):
        return os.path.join(self.path, *args)

    def load_partition(self, name, columns, categorize=True):
        if isinstance(columns, Iterator):
            columns = list(columns)
        if not isinstance(columns, list):
            df = self.load_partition(name, [columns], categorize=categorize)
            return df.iloc[:, 0]
        arrays = [unpack_file(self.dirname(name, col)) for col in columns]
        index = unpack_file(self.dirname(name, '.index'))

        df = pd.DataFrame(dict(zip(columns, arrays)),
                          columns=pd.Index(columns, name=self.axis_names[1]),
                          index=pd.Index(index, dtype=self.index_dtype,
                                         name=self.axis_names[0]))
        if categorize:
            df = _categorize(self.categories, df)
        return df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key, columns = key
        else:
            columns = self.columns
        start, stop = key.start, key.stop
        names = select_partitions(self.partitions, key)

        data_frames = [self.load_partition(name, columns, categorize=False)
                       for name in names]

        data_frames[0] = data_frames[0].loc[start:]
        data_frames[-1] = data_frames[-1].loc[:stop]
        df = pd.concat(data_frames)
        df = _categorize(self.categories, df)
        return df

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
            self.flush()

    def __del__(self):
        if not self._explicitly_given_path:
            self.drop()
        else:
            self.flush()

    def __getstate__(self):
        self.flush()
        return (self.path, self._explicitly_given_path)

    def __setstate__(self, state):
        self.path = state[0]
        self._explicitly_given_path = state[1]
        self.load_meta()
        self.load_partitions()
        self.load_categories()

    def to_dask(self, columns=None):
        if columns is None:
            columns = self.columns
        import dask.dataframe as dd
        name = 'from-castra' + next(dd.core.tokens)
        dsk = dict(((name, i), (Castra.load_partition, self, part, columns))
                    for i, part in enumerate(self.partitions.values))
        divisions = [self.minimum] + list(self.partitions.index)
        if isinstance(columns, list):
            return dd.DataFrame(dsk, name, columns, divisions)
        else:
            return dd.Series(dsk, name, columns, divisions)


def pack_file(x, fn, encoding='utf8'):
    """ Pack numpy array into filename

    Supports binary data with bloscpack and text data with msgpack+blosc

    >>> pack_file(np.array([1, 2, 3]), 'foo.blp')  # doctest: +SKIP

    See also:
        unpack_file
    """
    if x.dtype != 'O':
        bloscpack.pack_ndarray_file(x, fn)
    else:
        bytes = blosc.compress(msgpack.packb(x.tolist(), encoding=encoding), 1)
        with open(fn, 'wb') as f:
            f.write(bytes)


def unpack_file(fn, encoding='utf8'):
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
            return np.array(msgpack.unpackb(blosc.decompress(f.read()),
                                            encoding=encoding))


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
    assert key.step is None, 'step must be None but was %s' % key.step
    start, stop = key.start, key.stop
    if start is not None:
        start = coerce_index(partitions.index.dtype, start)
        istart = partitions.index.searchsorted(start)
    else:
        istart = 0
    if stop is not None:
        stop = coerce_index(partitions.index.dtype, stop)
        istop = partitions.index.searchsorted(stop)
    else:
        istop = len(partitions) - 1

    names = partitions.iloc[istart: istop + 1].values.tolist()
    return names


def _decategorize(categories, df):
    """ Strip object dtypes from dataframe, update categories

    Given a DataFrame

    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': ['C', 'B', 'B']})

    And a dict of known categories

    >>> _ = categories = {'y': ['A', 'B']}

    Update dict and dataframe in place

    >>> extra, categories, df = _decategorize(categories, df)
    >>> extra
    {'y': ['C']}
    >>> categories
    {'y': ['A', 'B', 'C']}
    >>> df
       x  y
    0  1  2
    1  2  1
    2  3  1
    """
    extra = dict()
    new_categories = dict()
    new_columns = dict((col, df[col]) for col in df.columns)
    for col, cat in categories.items():
        idx = pd.Index(df[col])
        idx = getattr(idx, 'categories', idx)
        extra[col] = idx[~idx.isin(cat)].unique().tolist()
        new_categories[col] = cat + extra[col]
        new_columns[col] = pd.Categorical(df[col], new_categories[col]).codes
    new_df = pd.DataFrame(new_columns, columns=df.columns, index=df.index)
    return extra, new_categories, new_df


def _categorize(categories, df):
    """ Categorize columns in dataframe

    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 2, 0]})
    >>> categories = {'y': ['A', 'B', 'c']}
    >>> _categorize(categories, df)
       x  y
    0  1  A
    1  2  c
    2  3  A
    """
    if isinstance(df, pd.Series):
        if df.name in categories:
            cat = pd.Categorical.from_codes(df.values, categories[df.name])
            return pd.Series(cat, index=df.index)
        else:
            return df

    else:
        return pd.DataFrame(
                dict((col, pd.Categorical.from_codes(df[col], categories[col])
                           if col in categories
                           else df[col])
                    for col in df.columns),
                columns=df.columns,
                index=df.index)
