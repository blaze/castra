from collections import Iterator

import os

from os.path import exists, isdir

try:
    import cPickle as pickle
except ImportError:
    import pickle

import shutil
import tempfile
from hashlib import md5

from functools import partial

import blosc
import bloscpack

import numpy as np
import pandas as pd

from pandas import msgpack


bp_args = bloscpack.BloscpackArgs(offsets=False, checksum='None')

def blosc_args(dt):
    if np.issubdtype(dt, int):
        return bloscpack.BloscArgs(dt.itemsize, clevel=3, shuffle=True)
    if np.issubdtype(dt, np.datetime64):
        return bloscpack.BloscArgs(dt.itemsize, clevel=3, shuffle=True)
    if np.issubdtype(dt, float):
        return bloscpack.BloscArgs(dt.itemsize, clevel=1, shuffle=False)
    return None


# http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename-in-python
import string
valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)

def escape(text):
    """

    >>> escape("Hello!")  # Remove punctuation from names
    'Hello'

    >>> escape("/!.")  # completely invalid names produce hash string
    'cb6698330c63e87fc35933a0474238b0'
    """
    result = ''.join(c for c in str(text) if c in valid_chars)
    if not result:
        result = md5(str(text).encode()).hexdigest()
    return result


def mkdir(path):
    if not exists(path):
        os.makedirs(path)


class Castra(object):
    meta_fields = ['columns', 'dtypes', 'index_dtype', 'axis_names']

    def __init__(self, path=None, template=None, categories=None, readonly=False):
        self._readonly = readonly
        # check if we should create a random path
        self._explicitly_given_path = path is not None

        if not self._explicitly_given_path:
            self.path = tempfile.mkdtemp(prefix='castra-')
        else:
            self.path = path

        # either we have a meta directory
        if isdir(self.dirname('meta')):
            if template is not None:
                raise ValueError(
                    "Opening a castra with a template, yet this castra\n"
                    "already exists.  Filename: %s" % self.path)
            self.load_meta()
            self.load_partitions()
            self.load_categories()

        # or we don't, in which case we need a template
        elif template is not None:
            if self._readonly:
                ValueError("Can't create new castra in readonly mode")

            if isinstance(categories, (list, tuple)):
                if template.index.name in categories:
                    categories.remove(template.index.name)
                    categories.append('.index')
                self.categories = dict((col, []) for col in categories)
            elif categories is True:
                self.categories = dict((col, [])
                                       for col in template.columns
                                       if template.dtypes[col] == 'object')
                if isinstance(template.index, pd.CategoricalIndex):
                    self.categories['.index'] = []
            else:
                self.categories = dict()

            if self.categories:
                categories = set(self.categories)
                template_categories = set(template.dtypes.index.values)
                if categories.difference(template_categories) - set(['.index']):
                    raise ValueError('passed in categories %s are not all '
                                     'contained in template dataframe columns '
                                     '%s' % (categories, template_categories))

            template2 = _decategorize(self.categories, template)[2]

            self.columns, self.dtypes, self.index_dtype = \
                list(template2.columns), template2.dtypes, template2.index.dtype
            self.axis_names = [template2.index.name, template2.columns.name]

            self.partitions = pd.Series([], dtype='O',
                                        index=template2.index.__class__([]))
            self.minimum = None

            # check if the given path exists already and create it if it doesn't
            mkdir(self.path)

            # raise an Exception if it isn't a directory
            if not isdir(self.path):
                raise ValueError("'path': %s must be a directory")

            mkdir(self.dirname('meta', 'categories'))
            self.flush_meta()
            self.save_partitions()
        else:
            raise ValueError(
                "must specify a 'template' when creating a new Castra")

    def _empty_dataframe(self):
        return pd.DataFrame([],
                        columns=pd.Index(self.columns, name=self.axis_names[1]),
                        index=pd.Index([], name=self.axis_names[0]))

    def load_meta(self, loads=pickle.loads):
        for name in self.meta_fields:
            with open(self.dirname('meta', name), 'rb') as f:
                setattr(self, name, loads(f.read()))

    def flush_meta(self, dumps=partial(pickle.dumps, protocol=2)):
        if self._readonly:
            raise IOError('File not open for writing')
        for name in self.meta_fields:
            with open(self.dirname('meta', name), 'wb') as f:
                f.write(dumps(getattr(self, name)))

    def load_partitions(self, loads=pickle.loads):
        with open(self.dirname('meta', 'plist'), 'rb') as f:
            self.partitions = loads(f.read())
        with open(self.dirname('meta', 'minimum'), 'rb') as f:
            self.minimum = loads(f.read())

    def save_partitions(self, dumps=partial(pickle.dumps, protocol=2)):
        if self._readonly:
            raise IOError('File not open for writing')
        with open(self.dirname('meta', 'minimum'), 'wb') as f:
            f.write(dumps(self.minimum))
        with open(self.dirname('meta', 'plist'), 'wb') as f:
            f.write(dumps(self.partitions))

    def append_categories(self, new, dumps=partial(pickle.dumps, protocol=2)):
        if self._readonly:
            raise IOError('File not open for writing')
        separator = b'-sep-'
        for col, cat in new.items():
            if cat:
                with open(self.dirname('meta', 'categories', col), 'ab') as f:
                    f.write(separator.join(map(dumps, cat)))
                    f.write(separator)

    def load_categories(self, loads=pickle.loads):
        separator = b'-sep-'
        self.categories = dict()
        for col in list(self.columns) + ['.index']:
            fn = self.dirname('meta', 'categories', col)
            if os.path.exists(fn):
                with open(fn, 'rb') as f:
                    text = f.read()
                self.categories[col] = [loads(x)
                                        for x in text.split(separator)[:-1]]

    def extend(self, df):
        if self._readonly:
            raise IOError('File not open for writing')
        if len(df) == 0:
            return
        # TODO: Ensure that df is consistent with existing data
        if not df.index.is_monotonic_increasing:
            df = df.sort_index(inplace=False)

        new_categories, self.categories, df = _decategorize(self.categories,
                                                            df)
        self.append_categories(new_categories)

        if len(self.partitions) and df.index[0] <= self.partitions.index[-1]:
            if is_trivial_index(df.index):
                df = df.copy()
                start = self.partitions.index[-1] + 1
                new_index = pd.Index(np.arange(start, start + len(df)),
                                     name = df.index.name)
                df.index = new_index
            else:
                raise ValueError("Index of new dataframe less than known data")

        index = df.index.values
        partition_name = '--'.join([escape(index.min()), escape(index.max())])

        mkdir(self.dirname(partition_name))

        # Store columns
        for col in df.columns:
            pack_file(df[col].values, self.dirname(partition_name, col))

        # Store index
        fn = self.dirname(partition_name, '.index')
        bloscpack.pack_ndarray_file(index, fn, bloscpack_args=bp_args,
                                    blosc_args=blosc_args(index.dtype))

        if not len(self.partitions):
            self.minimum = coerce_index(index.dtype, index.min())
        self.partitions.loc[index.max()] = partition_name
        self.flush()

    def extend_sequence(self, seq, freq=None):
        """Add dataframes from an iterable, optionally repartitioning by freq.

        Parameters
        ----------
        seq : iterable
            An iterable of dataframes
        freq : frequency, optional
            A pandas datetime offset. If provided, the dataframes will be
            partitioned by this frequency.
        """
        if self._readonly:
            raise IOError('File not open for writing')
        if isinstance(freq, str):
            freq = pd.datetools.to_offset(freq)
            partitioner = lambda buf, df: partitionby_freq(freq, buf, df)
        elif freq is None:
            partitioner = partitionby_none
        else:
            raise ValueError("Invalid 'freq': {0}".format(repr(freq)))
        buf = self._empty_dataframe()
        for df in seq:
            write, buf = partitioner(buf, df)
            for frame in write:
                self.extend(frame)
        if buf is not None and not buf.empty:
            self.extend(buf)

    def dirname(self, *args):
        return os.path.join(self.path, *list(map(escape, args)))

    def load_partition(self, name, columns, categorize=True):
        if isinstance(columns, Iterator):
            columns = list(columns)
        if '.index' in self.categories and name in self.partitions.index:
            name = self.categories['.index'].index(name) - 1
        if not isinstance(columns, list):
            df = self.load_partition(name, [columns], categorize=categorize)
            return df.iloc[:, 0]
        arrays = [unpack_file(self.dirname(name, col)) for col in columns]

        df = pd.DataFrame(dict(zip(columns, arrays)),
                          columns=pd.Index(columns, name=self.axis_names[1],
                                           tupleize_cols=False),
                          index=self.load_index(name))
        if categorize:
            df = _categorize(self.categories, df)
        return df

    def load_index(self, name):
        return pd.Index(unpack_file(self.dirname(name, '.index')),
                        dtype=self.index_dtype,
                        name=self.axis_names[0],
                        tupleize_cols=False)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key, columns = key
        else:
            columns = self.columns
        if isinstance(columns, slice):
            columns = self.columns[columns]

        if isinstance(key, slice):
            start, stop = key.start, key.stop
        else:
            start, stop = key, key

        if '.index' in self.categories:
            if start is not None:
                start = self.categories['.index'].index(start)
            if stop is not None:
                stop = self.categories['.index'].index(stop)
        key = slice(start, stop)

        names = select_partitions(self.partitions, key)

        if not names:
            return self._empty_dataframe()[columns]

        data_frames = [self.load_partition(name, columns, categorize=False)
                       for name in names]

        data_frames[0] = data_frames[0].loc[start:]
        data_frames[-1] = data_frames[-1].loc[:stop]
        df = pd.concat(data_frames)
        df = _categorize(self.categories, df)
        return df

    def drop(self):
        if self._readonly:
            raise IOError('File not open for writing')
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def flush(self):
        if self._readonly:
            raise IOError('File not open for writing')
        self.save_partitions()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self._explicitly_given_path:
            self.drop()
        elif not self._readonly:
            self.flush()

    __del__ = __exit__

    def __getstate__(self):
        if not self._readonly:
            self.flush()
        return (self.path, self._explicitly_given_path, self._readonly)

    def __setstate__(self, state):
        self.path = state[0]
        self._explicitly_given_path = state[1]
        self._readonly = state[2]
        self.load_meta()
        self.load_partitions()
        self.load_categories()

    def to_dask(self, columns=None):
        import dask.dataframe as dd

        if columns is None:
            columns = self.columns

        token = md5(str((self.path, os.path.getmtime(self.path))).encode()).hexdigest()
        name = 'from-castra-' + token

        divisions = [self.minimum] + self.partitions.index.tolist()
        if '.index' in self.categories:
            divisions = ([self.categories['.index'][0]]
                       + [self.categories['.index'][d + 1] for d in divisions[1:-1]]
                       + [self.categories['.index'][-1]])

        key_parts = list(enumerate(self.partitions.values))

        dsk = dict(((name, i), (Castra.load_partition, self, part, columns))
                   for i, part in key_parts)
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
        bloscpack.pack_ndarray_file(x, fn, bloscpack_args=bp_args,
                blosc_args=blosc_args(x.dtype))
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
            data = msgpack.unpackb(blosc.decompress(f.read()),
                                   encoding=encoding)
            return np.array(data, object, copy=False)


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
    new_columns = dict((col, df[col].values) for col in df.columns)
    for col, cat in categories.items():
        if col == '.index' or col not in df.columns:
            continue
        idx = pd.Index(df[col])
        idx = getattr(idx, 'categories', idx)
        ex = idx[~idx.isin(cat)].unique()
        if any(pd.isnull(c) for c in cat):
            ex = ex[~pd.isnull(ex)]
        extra[col] = ex.tolist()
        new_categories[col] = cat + extra[col]
        new_columns[col] = pd.Categorical(df[col].values, new_categories[col]).codes

    if '.index' in categories:
        idx = df.index
        idx = getattr(idx, 'categories', idx)
        ex = idx[~idx.isin(cat)].unique()
        if any(pd.isnull(c) for c in cat):
            ex = ex[~pd.isnull(ex)]
        extra['.index'] = ex.tolist()
        new_categories['.index'] = cat + extra['.index']

        new_index = pd.Categorical(df.index, new_categories['.index']).codes
        new_index = pd.Index(new_index, name=df.index.name)
    else:
        new_index = df.index

    new_df = pd.DataFrame(new_columns, columns=df.columns, index=new_index)
    return extra, new_categories, new_df


def make_categorical(s, categories):
    name = '.index' if isinstance(s, pd.Index) else s.name
    if name in categories:
        idx = pd.Index(categories[name], tupleize_cols=False, dtype='object')
        idx.is_unique = True
        cat = pd.Categorical(s.values, categories=idx, fastpath=True, ordered=False)
        return pd.CategoricalIndex(cat, name=s.name, ordered=True) if name == '.index' else cat
    return s if name == '.index' else s.values



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
        return pd.Series(make_categorical(df, categories),
                         index=make_categorical(df.index, categories),
                         name=df.name)
    else:
        return pd.DataFrame(dict((col, make_categorical(df[col], categories))
                                 for col in df.columns),
                            columns=df.columns,
                            index=make_categorical(df.index, categories))


def partitionby_none(buf, new):
    """Repartition to ensure partitions don't split duplicate indices"""
    if new.empty:
        return [], buf
    elif buf.empty:
        return [], new
    if not new.index.is_monotonic_increasing:
        new = new.sort_index(inplace=False)
    end = buf.index[-1]
    if end >= new.index[0] and not is_trivial_index(new.index):
        i = new.index.searchsorted(end, side='right')
        # Only need to concat, `castra.extend` will resort if needed
        buf = pd.concat([buf, new.iloc[:i]])
        new = new.iloc[i:]
    return [buf], new


def partitionby_freq(freq, buf, new):
    """Partition frames into blocks by a freq"""
    df = pd.concat([buf, new])
    if not df.index.is_monotonic_increasing:
        df = df.sort_index(inplace=False)
    start, end = pd.tseries.resample._get_range_edges(df.index[0],
                                                      df.index[-1], freq)
    inds = [df.index.searchsorted(i) for i in
            pd.date_range(start, end, freq=freq)[1:]]
    slices = [(inds[i-1], inds[i]) if i else (0, inds[i]) for i in
              range(len(inds))]
    frames = [df.iloc[i:j] for (i, j) in slices]
    return frames[:-1], frames[-1]


def is_trivial_index(ind):
    """ Is this index just 0..n ?

    If so then we can probably ignore or change it around as necessary

    >>> is_trivial_index(pd.Index([0, 1, 2]))
    True

    >>> is_trivial_index(pd.Index([0, 3, 5]))
    False
    """
    return ind[0] == 0 and (ind == np.arange(len(ind))).all()
