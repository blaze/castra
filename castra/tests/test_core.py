import os
import pandas as pd
import pandas.util.testing as tm
import numpy as np
from castra import Castra
from castra.core import _safe_mkdir
import tempfile
import pickle
import shutil
import nose.tools as nt
import unittest


A = pd.DataFrame({'x': [1, 2],
                  'y': [1., 2.]},
                 columns=['x', 'y'],
                 index=[1, 2])

B = pd.DataFrame({'x': [10, 20],
                  'y': [10., 20.]},
                 columns=['x', 'y'],
                 index=[10, 20])


class Base(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='castra-')

    def tearDown(self):
        shutil.rmtree(self.path)


class TestSafeMkdir(Base):

    def test_safe_mkdir_with_new(self):
        path = os.path.join(self.path, 'db')
        _safe_mkdir(path)
        nt.assert_true(os.path.exists(path))
        nt.assert_true(os.path.isdir(path))

    def test_safe_mkdir_with_existing(self):
        # an existing path should not raise an exception
        _safe_mkdir(self.path)


class TestConstructorAndContextManager(Base):

    def test_create_with_random_directory(self):
        Castra(template=A)

    def test_create_with_non_existing_path(self):
        path = os.path.join(self.path, 'db')
        Castra(path=path, template=A)

    def test_create_with_existing_path(self):
        Castra(path=self.path, template=A)

    def test_exception_with_non_dir(self):
        file_ = os.path.join(self.path, 'file')
        with open(file_, 'w') as f:
            f.write('file')
        nt.assert_raises(ValueError, Castra, path=file_)

    def test_exception_with_existing_castra_and_template(self):
        with Castra(path=self.path, template=A) as c:
            c.extend(A)
        nt.assert_raises(ValueError, Castra, path=self.path, template=A)

    def test_exception_with_empty_dir_and_no_template(self):
        nt.assert_raises(ValueError, Castra, path=self.path)

    def test_load(self):
        with Castra(path=self.path, template=A) as c:
            c.extend(A)
            c.extend(B)

        loaded = Castra(path=self.path)
        tm.assert_frame_equal(pd.concat([A, B]), loaded[:])

    def test_del_with_random_dir(self):
        c = Castra(template=A)
        assert os.path.exists(c.path)
        c.__del__()
        assert not os.path.exists(c.path)

    def test_context_manager_with_random_dir(self):
        with Castra(template=A) as c:
            assert os.path.exists(c.path)
        assert not os.path.exists(c.path)

    def test_context_manager_with_specific_dir(self):
        with Castra(path=self.path, template=A) as c:
            assert os.path.exists(c.path)
        assert os.path.exists(c.path)


def test_timeseries():
    indices = [pd.DatetimeIndex(start=str(i), end=str(i+1), freq='w')
                for i in range(2000, 2015)]
    dfs = [pd.DataFrame({'x': list(range(len(ind)))}, ind)
                for ind in indices]

    with Castra(template=dfs[0]) as c:
        for df in dfs:
            c.extend(df)
        df = c['2010-05': '2013-02']
        assert len(df) > 100


def test_Castra():
    c = Castra(template=A)
    c.extend(A)
    c.extend(B)

    assert c.columns == ['x', 'y']

    tm.assert_frame_equal(c[0:100], pd.concat([A, B]))
    tm.assert_frame_equal(c[:5], A)
    tm.assert_frame_equal(c[5:], B)

    tm.assert_frame_equal(c[2:5], A[1:])
    tm.assert_frame_equal(c[2:15], pd.concat([A[1:], B[:1]]))


def test_pickle_Castra():
    path = tempfile.mkdtemp(prefix='castra-')
    c = Castra(path=path, template=A)
    c.extend(A)
    c.extend(B)

    dumped = pickle.dumps(c)
    undumped = pickle.loads(dumped)

    tm.assert_frame_equal(pd.concat([A, B]), undumped[:])


def test_text():
    df = pd.DataFrame({'name': ['Alice', 'Bob'],
                       'balance': [100, 200]}, columns=['name', 'balance'])
    with Castra(template=df) as c:
        c.extend(df)

        tm.assert_frame_equal(c[:], df)


def test_column_access():
    with Castra(template=A) as c:
        c.extend(A)
        c.extend(B)
        df = c[:, ['x']]

        tm.assert_frame_equal(df, pd.concat([A[['x']], B[['x']]]))

        df = c[:, 'x']
        tm.assert_series_equal(df, pd.concat([A.x, B.x]))


def test_reload():
    path = tempfile.mkdtemp(prefix='castra-')
    try:
        c = Castra(template=A, path=path)
        c.extend(A)

        d = Castra(path=path)

        assert c.columns == d.columns
        assert (c.partitions == d.partitions).all()
        assert c.minimum == d.minimum
    finally:
        shutil.rmtree(path)


def test_index_dtype_matches_template():
    with Castra(template=A) as c:
        assert c.partitions.index.dtype == A.index.dtype


def test_to_dask_dataframe():
    try:
        import dask.dataframe as dd
    except ImportError:
        return

    with Castra(template=A) as c:
        c.extend(A)
        c.extend(B)

        df = c.to_dask()
        assert isinstance(df, dd.DataFrame)
        assert list(df.divisions) == [1, 2, 20]
        tm.assert_frame_equal(df.compute(), c[:])

        df = c.to_dask('x')
        assert isinstance(df, dd.Series)
        assert list(df.divisions) == [1, 2, 20]
        tm.assert_series_equal(df.compute(), c[:, 'x'])


def test_categorize():
    A = pd.DataFrame({'x': [1, 2, 3], 'y': ['A', None, 'A']},
                     columns=['x', 'y'], index=[0, 10, 20])
    B = pd.DataFrame({'x': [4, 5, 6], 'y': ['C', None, 'A']},
                     columns=['x', 'y'], index=[30, 40, 50])

    with Castra(template=A, categories=['y']) as c:
        c.extend(A)
        assert c[:].dtypes['y'] == 'category'
        assert c[:]['y'].cat.codes.dtype == np.dtype('i1')
        assert list(c[:, 'y'].cat.categories) == ['A', None]

        c.extend(B)
        assert list(c[:, 'y'].cat.categories) == ['A', None, 'C']

        assert c.load_partition(c.partitions.iloc[0], 'y').dtype == 'category'

        c.flush()

        d = Castra(path=c.path)
        tm.assert_frame_equal(c[:], d[:])
