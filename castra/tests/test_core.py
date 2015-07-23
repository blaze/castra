import os
import tempfile
import pickle
import shutil

import pandas as pd
import pandas.util.testing as tm

import pytest

import numpy as np

import pytest

from castra import Castra
from castra.core import mkdir, select_partitions


A = pd.DataFrame({'x': [1, 2],
                  'y': [1., 2.]},
                 columns=['x', 'y'],
                 index=[1, 2])

B = pd.DataFrame({'x': [10, 20],
                  'y': [10., 20.]},
                 columns=['x', 'y'],
                 index=[10, 20])


C = pd.DataFrame({'x': [10, 20],
                  'y': [10., 20.],
                  'z': [0, 1]},
                 columns=['x', 'y', 'z']).set_index('z')
C.columns.name = 'cols'


@pytest.yield_fixture
def base():
    d = tempfile.mkdtemp(prefix='castra-')
    try:
        yield d
    finally:
        shutil.rmtree(d)


def test_safe_mkdir_with_new(base):
    path = os.path.join(base, 'db')
    mkdir(path)
    assert os.path.exists(path)
    assert os.path.isdir(path)


def test_safe_mkdir_with_existing(base):
    # an existing path should not raise an exception
    mkdir(base)


def test_create_with_random_directory():
    Castra(template=A)


def test_create_with_non_existing_path(base):
    path = os.path.join(base, 'db')
    Castra(path=path, template=A)


def test_create_with_existing_path(base):
    Castra(path=base, template=A)


def test_exception_with_non_dir(base):
    file_ = os.path.join(base, 'file')
    with open(file_, 'w') as f:
        f.write('file')
    with pytest.raises(ValueError):
        Castra(file_)


def test_exception_with_existing_castra_and_template(base):
    with Castra(path=base, template=A) as c:
        c.extend(A)
    with pytest.raises(ValueError):
        Castra(path=base, template=A)


def test_exception_with_empty_dir_and_no_template(base):
    with pytest.raises(ValueError):
        Castra(path=base)


def test_load(base):
    with Castra(path=base, template=A) as c:
        c.extend(A)
        c.extend(B)

    loaded = Castra(path=base)
    tm.assert_frame_equal(pd.concat([A, B]), loaded[:])


def test_del_with_random_dir():
    c = Castra(template=A)
    assert os.path.exists(c.path)
    c.__del__()
    assert not os.path.exists(c.path)


def test_context_manager_with_random_dir():
    with Castra(template=A) as c:
        assert os.path.exists(c.path)
    assert not os.path.exists(c.path)


def test_context_manager_with_specific_dir(base):
    with Castra(path=base, template=A) as c:
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


def test_save_axis_names():
    with Castra(template=C) as c:
        c.extend(C)
        assert c[:].index.name == 'z'
        assert c[:].columns.name == 'cols'
        tm.assert_frame_equal(c[:], C)


def test_same_categories_when_already_categorized():
    A = pd.DataFrame({'x': [1, 2] * 1000,
                      'y': [1., 2.] * 1000,
                      'z': np.random.choice(list('abc'), size=2000)},
                     columns=list('xyz'))
    A['z'] = A.z.astype('category')
    with Castra(template=A, categories=['z']) as c:
        c.extend(A)
        assert c.categories['z'] == A.z.cat.categories.tolist()


def test_category_dtype():
    A = pd.DataFrame({'x': [1, 2] * 3,
                      'y': [1., 2.] * 3,
                      'z': list('abcabc')},
                     columns=list('xyz'))
    with Castra(template=A, categories=['z']) as c:
        c.extend(A)
        assert A.dtypes['z'] == 'object'
        assert c.dtypes['z'] == pd.core.categorical.CategoricalDtype()


def test_do_not_create_dirs_if_template_fails():
    A = pd.DataFrame({'x': [1, 2] * 3,
                      'y': [1., 2.] * 3,
                      'z': list('abcabc')},
                     columns=list('xyz'))
    with pytest.raises(ValueError):
        Castra(template=A, path='foo', categories=['w'])
    assert not os.path.exists(os.path.join('foo', 'meta'))
    assert not os.path.exists(os.path.join('foo', 'meta', 'categories'))


def test_sort_on_extend():
    df = pd.DataFrame({'x': [1, 2, 3]}, index=[3, 2, 1])
    expected = pd.DataFrame({'x': [3, 2, 1]}, index=[1, 2, 3])
    with Castra(template=df) as c:
        c.extend(df)
        tm.assert_frame_equal(c[:], expected)


def test_select_partitions():
    p = pd.Series(['a', 'b', 'c', 'd', 'e'], index=[0, 10, 20, 30, 40])
    assert select_partitions(p, slice(3, 25)) == ['b', 'c', 'd']
    assert select_partitions(p, slice(None, 25)) == ['a', 'b', 'c', 'd']
    assert select_partitions(p, slice(3, None)) == ['b', 'c', 'd', 'e']
    assert select_partitions(p, slice(None, None)) == ['a', 'b', 'c', 'd', 'e']
    assert select_partitions(p, slice(10, 30)) == ['b', 'c', 'd']


def test_minimum_dtype():
    df = tm.makeTimeDataFrame()

    with Castra(template=df) as c:
        c.extend(df)
        assert type(c.minimum) == type(c.partitions.index[0])


def test_many_default_indexes():
    a = pd.DataFrame({'x': [1, 2, 3]})
    b = pd.DataFrame({'x': [4, 5, 6]})

    with Castra(template=a) as c:
        c.extend(a)
        c.extend(b)

        assert (c[:, 'x'].values == [1, 2, 3, 4, 5, 6]).all()


def test_raise_error_on_mismatched_index():
    a = pd.DataFrame({'x': [1, 2, 3]}, index=[1, 2, 3])
    b = pd.DataFrame({'x': [4, 5, 6]}, index=[2, 3, 4])

    with Castra(template=a) as c:
        c.extend(a)

        with pytest.raises(ValueError):
            c.extend(b)
