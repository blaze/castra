import os
import tempfile
import pickle
import shutil

import pandas as pd
import pandas.util.testing as tm

import pytest

import numpy as np

from castra import Castra
from castra.core import mkdir, select_partitions, _decategorize, _categorize


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


def test_get_empty(base):
    df = Castra(path=base, template=A)[:]
    assert (df.columns == A.columns).all()


def test_get_empty_result(base):
    c = Castra(path=base, template=A)
    c.extend(A)

    df = c[100:200]

    assert (df.columns == A.columns).all()


def test_get_slice(base):
    c = Castra(path=base, template=A)
    c.extend(A)

    tm.assert_frame_equal(c[:], c[:, :])
    tm.assert_frame_equal(c[:, 1:], c[:][['y']])


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
    dfs = [pd.DataFrame({'x': list(range(len(ind)))}, ind).iloc[:-1]
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


def test_readonly():
    path = tempfile.mkdtemp(prefix='castra-')
    try:
        c = Castra(path=path, template=A)
        c.extend(A)
        d = Castra(path=path, readonly=True)
        with pytest.raises(IOError):
            d.extend(B)
        with pytest.raises(IOError):
            d.extend_sequence([B])
        with pytest.raises(IOError):
            d.flush()
        with pytest.raises(IOError):
            d.drop()
        with pytest.raises(IOError):
            d.save_partitions()
        with pytest.raises(IOError):
            d.flush_meta()
        assert c.columns == d.columns
        assert (c.partitions == d.partitions).all()
        assert c.minimum == d.minimum
    finally:
        shutil.rmtree(path)


def test_index_dtype_matches_template():
    with Castra(template=A) as c:
        assert c.partitions.index.dtype == A.index.dtype


def test_to_dask_dataframe():
    pytest.importorskip('dask.dataframe')

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


def test_do_not_create_dirs_if_template_fails():
    A = pd.DataFrame({'x': [1, 2] * 3,
                      'y': [1., 2.] * 3,
                      'z': list('abcabc')},
                     columns=list('xyz'))
    with pytest.raises(ValueError):
        Castra(template=A, path='foo', categories=['w'])
    assert not os.path.exists('foo')


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


def test_first_index_is_timestamp():
    pytest.importorskip('dask.dataframe')

    df = pd.DataFrame({'x': [1, 2] * 3,
                       'y': [1., 2.] * 3,
                       'z': list('abcabc')},
                      columns=list('xyz'),
                      index=pd.date_range(start='20120101', periods=6))
    with Castra(template=df) as c:
        c.extend(df)

        assert isinstance(c.minimum, pd.Timestamp)
        assert isinstance(c.to_dask().divisions[0], pd.Timestamp)


def test_minimum_dtype():
    df = tm.makeTimeDataFrame()

    with Castra(template=df) as c:
        c.extend(df)
        assert type(c.minimum) == type(c.partitions.index[0])


def test_many_default_indexes():
    a = pd.DataFrame({'x': [1, 2, 3]})
    b = pd.DataFrame({'x': [4, 5, 6]})
    c = pd.DataFrame({'x': [7, 8, 9]})

    e = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

    with Castra(template=a) as C:
        C.extend(a)
        C.extend(b)
        C.extend(c)

        tm.assert_frame_equal(C[:], e)


def test_raise_error_on_mismatched_index():
    x = pd.DataFrame({'x': [1, 2, 3]}, index=[1, 2, 3])
    y = pd.DataFrame({'x': [1, 2, 3]}, index=[4, 5, 6])
    z = pd.DataFrame({'x': [4, 5, 6]}, index=[5, 6, 7])

    with Castra(template=x) as c:
        c.extend(x)
        c.extend(y)

        with pytest.raises(ValueError):
            c.extend(z)


def test_raise_error_on_equal_index():
    a = pd.DataFrame({'x': [1, 2, 3]}, index=[1, 2, 3])
    b = pd.DataFrame({'x': [4, 5, 6]}, index=[3, 4, 5])

    with Castra(template=a) as c:
        c.extend(a)

        with pytest.raises(ValueError):
            c.extend(b)


def test_categories_nan():
    a = pd.DataFrame({'x': ['A', np.nan]})
    b = pd.DataFrame({'x': ['B', np.nan]})

    with Castra(template=a, categories=['x']) as c:
        c.extend(a)
        c.extend(b)
        assert len(c.categories['x']) == 3


def test_extend_sequence_freq():
    df = pd.util.testing.makeTimeDataFrame(1000, 'min')
    seq = [df.iloc[i:i+100] for i in range(0,1000,100)]
    with Castra(template=df) as c:
        c.extend_sequence(seq, freq='h')
        tm.assert_frame_equal(c[:], df)
        parts = pd.date_range(start=df.index[59], freq='h',
                              periods=16).insert(17, df.index[-1])
        tm.assert_index_equal(c.partitions.index, parts)

    with Castra(template=df) as c:
        c.extend_sequence(seq, freq='d')
        tm.assert_frame_equal(c[:], df)
        assert len(c.partitions) == 1


def test_extend_sequence_none():
    data = {'a': range(5), 'b': range(5)}
    p1 = pd.DataFrame(data, index=[1, 2, 3, 4, 5])
    p2 = pd.DataFrame(data, index=[5, 5, 5, 6, 7])
    p3 = pd.DataFrame(data, index=[7, 9, 10, 11, 12])
    seq = [p1, p2, p3]
    df = pd.concat(seq)
    with Castra(template=df) as c:
        c.extend_sequence(seq)
        tm.assert_frame_equal(c[:], df)
        assert len(c.partitions) == 3
        assert len(c.load_partition('1--5', ['a', 'b']).index) == 8
        assert len(c.load_partition('6--7', ['a', 'b']).index) == 3
        assert len(c.load_partition('9--12', ['a', 'b']).index) == 4


def test_extend_sequence_overlap():
    df = pd.util.testing.makeTimeDataFrame(20, 'min')
    p1 = df.iloc[:15]
    p2 = df.iloc[10:20]
    seq = [p1,p2]
    df = pd.concat(seq)
    with Castra(template=df) as c:
        c.extend_sequence(seq)
        tm.assert_frame_equal(c[:], df.sort_index())
        assert (c.partitions.index == [p.index[-1] for p in seq]).all()
    # Check with trivial index
    p1 = pd.DataFrame({'a': range(10), 'b': range(10)})
    p2 = pd.DataFrame({'a': range(10, 17), 'b': range(10, 17)})
    seq = [p1,p2]
    df = pd.DataFrame({'a': range(17), 'b': range(17)})
    with Castra(template=df) as c:
        c.extend_sequence(seq)
        tm.assert_frame_equal(c[:], df)
        assert (c.partitions.index == [9, 16]).all()


def test_extend_sequence_single_frame():
    df = pd.util.testing.makeTimeDataFrame(100, 'h')
    seq = [df]
    with Castra(template=df) as c:
        c.extend_sequence(seq, freq='d')
        assert (c.partitions.index == ['2000-01-01 23:00:00', '2000-01-02 23:00:00',
                 '2000-01-03 23:00:00', '2000-01-04 23:00:00', '2000-01-05 03:00:00']).all()
    df = pd.DataFrame({'a': range(10), 'b': range(10)})
    seq = [df]
    with Castra(template=df) as c:
        c.extend_sequence(seq)
        tm.assert_frame_equal(c[:], df)


def test_column_with_period():
    df = pd.DataFrame({'x': [10, 20],
                       '.': [10., 20.]},
                       columns=['x', '.'],
                       index=[10, 20])

    with Castra(template=df) as c:
        c.extend(df)


def test_empty():
    with Castra(template=A) as c:
        c.extend(pd.DataFrame(columns=A.columns))
        assert len(c[:]) == 0


def test_index_with_single_value():
    df = pd.DataFrame({'x': [1, 2, 3]}, index=[1, 1, 2])
    with Castra(template=df) as c:
        c.extend(df)

        tm.assert_frame_equal(c[1], df.loc[1])


def test_categorical_index():
    df = pd.DataFrame({'x': [1, 2, 3]},
            index=pd.CategoricalIndex(['a', 'a', 'b'], ordered=True, name='foo'))

    with Castra(template=df, categories=True) as c:
        c.extend(df)
        result = c[:]
        tm.assert_frame_equal(c[:], df)

    A = pd.DataFrame({'x': [1, 2, 3]},
                    index=pd.Index(['a', 'a', 'b'], name='foo'))
    B = pd.DataFrame({'x': [4, 5, 6]},
                    index=pd.Index(['c', 'd', 'd'], name='foo'))

    path = tempfile.mkdtemp(prefix='castra-')
    try:
        with Castra(path=path, template=A, categories=['foo']) as c:
            c.extend(A)
            c.extend(B)

            c2 = Castra(path=path)
            result = c2[:]

            expected = pd.concat([A, B])
            expected.index = pd.CategoricalIndex(expected.index,
                    name=expected.index.name, ordered=True)
            tm.assert_frame_equal(result, expected)

            tm.assert_frame_equal(c['a'], expected.loc['a'])
    finally:
        shutil.rmtree(path)


def test_categorical_index_with_dask_dataframe():
    pytest.importorskip('dask.dataframe')
    import dask.dataframe as dd
    import dask

    A = pd.DataFrame({'x': [1, 2, 3, 4]},
                    index=pd.Index(['a', 'a', 'b', 'b'], name='foo'))
    B = pd.DataFrame({'x': [4, 5, 6]},
                    index=pd.Index(['c', 'd', 'd'], name='foo'))


    path = tempfile.mkdtemp(prefix='castra-')
    try:
        with Castra(path=path, template=A, categories=['foo']) as c:
            c.extend(A)
            c.extend(B)

            df = dd.from_castra(path)
            assert df.divisions == ('a', 'c', 'd')

            result = df.compute(get=dask.async.get_sync)

            expected = pd.concat([A, B])
            expected.index = pd.CategoricalIndex(expected.index,
                    name=expected.index.name, ordered=True)

            tm.assert_frame_equal(result, expected)

            tm.assert_frame_equal(df.loc['a'].compute(), expected.loc['a'])
            tm.assert_frame_equal(df.loc['b'].compute(get=dask.async.get_sync),
                                  expected.loc['b'])
    finally:
        shutil.rmtree(path)


def test__decategorize():
    df = pd.DataFrame({'x': [1, 2, 3]},
                      index=pd.CategoricalIndex(['a', 'a', 'b'], ordered=True,
                          name='foo'))

    extra, categories, df2 = _decategorize({'.index': []}, df)

    assert (df2.index == [0, 0, 1]).all()

    df3 = _categorize(categories, df2)

    tm.assert_frame_equal(df, df3)
