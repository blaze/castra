import os
import pandas as pd
import pandas.util.testing as tm
from castra import Castra
import tempfile


A = pd.DataFrame({'x': [1, 2],
                  'y': [1., 2.]},
                 columns=['x', 'y'],
                 index=[1, 2])

B = pd.DataFrame({'x': [10, 20],
                  'y': [10., 20.]},
                 columns=['x', 'y'],
                 index=[10, 20])


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


def test_del():
    c = Castra(template=A)
    assert os.path.exists(c.path)
    c.__del__()
    assert not os.path.exists(c.path)


def test_context_manager():
    with Castra(template=A) as c:
        assert os.path.exists(c.path)
    assert not os.path.exists(c.path)


def test_load_Castra():
    path = tempfile.mkdtemp(prefix='castra-')
    c = Castra(path=path, template=A)
    c.extend(A)
    c.extend(B)
    c.save_partition_list()

    loaded = Castra(path=path)
    tm.assert_frame_equal(c[:], loaded[:])
