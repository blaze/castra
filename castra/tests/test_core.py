import os
import pandas as pd
import pandas.util.testing as tm
from castra import Castra
from castra.core import _safe_mkdir
import tempfile
import pickle
import shutil
import nose.tools as nt


A = pd.DataFrame({'x': [1, 2],
                  'y': [1., 2.]},
                 columns=['x', 'y'],
                 index=[1, 2])

B = pd.DataFrame({'x': [10, 20],
                  'y': [10., 20.]},
                 columns=['x', 'y'],
                 index=[10, 20])


def test_safe_mkdir_with_new():
    path = os.path.join(tempfile.mkdtemp(prefix='castra-'), 'db')
    _safe_mkdir(path)
    nt.assert_true(os.path.exists(path))
    nt.assert_true(os.path.isdir(path))
    shutil.rmtree(path)


def test_safe_mkdir_with_existing():
    path = tempfile.mkdtemp(prefix='castra-')
    # an existing path should not raise an exception
    _safe_mkdir(path)
    shutil.rmtree(path)


def test_create_with_random_directory():
    c = Castra(template=A)

def test_create_with_non_existing_path():
    path = os.path.join(tempfile.mkdtemp(prefix='castra-'), 'db')
    c = Castra(path=path, template=A)
    # need to del c now so that it doesn't barf when we remove it's directory
    shutil.rmtree(path)

def test_create_with_existing_path():
    path = tempfile.mkdtemp(prefix='castra-')
    c = Castra(path=path, template=A)
    # need to del c now so that it doesn't barf when we remove it's directory
    shutil.rmtree(path)

def test_exception_with_non_dir():
    path = tempfile.mkdtemp(prefix='castra-')
    file_ = os.path.join(path, 'file')
    with open(file_, 'w') as f:
        f.write('file')
    nt.assert_raises(ValueError, Castra, path=file_)
    shutil.rmtree(path)

def test_exception_with_existing_castra_and_template():
    path = tempfile.mkdtemp(prefix='castra-')
    with Castra(path=path, template=A) as c:
        c.extend(A)
    nt.assert_raises(ValueError, Castra, path=path, template=A)
    shutil.rmtree(path)

def test_exception_with_empty_dir_and_no_template():
    path = tempfile.mkdtemp(prefix='castra-')
    nt.assert_raises(ValueError, Castra, path=path)
    shutil.rmtree(path)

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

    path = tempfile.mkdtemp(prefix='castra-')
    with Castra(path=path, template=A) as c:
        assert os.path.exists(c.path)
    assert os.path.exists(c.path)


def test_load_Castra():
    path = tempfile.mkdtemp(prefix='castra-')
    with Castra(path=path, template=A) as c:
        c.extend(A)
        c.extend(B)

    loaded = Castra(path=path)
    tm.assert_frame_equal(pd.concat([A, B]), loaded[:])


def test_pickle_Castra():
    path = tempfile.mkdtemp(prefix='castra-')
    c = Castra(path=path, template=A)
    c.extend(A)
    c.extend(B)

    dumped = pickle.dumps(c)
    undumped = pickle.loads(dumped)

    tm.assert_frame_equal(pd.concat([A, B]), undumped[:])
