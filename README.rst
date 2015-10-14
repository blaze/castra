Castra
======

|Build Status|

Castra is an on-disk, partitioned, compressed, column store.
Castra provides efficient columnar range queries.

*  **Efficient on-disk:**  Castra stores data on your hard drive in a way that you can load it quickly, increasing the comfort of inconveniently large data.
*  **Partitioned:**  Castra partitions your data along an index, allowing rapid loads of ranges of data like "All records between January and March"
*  **Compressed:**  Castra uses Blosc_ to compress data, increasing effective disk bandwidth and decreasing storage costs
*  **Column-store:**  Castra stores columns separately, drastically reducing I/O costs for analytic queries
*  **Tabular data:**  Castra plays well with Pandas and is an ideal fit for append-only applications like time-series

Example
-------

Consider some Pandas DataFrames

.. code-block:: python

   In [1]: import pandas as pd
   In [2]: A = pd.DataFrame({'price': [10.0, 11.0], 'volume': [100, 200]},
      ...:                  index=pd.DatetimeIndex(['2010', '2011']))

   In [3]: B = pd.DataFrame({'price': [12.0, 13.0], 'volume': [300, 400]},
      ...:                  index=pd.DatetimeIndex(['2012', '2013']))

We create a Castra with a filename and a template dataframe from which to get
column name, index, and dtype information

.. code-block:: python

   In [4]: from castra import Castra
   In [5]: c = Castra('data.castra', template=A)

The castra starts empty but we can extend it with new dataframes:

.. code-block:: python

   In [6]: c.extend(A)

   In [7]: c[:]
   Out[7]:
               price  volume
   2010-01-01     10     100
   2011-01-01     11     200

   In [8]: c.extend(B)

   In [9]: c[:]
   Out[9]:
               price  volume
   2010-01-01     10     100
   2011-01-01     11     200
   2012-01-01     12     300
   2013-01-01     13     400

We can select particular columns

.. code-block:: python

   In [10]: c[:, 'price']
   Out[10]:
   2010-01-01    10
   2011-01-01    11
   2012-01-01    12
   2013-01-01    13
   Name: price, dtype: float64

Particular ranges

.. code-block:: python

   In [12]: c['2011':'2013']
   Out[12]:
               price  volume
   2011-01-01     11     200
   2012-01-01     12     300
   2013-01-01     13     400

Or both

.. code-block:: python

   In [13]: c['2011':'2013', 'volume']
   Out[13]:
   2011-01-01    200
   2012-01-01    300
   2013-01-01    400
   Name: volume, dtype: int64

Storage
-------

Castra stores your dataframes as they arrived, you can see the divisions along
which you data is divided.

.. code-block:: python

   In [14]: c.partitions
   Out[14]:
   2011-01-01    2009-12-31T16:00:00.000000000-0800--2010-12-31...
   2013-01-01    2011-12-31T16:00:00.000000000-0800--2012-12-31...
   dtype: object

Each column in each partition lives in a separate compressed file::

   $ ls -a data.castra/2011-12-31T16:00:00.000000000-0800--2012-12-31T16:00:00.000000000-0800
   .  ..  .index  price  volume

Restrictions
------------

Castra is both fast and restrictive.

*  You must always give it dataframes that match its template (same column
   names, index type, dtypes).
*  You can only give castra dataframes with **increasing index values**.  For
   example you can give it one dataframe a day for values on that day.  You can
   not go back and update previous days.

Text and Categoricals
---------------------

Castra tries to encode text and object dtype columns with
msgpack_, using the implementation found in
the Pandas library.  It falls back to `pickle` with a high protocol if that
fails.

Alternatively, Castra can categorize your data as it receives it

.. code-block:: python

   >>> c = Castra('data.castra', template=df, categories=['list', 'of', 'columns'])

   or

   >>> c = Castra('data.castra', template=df, categories=True) # all object dtype columns

Categorizing columns that have repetitive text, like ``'sex'`` or
``'ticker-symbol'`` can greatly improve both read times and computational
performance with Pandas.  See this blogpost_ for more information.

.. _msgpack: http://msgpack.org/index.html


Dask dataframe
--------------

Castra interoperates smoothly with dask.dataframe_

.. code-block:: python

   >>> import dask.dataframe as dd
   >>> df = dd.read_csv('myfiles.*.csv')
   >>> df.set_index('timestamp', compute=False).to_castra('myfile.castra', categories=True)

   >>> df = dd.from_castra('myfile.castra')

Work in Progress
----------------

Castra is immature and largely for experimental use.

The developers do not promise backwards compatibility with future versions.
You should treat castra as a very efficient temporary format and archive your
data with some other system.



.. _Blosc: https://github.com/Blosc

.. _dask.dataframe: https://dask.pydata.org/en/latest/dataframe.html

.. _blogpost: http://matthewrocklin.com/blog/work/2015/06/18/Categoricals/

.. |Build Status| image:: https://travis-ci.org/blaze/castra.svg
   :target: https://travis-ci.org/blaze/castra
