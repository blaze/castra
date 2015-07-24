Castra
======

|Build Status|

Castra is an on-disk, partitioned, compressed, column store with a focus on
time-series applications.  It provides efficient columnar range queries.

*  **Efficient on-disk:**  Castra stores data on your hard drive in a way that you can load it quickly, increasing the comfort of inconveniently large data.
*  **Partitioned:**  Castra partitions your data along an index, allowing rapid loads of ranges of data like "All records between January and March"
*  **Compressed:**  Castra uses Blosc_ to compress data, increasing effective disk bandwidth and decreasing storage costs
*  **Column-store:**  Castra stores columns separately, drastically reducing I/O costs for analytic queries
*  **Tabular and time-series data:**  Castra plays well with Pandas and is an ideal fit for append-only applications like time-series

.. _Blosc: https://github.com/Blosc

.. |Build Status| image:: https://travis-ci.org/Blosc/castra.svg
   :target: https://travis-ci.org/Blosc/castra
