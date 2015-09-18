#!/usr/bin/env python

from os.path import exists
from setuptools import setup

setup(name='castra',
      version='0.1.6',
      description='On-disk partitioned store',
      url='http://github.com/blaze/Castra/',
      maintainer='Matthew Rocklin',
      maintainer_email='mrocklin@gmail.com',
      license='BSD',
      keywords='',
      packages=['castra'],
      package_data={'castra': ['tests/*.py']},
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      zip_safe=False)
