#!/usr/bin/env python
"""A package based on sklearn Random Forest to compute 
conditional quantiles.

See:
https://github.com/NazBen/pyquantregForest
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyquantregForest',

    version='0.1',

    description='A sample Python project',
    #long_description=long_description,

    url='https://github.com/NazBen/pyquantregForest',

    author='Nazih Benoumechiara',
    author_email='nazih.benoumechiara@gmail.com',

    license='MIT',

    keywords='sklearn randomforest quantile',
    
    packages=['pyquantregForest'],  # Python packages to install
    # (If we have individual .py modules we can use the py_module argument instead)
    # This is the full name of the script "simcluster"; this will be installed to a
    # bin/ directory

    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn'],
)