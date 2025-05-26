from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="cython_dbscan",
    ext_modules=cythonize("cython_dbscan/dbscan_core.pyx"),
    include_dirs=[numpy.get_include()]
)
