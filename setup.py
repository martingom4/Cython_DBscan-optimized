

# Ensure libomp and LLVM are installed: brew install libomp llvm
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "cython_dbscan.dbscan_core",
        ["cython_dbscan/dbscan_core.pyx"],
        language="c++",
        include_dirs=[
            "/opt/homebrew/opt/libomp/include",
            "/opt/homebrew/include",
            "/opt/homebrew/opt/llvm/include",
            numpy.get_include()
        ],
        library_dirs=["/opt/homebrew/opt/libomp/lib"],
        extra_compile_args=["-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-lomp"],
    )
]

setup(
    name="cython_dbscan",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    zip_safe=False,
)
