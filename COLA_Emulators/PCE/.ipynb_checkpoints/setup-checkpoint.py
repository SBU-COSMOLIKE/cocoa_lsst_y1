
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np  # Make sure to import numpy

extension = Extension(
    "pce_",
    sources=["cython_functions.pyx"],
    libraries=["gsl", "gslcblas", "m"],
    library_dirs=["$CONDA_PREFIX/lib"],
    include_dirs=[np.get_include(), "$CONDA_PREFIX/include"],  # np.get_include() provides the path to numpy headers
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='PCE Functions',
    ext_modules=cythonize([extension]),
    zip_safe=False,
)



