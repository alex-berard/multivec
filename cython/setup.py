from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

sources = ["multivec.pyx", "../multivec/monolingual.cpp", "../multivec/bilingual.cpp", "../multivec/distance.cpp"]
module = Extension("multivec", sources, undef_macros=['NDEBUG'], language="c++")
module.extra_compile_args = ['--std=c++11', '-w', '-I../multivec', '-O3']
module.libraries = ['m']
setup(name="multivec", version="1.0", ext_modules=cythonize([module]), include_dirs=[numpy.get_include()])
