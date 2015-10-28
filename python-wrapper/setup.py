from distutils.core import setup, Extension
import numpy
module = Extension("multivec", ["wrapper.cpp", "../multivec/multivec-mono.cpp", "../multivec/multivec-bi.cpp"],
    undef_macros=['NDEBUG'])
module.extra_compile_args = ['--std=c++0x', '-w', '-I../multivec', '-O3']
module.libraries = ['m']
setup(name="multivec", version="1.0", ext_modules=[module], include_dirs=[numpy.get_include()])
