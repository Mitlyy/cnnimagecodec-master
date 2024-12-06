from setuptools import setup, Extension
import os
import sys
import pybind11

#functions_module = Extension(
#    name='EntropyCodec',
#    sources=['wrapper.cpp'],
#   include_dirs=[os.path.join(os.getenv('PYTHON_DIR'), 'include'),
#                os.path.join(pybind11.__path__[0], 'include')]
#)

functions_module = Extension(
    name='EntropyCodec',
    sources=['wrapper.cpp'],
   include_dirs=[os.path.join('/home/eabelyaev/miniconda3/bin/', 'include'),
                os.path.join(pybind11.__path__[0], 'include')]
)

#print(pybind11.__path__[0])
#print(os.getenv('PYTHONPATH'))
#print(sys.path)

setup(ext_modules=[functions_module], options={"build_ext": {"build_lib": ".."}})
