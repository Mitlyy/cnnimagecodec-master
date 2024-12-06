from setuptools import setup, Extension
import os
import pybind11

functions_module = Extension(
    name='EntropyCodec',
    sources=['wrapper.cpp'],
    include_dirs=[os.path.join(os.getenv('VIRTUAL_ENV'), 'include'),
                os.path.join(pybind11.__path__[0], 'include')]
)

setup(ext_modules=[functions_module], options={"build_ext": {"build_lib": ".."}})
