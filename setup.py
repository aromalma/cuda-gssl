from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import subprocess

# Customize these variables based on your project structure

from setuptools import setup

from setuptools_cuda import CudaExtension
from setuptools.command.build_ext import build_ext
# from torch.utils.cpp_extension  import CudaExtension
# class CustomBuildExtension(build_ext):
#     def build_extensions(self):
#         for ext in self.extensions:
#             ext.extra_compile_args = { 'nvcc': ["-arch=compute_60 -code=sm_60"]}
#         super().build_extensions()
setup(
    cuda_extensions=[
        Extension(
            name="gssl",
            sources=['gssl.cu', 'kernels.cu'],
            extra_compile_args={"cxx": [],'nvcc':["-arch=compute_60","-code=sm_60"]},#"-arch=compute_75","code=sm_75"]},
            include_dirs=["/home/tl028/dev_env/lib/python3.10/site-packages/pybind11/include"],
            # extra_link_args={'nvcc': ["-arch=compute_60", "-code=sm_60"]}
        ),

    ],
    # cmdclass={'build_ext': CustomBuildExtension},

)















# source_files = ['gssl.cu', 'kernels.cu']

# class CMakeExtension(Extension):
#     def __init__(self, name, sources=[]):
#         super().__init__(name=name, sources=sources)

# class BuildExtension(build_ext):
#     def run(self):
#         try:
#             subprocess.check_output(['nvcc', '--version'])
#         except OSError:
#             raise RuntimeError("nvcc compiler is not installed. Please install CUDA.")

#         super().run()

#     def build_extensions(self):
#         for ext in self.extensions:
#             print(ext,"!@#@##@")
#             self.build_extension(ext)

#     def build_extension(self, ext):
#         sources = ext.sources or []
#         sources += source_files

#         extra_compile_args = ['-O3', '-gencode', 'arch=compute_60,code=sm_60 ', '-shared']
        
#         ext.extra_compile_args = {'cxx': [],
#                                   'nvcc': extra_compile_args}

#         ext.include_dirs.append('/usr/include/python3.10')  # Add path to your CUDA includes
#         ext.include_dirs.append("/home/tl028/dev_env/lib/python3.10/site-packages/pybind11/include")
#         # print(ext.)

#         setuptools.command.build_ext.build_ext.build_extension(self, ext)

# setup(
#     name='gssl',
#     version='0.1',
#     ext_modules=[CMakeExtension('gssl',source_files)],
#     cmdclass={'build_ext': BuildExtension},
# )
