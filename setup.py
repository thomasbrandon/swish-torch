from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXT_SRCS = [
    'csrc/swish.cpp',
    'csrc/swish_kernel.cu',
]

setup(
    name='swish_torch',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    install_requires=['torch>=1.2'],
    ext_modules=[
        CUDAExtension(
            'swish_torch._C',
            EXT_SRCS,
            extra_compile_args={
                'cxx': [],
                'nvcc': ['--expt-extended-lambda']
            },
            include_dirs=['external']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
