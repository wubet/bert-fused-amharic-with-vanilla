#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import sys
# sys.path.append('/data/shared/wubet/project/lib64/python3.8/site-packages')
# sys.path.append('/opt/rh/gcc-toolset-11/root/usr/libexec/gcc/x86_64-redhat-linux/11/cc1plus')
from setuptools import setup, find_packages, Extension


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')

with open('README.md') as f:
    readme = f.read()


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


bleu = Extension(
    'fairseq.libbleu',
    sources=[
        'fairseq/clib/libbleu/libbleu.cpp',
        'fairseq/clib/libbleu/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)

# New extensions
extra_compile_args = ["-std=c++11", "-O3"]

data_utils_fast = NumpyExtension(
    "fairseq.data.data_utils_fast",
    sources=["fairseq/data/data_utils_fast.pyx"],
    language="c++",
    extra_compile_args=extra_compile_args,
)

token_block_utils_fast = NumpyExtension(
    "fairseq.data.token_block_utils_fast",
    sources=["fairseq/data/token_block_utils_fast.pyx"],
    language="c++",
    extra_compile_args=extra_compile_args,
)

# Add the new extensions to the list
ext_modules = [bleu, data_utils_fast, token_block_utils_fast]

setup(
    name='fairseq',
    version='0.6.2',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    url='https://github.com/pytorch/fairseq',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=readme,
    install_requires=[
        'cffi',
        'numpy',
        'sacrebleu',
        'torch',
        'tqdm',
        'boto3',
        'requests',
        "cython",
        "hydra-core>=1.0.7,<1.1",
        "omegaconf<2.1",
        "numpy>=1.21.6",
        "regex",
        "sacrebleu>=1.4.12",
        "torch>=1.13.0",
        "tqdm",
        "bitarray",
        "torchaudio>=0.8.0",
        "scikit-learn",
        "packaging",
    ],
    packages=find_packages(exclude=['scripts', 'tests']),
    ext_modules=ext_modules,
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'fairseq-eval-lm = fairseq_cli.eval_lm:cli_main',
            'fairseq-generate = fairseq_cli.generate:cli_main',
            'fairseq-interactive = fairseq_cli.interactive:cli_main',
            'fairseq-preprocess = fairseq_cli.preprocess:cli_main',
            'fairseq-train = fairseq_cli.train:cli_main',
            'fairseq-score = fairseq_cli.score:main',
        ],
    },
)
