# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md'), encoding='utf-8').read()
except IOError:
    README = ''

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

__version__ = None

with open(os.path.join(here, 'version.py')) as f:
    exec(f.read(), globals())  # pylint: disable=exec-used

setup(
    name='passl',
    version=__version__,
    description='PAddlePaddle Self-Supervised Learning development toolkit',
    long_description=README,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=[
        'self-supervised learning', 'unsupervised learning', 'ssl',
        'contrastive learning', 'mask image modeling', 'mim',
    ],
    author='PASSL Contributors',
    url='https://github.com/PaddlePaddle/PASSL',
    download_url='https://github.com/PaddlePaddle/PASSL.git',
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "passl-train = tools.train:main",
            "passl-eval = tools.eval:main",
            "passl-export = tools.export:main",
        ],
    },
    install_requires=requirements, )
