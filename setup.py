# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools
import sys

long_description = "PaddlePaddle Self-Supervised Development Toolkit"

setuptools.setup(
    name="passl",
    version='0.0.4',
    author="duanboqiang",
    author_email="duanboqiang@baidu.com",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/PASSL",
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=['cython', 'numpy'],
    install_requires=[
        'ftfy', 'regex',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['passl=passl.command:main', ]})
