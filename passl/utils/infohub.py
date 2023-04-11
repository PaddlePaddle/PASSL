# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

from passl.utils.misc import AttrDict

class RuntimeInfoHub(AttrDict):
    def __getattr__(self, key):
        if key not in self:
            raise ValueError(f'`{key}` not in RuntimeInfoHub, '
                         'please set it firstly by `runtime_info_hub.key = value`.'
                         '\n\nFor Example:\n\n'
                         'from passl.utils.infohub import runtime_info_hub\n'
                         'runtime_info_hub.max_steps = 10000\n'
                         'runtime_info_hub.epochs = 100\n')
        return self[key]

runtime_info_hub = RuntimeInfoHub()
