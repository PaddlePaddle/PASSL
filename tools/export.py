# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
paddle.disable_static()

from passl.utils import config as cfg_util
from passl.engine.engine import Engine


def main():
    args = cfg_util.parse_args()
    config = cfg_util.get_config(
        args.config, overrides=args.override, show=False)
    config.profiler_options = args.profiler_options
    engine = Engine(config, mode="export")
    engine.export()


if __name__ == "__main__":
    main()
