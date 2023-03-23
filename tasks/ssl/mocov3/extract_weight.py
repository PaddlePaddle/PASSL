# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import paddle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert MoCo Pre-Traind Model to DEiT')
    parser.add_argument(
        '--input',
        default='',
        type=str,
        metavar='PATH',
        required=True,
        help='path to moco pre-trained checkpoint')
    parser.add_argument(
        '--output',
        default='',
        type=str,
        metavar='PATH',
        required=True,
        help='path to output checkpoint in DEiT format')
    args = parser.parse_args()
    print(args)

    # load input
    checkpoint = paddle.load(args.input)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('base_encoder') and not k.startswith(
                'base_encoder.head'):
            # remove prefix
            state_dict[k[len("base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # make output directory if necessary
    output_dir = os.path.dirname(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # save to output
    paddle.save(state_dict, args.output)
