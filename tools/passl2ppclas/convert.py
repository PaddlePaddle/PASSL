# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import argparse
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Convert to paddleclas format')
    parser.add_argument('--type', type=str, help='model type', default='res50')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--output', type=str, help='destination file name')
    parser.add_argument('--ppclas', type=str, help='paddleclas keys')
    parser.add_argument('--passl', type=str, help='passals keys')
    args = parser.parse_args()
    return args


def get_ppclas_keys(file):
    keys = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            keys.append(line)

    return keys


def main():
    args = parse_args()
    assert args.output.endswith(".pdparams")
    ckpt = paddle.load(args.checkpoint)
    new_weight_dict = OrderedDict()

    if args.type == 'res50':
        ppclas_res18_keys = get_ppclas_keys('ppclas_res50_keys.txt')
        passl = get_ppclas_keys(args.ppclas)
        for i, weight in enumerate(passl):
            new_weight_dict[ppclas_res18_keys[i]] = ckpt['state_dict'][weight]

    else:
        raise ValueError(f'Invalid keywords "{args.type}"')

    paddle.save(new_weight_dict, args.output)


if __name__ == '__main__':
    main()
