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


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--prefix',
                        type=str,
                        default='backbone',
                        help='destination file name')
    parser.add_argument('--remove_prefix',
                        action='store_true',
                        help='remove prefix from keys of state dict')
    parser.add_argument('--output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pdparams")
    ckpt = paddle.load(args.checkpoint)
    output_dict = dict()
    has_prefix = False
    for key, value in ckpt['state_dict'].items():
        if key.startswith(args.prefix):
            if args.remove_prefix:
                key = key[len(args.prefix) + 1:]
            output_dict[key] = value
            has_prefix = True
    if not has_prefix:
        raise Exception("Cannot find a {} layer in the checkpoint.".format(
            args.prefix))
    paddle.save(output_dict, args.output)


if __name__ == '__main__':
    main()
