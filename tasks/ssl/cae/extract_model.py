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
import paddle

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    type=str,
    default='output/ep800_fp16o1/ep800_fp16o1_checkpoint-799.pd')
parser.add_argument(
    '--output',
    type=str,
    default='output/ep800_fp16o1/cae_base_patch16_224_8k_vocab_pretrained_800ep.pd'
)

args = parser.parse_args()

checkpoint = paddle.load(args.input)
checkpoint_model = checkpoint['model']
paddle.save({'model': checkpoint_model}, args.output)
