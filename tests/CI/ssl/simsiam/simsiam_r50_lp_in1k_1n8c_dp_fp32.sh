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

FLAGS_cudnn_exhaustive_search=0
FLAGS_cudnn_deterministic=1
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --devices=$CUDA_VISIBLE_DEVICES ../../tools_v110/train.py \
       -c ../../configs/simsiam/simsiam_clas_r50.yaml \
       -o total_iters=51 \
       -o seed=2023 \
       --pretrain ./pretrained/simsiam/simsiam_r50_ext_backbone.pd
