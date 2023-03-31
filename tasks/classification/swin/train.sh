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

# Note: Set the following environment variables 
# and then need to run the script on each node.

#export PADDLE_NNODES=1
#export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/swin_base_patch4_window7_224_in1k_1n8c_dp_fp16o2.yaml
