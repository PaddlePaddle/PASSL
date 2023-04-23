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

export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.0:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_stop_check_timeout=3600

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ../../tasks/ssl/simsiam/configs/simsiam_resnet50_lp_in1k_1n8c_dp_fp32.yaml \
    -o Global.print_batch_step=1 \
    -o Global.max_train_step=51 \
    -o Global.flags.FLAGS_cudnn_exhaustive_search=0 \
    -o Global.flags.FLAGS_cudnn_deterministic=1 \
    -o Global.pretrained_model=./pretrained/simsiam/simsiam_resnet50_in1k_100ep_bz512_pretrained
