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

#unset PADDLE_TRAINER_ENDPOINTS
#export PADDLE_NNODES=4
#export PADDLE_MASTER="10.67.228.16:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export PADDLE_JOB_ID=MAE

# If you use single node
# batch_size 64, ACCUM_ITER=8, effective batch size: 4096
# batch_size 256, ACCUM_ITER=2, effective batch size: 4096

# 4 nodes for pretrain
ACCUM_ITER=1
IMAGENET_DIR=./dataset/ILSVRC2012/
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    main_pretrain.py \
    --accum_iter $ACCUM_ITER \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
