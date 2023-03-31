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
#export PADDLE_JOB_ID=ConvMAE

IMAGENET_DIR=./dataset/ILSVRC2012/

# 1 for four node, 4 for single node
ACCUM_ITER=1
PRETRAIN_CHKPT='./output_dir/checkpoint-1599.pd'
python -m paddle.distributed.launch \
   --nnodes=$PADDLE_NNODES \
   --master=$PADDLE_MASTER \
   --devices=$CUDA_VISIBLE_DEVICES \
   main_linprobe.py \
   --accum_iter $ACCUM_ITER \
   --batch_size 128 \
   --model convvit_base_patch16 \
   --global_pool \
   --finetune ${PRETRAIN_CHKPT} \
   --epochs 90 \
   --blr 0.1 \
   --weight_decay 0.0 \
   --dist_eval --data_path ${IMAGENET_DIR}

#export CUDA_VISIBLE_DEVICES=0
#python -m paddle.distributed.launch \
#    --nnodes=$PADDLE_NNODES \
#    --master=$PADDLE_MASTER \
#    --devices=$CUDA_VISIBLE_DEVICES \
#    main_linprobe.py --eval \
#    --resume output_dir/checkpoint-88.pd \
#    --model vit_base_patch16 \
#    --batch_size 512 \
#    --data_path ${IMAGENET_DIR}
