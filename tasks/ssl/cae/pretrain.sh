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

# unset PADDLE_TRAINER_ENDPOINTS
# export PADDLE_NNODES=4
# export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PADDLE_JOB_ID=CAE

tmp_my_name=ep800_fp16o1
my_name=${tmp_my_name%.*}
OUTPUT_DIR='./output/'$my_name
echo $OUTPUT_DIR
DATA_PATH='./dataset/ILSVRC2012/'
TOKENIZER_PATH=dalle-weights
FLAGS_cudnn_exhaustive_search=True
export FLAGS_gemm_use_half_precision_compute_type=False

python -m paddle.distributed.launch  \
  --nnodes=$PADDLE_NNODES \
  --master=$PADDLE_MASTER \
  --devices=$CUDA_VISIBLE_DEVICES \
  main_pretrain.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --batch_size 64 --lr 1.5e-3 --warmup_epochs 10 --epochs 800 \
  --clip_grad 3.0 --layer_scale_init_value 0.1 \
  --imagenet_default_mean_and_std \
  --color_jitter 0 \
  --drop_path 0 \
  --sincos_pos_emb \
  --mask_generator block \
  --num_mask_patches 75 \
  --decoder_layer_scale_init_value 0.1 \
  --no_auto_resume \
  --save_ckpt_freq 50 \
  --exp_name $my_name \
  --regressor_depth 4 \
  --seed 0 \
  --log_dir vdl \
  --num_decoder_self_attention 4 \
  --dual_loss_weight 2 \
  --dual_path_ema 0 \
  --amp
