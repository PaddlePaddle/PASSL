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

tmp_my_name=linprobe_ep90_fp16o1
my_name=${tmp_my_name%.*}
OUTPUT_DIR='./output/'$my_name
echo $OUTPUT_DIR
DATA_PATH='./dataset/ILSVRC2012/'
MODEL_PATH='pretrained/cae/cae_base_patch16_224_8k_vocab_pretrained_800ep.pd'
FLAGS_cudnn_exhaustive_search=True
export FLAGS_gemm_use_half_precision_compute_type=False

python -m paddle.distributed.launch  \
  --nnodes=$PADDLE_NNODES \
  --master=$PADDLE_MASTER \
  --devices=$CUDA_VISIBLE_DEVICES \
  ../../tasks/ssl/cae/main_linprobe.py \
  --print_freq 1 \
  --max_train_step 200 \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_base_patch16_224 \
  --finetune $MODEL_PATH \
  --nb_classes 1000 \
  --batch_size 512 \
  --epochs 90 \
  --blr 0.1 \
  --weight_decay 0.0 \
  --dist_eval \
  --log_dir $OUTPUT_DIR \
  --enable_linear_eval \
  --use_cls \
  --save_freq 50 \
  --disable_rel_pos_bias \
  --linear_type standard \
  --exp_name $my_name
