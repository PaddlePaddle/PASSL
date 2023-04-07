 # Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

model_item=cae_base_patch16_224_pt
fp_item=fp16o1
bs_item=64
run_mode=DP8-MP1
device_num=N1C8
mode=pt
model=cae_base_patch16_224_8k_vocab
max_iter=5003 # epoch=2
PRETRAIN_CHKPT='pretrained/dalle-weights'

bash ./tests/test_tipc/ssl/cae/benchmark_common/prepare.sh
# run
bash ./tests/test_tipc/ssl/cae/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${bs_item} ${run_mode} ${device_num} \
${mode} ${model} ${max_iter} ${PRETRAIN_CHKPT} 2>&1;
