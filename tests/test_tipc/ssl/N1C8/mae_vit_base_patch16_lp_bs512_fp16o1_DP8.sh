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

model_item=mae_vit_base_patch16_lp
fp_item=fp16o1
bs_item=512
run_mode=DP8
device_num=N1C8
mode=lp
model=maevit_base_patch16
max_iter=1559 # epoch=5
PRETRAIN_CHKPT='pretrained/mae/mae_pretrain_vit_base_1599ep.pd'

bash ./tests/test_tipc/ssl/benchmark_common/prepare.sh
# run
bash ./tests/test_tipc/ssl/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${bs_item} ${run_mode} ${device_num} \
${mode} ${model} ${max_iter} ${PRETRAIN_CHKPT} 2>&1;
