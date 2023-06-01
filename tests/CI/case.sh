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

#!/usr/bin/env bash
set -e

export passl_path=/paddle/PASSL/tests/CI
export log_path=/paddle/log_passl

function model_list(){
    ViT_base_patch16_224_in1k_1n8c_dp_fp16o2
    ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2
    DeiT_base_patch16_224_in1k_1n8c_dp_fp32
    DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2
    cait_s24_224_in1k_1n8c_dp_fp16o2
    swin_base_patch4_window7_224_fp16o2
    ConvNeXt_base_224_in1k_1n8c_dp_fp32
    mae_vit_base_patch16_pt_in1k_1n8c_dp_fp16o1
    mae_vit_base_patch16_ft_in1k_1n8c_dp_fp16o1
    mae_vit_base_patch16_lp_in1k_1n8c_dp_fp16o1
    convmae_convvit_base_patch16_pt_in1k_1n8c_dp_fp16o1
    convmae_convvit_base_patch16_ft_in1k_1n8c_dp_fp16o1
    convmae_convvit_base_patch16_lp_in1k_1n8c_dp_fp16o1
    cae_base_patch16_224_pt_in1k_1n8c_dp_fp16o1
    cae_base_patch16_224_ft_in1k_1n8c_dp_fp16o1
    cae_base_patch16_224_lp_in1k_1n8c_dp_fp16o1
    mocov3_vit_base_patch16_224_pt_in1k_1n8c_dp_fp16o1
    mocov3_deit_base_patch16_224_ft_in1k_1n8c_dp_fp16o1
    mocov3_vit_base_patch16_224_lp_in1k_1n8c_dp_fp16o1
    simsiam_resnet50_pt_in1k_1n8c_dp_fp32
    simsiam_resnet50_lp_in1k_1n8c_dp_fp32
    swav_resnet50_224_ft_in1k_1n4c_dp_fp32
    swav_resnet50_224_lp_in1k_1n8c_dp_fp32
    swav_resnet50_224_pt_in1k_1n8c_dp_fp16o1
    dino_deit_small_patch16_224_lp_in1k_1n8c_dp_fp16o1
    dinov2_vit_small_patch14_224_lp_in1k_1n8c_dp_fp16o1
}

############ case start ############

###### ViT ######
function ViT_base_patch16_224_in1k_1n8c_dp_fp16o2() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./classification/vit/ViT_base_patch16_224_in1k_1n8c_dp_fp16o2.sh
    
    loss=`cat log/workerlog.0 | grep '49/313' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/313' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=10.47853
    ips_base=2140.74
    mem_base=21.04
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./classification/vit/ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2.sh
    
    loss=`cat log/workerlog.0 | grep '49/2502' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/2502' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.90351
    ips_base=420.1
    mem_base=10.04
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### DeiT ######
function DeiT_base_patch16_224_in1k_1n8c_dp_fp32() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./classification/deit/DeiT_base_patch16_224_in1k_1n8c_dp_fp32.sh

    loss=`cat log/workerlog.0 | grep '49/1251' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/1251' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.90003
    ips_base=783.895
    mem_base=11.40
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./classification/deit/DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2.sh

    loss=`cat log/workerlog.0 | grep '49/1251' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/1251' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.90155
    ips_base=2079.68
    mem_base=11.40
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### CaiT ######
function cait_s24_224_in1k_1n8c_dp_fp16o2() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./classification/cait/cait_s24_224_in1k_1n8c_dp_fp16o2.sh

    loss=`cat log/workerlog.0 | grep '49/1251' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/1251' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.93708
    ips_base=1824.29
    mem_base=17.53
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### Swin ######
function swin_base_patch4_window7_224_fp16o2() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./classification/swin/swin_base_patch4_window7_224_fp16o2.sh

    loss=`cat log/workerlog.0 | grep '49/1252' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/1252' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=7.06580
    ips_base=944.051
    mem_base=17.52
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### MAE ######
function mae_vit_base_patch16_pt_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/mae/mae_vit_base_patch16_pt_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '199/1251' | awk -F 'loss: ' '{print $2}' | awk -F '  time' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '199/1251' | awk -F 'max mem: ' '{print $2}'`
    loss_base=1.0064
    ips_base=0.4965
    mem_base=13245
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function mae_vit_base_patch16_ft_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/mae/mae_vit_base_patch16_ft_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '599/5004' | awk -F 'loss: ' '{print $2}' | awk -F '  time' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '599/5004' | awk -F 'max mem: ' '{print $2}'`
    loss_base=6.7559
    ips_base=0.2332
    mem_base=7167
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function mae_vit_base_patch16_lp_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/mae/mae_vit_base_patch16_lp_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '199/312' | awk -F 'loss: ' '{print $2}' | awk -F '  time' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '199/312' | awk -F 'max mem: ' '{print $2}'`
    loss_base=6.6991
    ips_base=1.072845
    mem_base=6550
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### ConvMAE ######
function convmae_convvit_base_patch16_pt_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/convmae/convmae_convvit_base_patch16_pt_in1k_1n8c_dp_fp16o1.sh
    
    loss=`cat log/workerlog.0 | grep '99/2502' | awk -F 'loss: ' '{print $2}' | awk -F '  time' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '99/2502' | awk -F 'max mem: ' '{print $2}'`
    loss_base=1.5487
    ips_base=0.456938
    mem_base=14574
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function convmae_convvit_base_patch16_ft_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/convmae/convmae_convvit_base_patch16_ft_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '599/5004' | awk -F 'loss: ' '{print $2}' | awk -F '  time' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '599/5004' | awk -F 'max mem: ' '{print $2}'`
    loss_base=6.7890
    ips_base=0.2964
    mem_base=9896
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function convmae_convvit_base_patch16_lp_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/convmae/convmae_convvit_base_patch16_lp_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '199/1251' | awk -F 'loss: ' '{print $2}' | awk -F '  time' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '199/1251' | awk -F 'max mem: ' '{print $2}'`
    loss_base=6.9417
    ips_base=0.3474
    mem_base=5940
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### ConvNeXt ######
function ConvNeXt_base_224_in1k_1n8c_dp_fp32() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./classification/convnext/ConvNeXt_base_224_in1k_1n8c_dp_fp32.sh

    loss=`cat log/workerlog.0 | grep '50/312' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '50/312' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.91222
    ips_base=708.5226
    mem_base=18.38
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### CAE ######
function cae_base_patch16_224_pt_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/cae/cae_base_patch16_224_pt_in1k_1n8c_dp_fp16o1.sh
    
    loss=`cat log/workerlog.0 | grep '199/2502' | awk -F 'loss: ' '{print $2}' | awk -F ' ' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '199/2502' | awk -F 'max mem: ' '{print $2}'`
    loss_base=9.6744
    ips_base=0.54708
    mem_base=10789
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function cae_base_patch16_224_ft_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/cae/cae_base_patch16_224_ft_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '199/1251' | awk -F 'loss: ' '{print $2}' | awk -F ' ' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '199/1251' | awk -F 'max mem: ' '{print $2}'`
    loss_base=6.3033
    ips_base=2.49244
    mem_base=21131
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function cae_base_patch16_224_lp_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/cae/cae_base_patch16_224_lp_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '199/312' | awk -F 'loss: ' '{print $2}' | awk -F ' ' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'time: ' | awk -F 'time: ' '{print $2}' | awk -F '  data:' '{print $1}'| awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '199/312' | awk -F 'max mem: ' '{print $2}'`
    loss_base=6.7196
    ips_base=1.07848
    mem_base=5599
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### MoCoV3 ######
function mocov3_vit_base_patch16_224_pt_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/mocov3/mocov3_vit_base_patch16_224_pt_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '49/2503' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/2503' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=4.43806
    ips_base=539.487
    mem_base=16.66
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function mocov3_deit_base_patch16_224_ft_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/mocov3/mocov3_deit_base_patch16_224_ft_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '49/1251' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/1251' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.90772
    ips_base=1536.56
    mem_base=18.65
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function mocov3_vit_base_patch16_224_lp_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/mocov3/mocov3_vit_base_patch16_224_lp_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '49/1252' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '49/1252' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.57024
    ips_base=3795.44
    mem_base=1.53
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

###### SimSiam ######

function simsiam_resnet50_pt_in1k_1n8c_dp_fp32() {
      echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/simsiam/simsiam_resnet50_pt_in1k_1n8c_dp_fp32.sh

    loss=`cat log/workerlog.0 | grep '50/2502' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '50/2502' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=-0.32798
    ips_base=1731.37
    mem_base=10.55
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function simsiam_resnet50_lp_in1k_1n8c_dp_fp32() {
      echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/simsiam/simsiam_resnet50_lp_in1k_1n8c_dp_fp32.sh

    loss=`cat log/workerlog.0 | grep '50/313' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '50/313' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.89298
    ips_base=6285.21
    mem_base=5.38
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function swav_resnet50_224_ft_in1k_1n4c_dp_fp32() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/swav/swav_resnet50_224_ft_in1k_1n4c_dp_fp32.sh

    loss=`cat log/workerlog.0 | grep '120/126' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '120/126' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=2.01301
    ips_base=1919.8
    mem_base=10.50
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function swav_resnet50_224_lp_in1k_1n8c_dp_fp32() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/swav/swav_resnet50_224_lp_in1k_1n8c_dp_fp32.sh

    loss=`cat log/workerlog.0 | grep '200/5005' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '200/5005' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=3.83529
    ips_base=5620.26
    mem_base=0.46
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function swav_resnet50_224_pt_in1k_1n8c_dp_fp16o1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/swav/swav_resnet50_224_pt_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '200/2599' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '200/2599' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=7.93896
    ips_base=1000.3
    mem_base=8.37
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### DINO ######
function dino_deit_small_patch16_224_lp_in1k_1n8c_dp_fp16o1() {
      echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/dino/dino_deit_small_patch16_224_lp_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '100/5005' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '100/5005' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.87701
    ips_base=4966.58
    mem_base=0.30
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


###### DINOv2 ######
function dinov2_vit_small_patch14_224_lp_in1k_1n8c_dp_fp16o1() {
      echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/dinov2/dinov2_vit_small_patch14_224_lp_in1k_1n8c_dp_fp16o1.sh

    loss=`cat log/workerlog.0 | grep '100/10010' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '100/10010' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=6.86828
    ips_base=3950.49
    mem_base=0.21
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}


function check_result() {
    if [ $? -ne 0 ];then
      echo -e "\033 $1 model runs failed! \033" | tee -a $log_path/result.log
      exit -1
    fi

    if [ $# -ne 7 ]; then
        echo -e "\033 parameter transfer failed: $@ \033" | tee -a $log_path/result.log
        exit -1
    fi

    echo -e "loss_base: $2 loss_test: $3" | tee -a $log_path/result.log
    if [ $2 != $3 ];then
        echo -e "\033 $1 loss diff check failed! \033" | tee -a $log_path/result.log
        exit -1
    fi


    diff=$(echo $4 $5|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "ips_base: $4 ips_test: $5 ips_diff: $diff% " | tee -a $log_path/result.log
    # 设置不同ips校验阈值
    if [ $1 == mae_vit_base_patch16_lp_in1k_1n8c_dp_fp16o1 ];then
        v1=$(echo $diff 10.0|awk '{print($1>=$2)?"0":"1"}')
        v2=$(echo $diff -10.0|awk '{print($1<=$2)?"0":"1"}')
    else
        v1=$(echo $diff 5.0|awk '{print($1>=$2)?"0":"1"}')
        v2=$(echo $diff -5.0|awk '{print($1<=$2)?"0":"1"}')
    fi
    if [[ $v1 == 0 ]] || [[ $v2 == 0 ]];then
      echo -e "\033 $1 ips diff check failed! \033" | tee -a $log_path/result.log
      exit -1
    fi

    echo -e "mem_base: $6 mem_test: $7" | tee -a $log_path/result.log
    if [ $6 != $7 ];then
      echo -e "\033 $1 mem diff check failed! \033" | tee -a $log_path/result.log
      exit -1
    fi

}


main() {
    cd ${passl_path}
    
    model_list
}

main$@
