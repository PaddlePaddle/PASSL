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
passl_gpu_model_list=( \
    ViT_base_patch16_224_in1k_1n8c_dp_fp16o2 \
    ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2 \
    DeiT_base_patch16_224_in1k_1n8c_dp_fp32 \
    DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2 \
    cait_s24_224_in1k_1n8c_dp_fp16o2 \
    swin_base_patch4_window7_224_fp16o2 \
    mae_vit_base_patch16_pt_in1k_1n8c_dp_fp16o1 \
    mae_vit_base_patch16_ft_in1k_1n8c_dp_fp16o1 \
    mae_vit_base_patch16_lp_in1k_1n8c_dp_fp16o1 \
    convmae_convvit_base_patch16_pt_in1k_1n8c_dp_fp16o1 \
    convmae_convvit_base_patch16_ft_in1k_1n8c_dp_fp16o1 \
    convmae_convvit_base_patch16_lp_in1k_1n8c_dp_fp16o1 \
    ConvNeXt_base_224_in1k_1n8c_dp_fp32 \
    cae_base_patch16_224_pt_in1k_1n8c_dp_fp16o1 \
    cae_base_patch16_224_ft_in1k_1n8c_dp_fp16o1 \
    cae_base_patch16_224_lp_in1k_1n8c_dp_fp16o1 \
    mocov3_vit_base_patch16_224_lp_in1k_1n8c_dp_fp16o1 \
)


###### ViT ######
function ViT_base_patch16_224_in1k_1n8c_dp_fp16o2() {
    cd ${passl_path}
    rm -rf log
    bash ./classification/vit/ViT_base_patch16_224_in1k_1n8c_dp_fp16o2.sh
    loss=`tail log/workerlog.0 | grep "49/313" | cut -d " " -f19 `
    ips=`cat log/workerlog.0 |grep ips: |cut -d " " -f25 |awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 10.47853 ${loss%?} 2140.74 ${ips} $FUNCNAME
}

function ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2() {
    cd ${passl_path}
    rm -rf log
    bash ./classification/vit/ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2.sh
    loss=`tail log/workerlog.0 | grep "49/2502" | cut -d " " -f19 `
    ips=`cat log/workerlog.0 |grep ips: |cut -d " " -f25 |awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.90351 ${loss%?} 420.1 ${ips} $FUNCNAME
}


###### DeiT ######
function DeiT_base_patch16_224_in1k_1n8c_dp_fp32() {
    cd ${passl_path}
    rm -rf log
    bash ./classification/deit/DeiT_base_patch16_224_in1k_1n8c_dp_fp32.sh
    loss=`tail log/workerlog.0 | grep "49/1251" | cut -d " " -f13 `
    ips=`cat log/workerlog.0 |grep ips: |cut -d " " -f19 |awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.90003 ${loss%?} 783.895 ${ips} $FUNCNAME
}


function DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2() {
    cd ${passl_path}
    rm -rf log
    bash ./classification/deit/DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2.sh
    loss=`tail log/workerlog.0 | grep "49/1251" | cut -d " " -f13 `
    ips=`cat log/workerlog.0 |grep ips: |cut -d " " -f19 |awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.90155 ${loss%?} 2079.68 ${ips} $FUNCNAME
}


###### CaiT ######
function cait_s24_224_in1k_1n8c_dp_fp16o2() {
    cd ${passl_path}
    rm -rf log
    bash ./classification/cait/cait_s24_224_in1k_1n8c_dp_fp16o2.sh
    loss=`tail log/workerlog.0 | grep "49/1251" | cut -d " " -f13 `
    ips=`cat log/workerlog.0 |grep ips: |cut -d " " -f19 |awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.93708 ${loss%?} 1824.29 ${ips} $FUNCNAME
}


###### Swin ######
function swin_base_patch4_window7_224_fp16o2() {
    cd ${passl_path}
    rm -rf log
    bash ./classification/swin/swin_base_patch4_window7_224_fp16o2.sh
    loss=`tail log/workerlog.0 | grep "49/1252" | cut -d " " -f13 `
    ips=`cat log/workerlog.0 |grep ips: |cut -d " " -f19 |awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 7.06612 ${loss%?} 944.051 ${ips} $FUNCNAME
}


###### MAE ######
function mae_vit_base_patch16_pt_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/mae/mae_vit_base_patch16_pt_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "199/1251" | cut -d " " -f15 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $10}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 1.0064 ${loss} 0.4965 ${ips} $FUNCNAME
}


function mae_vit_base_patch16_ft_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/mae/mae_vit_base_patch16_ft_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "599/5004" | cut -d " " -f15 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $10}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.7559 ${loss} 0.2332 ${ips} $FUNCNAME
}


function mae_vit_base_patch16_lp_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/mae/mae_vit_base_patch16_lp_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "199/312" | cut -d " " -f14 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $10}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.6991 ${loss} 1.072845 ${ips} $FUNCNAME
}


###### ConvMAE ######
function convmae_convvit_base_patch16_pt_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/convmae/convmae_convvit_base_patch16_pt_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "99/2502" | cut -d " " -f16 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $10}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 1.5487 ${loss} 0.456938 ${ips} $FUNCNAME
}


function convmae_convvit_base_patch16_ft_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/convmae/convmae_convvit_base_patch16_ft_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "599/5004" | cut -d " " -f15 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $10}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.7890 ${loss} 0.2964 ${ips} $FUNCNAME
}


function convmae_convvit_base_patch16_lp_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/convmae/convmae_convvit_base_patch16_lp_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "199/1251" | cut -d " " -f15 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $10}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.9417 ${loss} 0.3474 ${ips} $FUNCNAME
}


###### ConvNeXt ######
function ConvNeXt_base_224_in1k_1n8c_dp_fp32() {
    cd ${passl_path}
    rm -rf log
    bash ./classification/convnext/ConvNeXt_base_224_in1k_1n8c_dp_fp32.sh
    loss=`tail log/workerlog.0 | grep "50/312" | cut -d " " -f13 `
    ips=`cat log/workerlog.0 |grep ips: |cut -d " " -f21 |awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.91436 ${loss%?} 708.5226 ${ips} $FUNCNAME
}


###### CAE ######
function cae_base_patch16_224_pt_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/cae/cae_base_patch16_224_pt_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "199/2502" | cut -d " " -f19 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $16}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 9.6744 ${loss} 0.54708 ${ips} $FUNCNAME
}


function cae_base_patch16_224_ft_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/cae/cae_base_patch16_224_ft_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "199/1251" | cut -d " " -f15 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $10}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.3034 ${loss} 2.49244 ${ips} $FUNCNAME
}


function cae_base_patch16_224_lp_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/cae/cae_base_patch16_224_lp_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "199/312" | cut -d " " -f14 `
    ips=`cat log/workerlog.0 |grep time: |awk -F: '{print $10}' |cut -d " " -f2|awk 'NR>20 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.7196 ${loss} 1.07848 ${ips} $FUNCNAME
}


###### MoCoV3 ######
function mocov3_vit_base_patch16_224_lp_in1k_1n8c_dp_fp16o1() {
    cd ${passl_path}
    rm -rf log
    bash ./ssl/mocov3/mocov3_vit_base_patch16_224_lp_in1k_1n8c_dp_fp16o1.sh
    loss=`tail log/workerlog.0 | grep "49/1252" | cut -d " " -f19 `
    ips=`cat log/workerlog.0 |grep ips: |cut -d " " -f25 |awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    check_result 6.57017 ${loss} 3795.44 ${ips} $FUNCNAME
}


function check_result() {
    if [ $? -ne 0 ];then
      echo -e "\033 $5 model runs failed! \033" | tee -a $log_path/result.log
      exit -1
    fi

    echo -e "loss_base: $1 loss_test: $2" | tee -a $log_path/result.log
    if [ $1 != $2 ];then
      echo -e "\033 $5 loss diff check failed! \033" | tee -a $log_path/result.log
      exit -1
    fi

    diff=$(echo $3 $4|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "ips_base: $3 ips_test: $4 ips_diff: $diff% " | tee -a $log_path/result.log
    if [ $5 == mae_vit_base_patch16_lp_in1k_1n8c_dp_fp16o1 ];then
        v1=$(echo $diff 10.0|awk '{print($1>=$2)?"0":"1"}')
        v2=$(echo $diff -10.0|awk '{print($1<=$2)?"0":"1"}')
    else
        v1=$(echo $diff 5.0|awk '{print($1>=$2)?"0":"1"}')
        v2=$(echo $diff -5.0|awk '{print($1<=$2)?"0":"1"}')
    fi
    if [[ $v1 == 0 ]] || [[ $v2 == 0 ]];then
      echo -e "\033 $5 ips diff check failed! \033" | tee -a $log_path/result.log
      exit -1
    fi
}

function run_gpu_models(){
    cd
      for model in ${passl_gpu_model_list[@]}
      do
        echo "=========== ${model} run begin ==========="
        $model
        echo "=========== ${model} run  end ==========="
      done
}

main() {
    run_gpu_models
}

main$@
