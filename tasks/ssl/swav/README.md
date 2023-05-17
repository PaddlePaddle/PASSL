## SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments


PaddlePaddle reimplementation of [facebookresearch's repository for the SwAV model](https://github.com/facebookresearch/swav) that was released with the paper [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882).

## Requirements
To enjoy some new features, PaddlePaddle develop is required. For more installation tutorials
refer to [installation.md](../../../tutorials/get_started/installation.md)

## Data Preparation

Prepare the data into the following directory:
```text
dataset/
└── ILSVRC2012
    ├── train
    └── val
```


## How to Self-supervised Pre-Training

With a batch size of 4096, SwAV is trained with 4 nodes:

```bash
# Note: Set the following environment variables
# and then need to run the script on each node.
unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=4
export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_stop_check_timeout=3600

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/swav_resnet50_224_pt_in1k_4n32c_dp_fp16o1.yaml
```

## How to Linear Classification
By default, we use momentum-SGD and a batch size of 256 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

- Download pretrained model
```bash
mkdir -p pretrained/swav
wget -O ./pretrained/swav/swav_resnet50_in1k_800ep_bz4096_pretrained.pdparams https://passl.bj.bcebos.com/models/swav/swav_resnet50_in1k_800ep_bz4096_pretrained.pdparams
```

- Train linear classification model

```bash
unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_stop_check_timeout=3600

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/swav_resnet50_224_lp_in1k_1n8c_dp_fp16o1.yaml
```

## How to End-to-End Fine-tuning
To perform end-to-end fine-tuning for SwAV:

* First download the data split text file with following commands:
    ```bash
    cd PASSL

    wget "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/10percent.txt"

    wget "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/1percent.txt"
    ```

* Then, download the pretrained models to `./pretrained/swav/swav_resnet50_in1k_800ep_bz4096_pretrained.pdparams`

- Download pretrained model
```bash
mkdir -p pretrained/swav
wget -O ./pretrained/swav/swav_resnet50_in1k_800ep_bz4096_pretrained.pdparams https://passl.bj.bcebos.com/models/swav/swav_resnet50_in1k_800ep_bz4096_pretrained.pdparams
```

* Finally, run the training with the trained PASSL format checkpoint:
    ```bash
    unset PADDLE_TRAINER_ENDPOINTS
    export PADDLE_NNODES=1
    export PADDLE_MASTER="127.0.0.1:12538"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export FLAGS_stop_check_timeout=3600

    python -m paddle.distributed.launch \
        --nnodes=$PADDLE_NNODES \
        --master=$PADDLE_MASTER \
        --devices=$CUDA_VISIBLE_DEVICES \
        passl-train \
        -c ./configs/swav_resnet50_224_ft_in1k_1n4c_dp_fp16o1.yaml
        -o Global.pretrained_model=./pretrained/swav/swav_resnet50_in1k_800ep_pretrained
    ```

## Other Configurations
We provide more directly runnable configurations, see [SwAV Configurations](./configs/).

## Models

### ViT-Base
| Model         | Phase       | Dataset      | Configs  | GPUs       | Epochs | Top1 Acc (%) | Links                                                   |
| ------------- | ----------- | ------------ | ------------------------------------------------------------ | ---------- | ------ | -------- | ------------------------------------------------------------ |
| resnet50 | pretrain    | ImageNet2012 | [config](./configs/swav_resnet50_224_pt_in1k_4n32c_dp_fp16o1.yaml) | A100*N4C32 | 800    | -        | [model](https://passl.bj.bcebos.com/models/swav/swav_resnet50_in1k_800ep_bz4096_pretrained.pdparams) \| [log](https://github.com/shiyutang/files/files/11493437/pretrain_train.log) |
| resnet50 | linear probe | ImageNet2012 | [config](./configs/swav_resnet50_224_lp_in1k_4n32c_dp_fp16o1.yaml) | A100*N1C8  |  100  | 75.3    |        [model](https://passl.bj.bcebos.com/models/swav/swav_resnet50_in1k_linearprobe.pdparams) \| [log](https://github.com/shiyutang/files/files/11493435/linear_train.log) |
| resnet50 | finetune-semi10    | ImageNet2012 | [config](./configs/swav_resnet50_224_ft_in1k_1n4c_dp_fp16o1.yaml) | A100*N1C4  | 20    | 69.0   | [model](https://passl.bj.bcebos.com/models/swav/swav_resnet50_in1k_finetune_10percent.pdparams) \| [log](https://github.com/shiyutang/files/files/11493438/semi10_train.log) |
| resnet50 | finetune-semi10    | ImageNet2012 | [config](./configs/swav_resnet50_224_ft_in1k_1n4c_dp_fp16o1.yaml) | A100*N1C4  | 20    | 55.0   | [model](https://passl.bj.bcebos.com/models/swav/swav_resnet50_in1k_finetune_1percent.pdparams) \| [log](https://github.com/shiyutang/files/files/11493451/semi1.log) |
## Citations

```bibtex
@misc{caron2021unsupervised,
      title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
      author={Mathilde Caron and Ishan Misra and Julien Mairal and Priya Goyal and Piotr Bojanowski and Armand Joulin},
      year={2021},
      eprint={2006.09882},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

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
    swav_resnet50_224_ft_in1k_1n4c_dp_fp32
    swav_resnet50_224_lp_in1k_1n8c_dp_fp32
    swav_resnet50_224_pt_in1k_1n8c_dp_fp16o1
}

############ case start ############

function swav_resnet50_224_ft_in1k_1n4c_dp_fp32() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    bash ./ssl/swav/swav_resnet50_224_ft_in1k_1n4c_dp_fp32.sh

    loss=`cat log/workerlog.0 | grep '120/126' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat log/workerlog.0 | grep 'ips: ' | awk -F 'ips: ' '{print $2}' | awk -F ' images/sec,' '{print $1}'| awk 'NR>1 {print}' | awk '{a+=$1}END{print a/NR}'`
    mem=`cat log/workerlog.0 | grep '120/126' | awk -F 'max mem: ' '{print $2}' | awk -F ' GB,' '{print $1}'`
    loss_base=2.01301
    ips_base=1922.62626
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
    loss_base=4.89133
    ips_base=11111.52955
    mem_base=0.83
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
    loss_base=8.00343
    ips_base=1385.94186
    mem_base=8.63
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
