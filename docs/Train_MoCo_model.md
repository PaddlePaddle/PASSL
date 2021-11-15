# Train MoCo Model

## Introduction

PASSL reproduces [MoCo](https://arxiv.org/abs/1911.05722), a way of building large and consistent dictionaries for **self-supervised learning** with a contrastive loss. It also includes the higher baseline [MoCo v2](https://arxiv.org/abs/1911.05722)

## Installation
- See [INSTALL.md](INSTALL.md)

## Data Preparation
- See [GETTING_STARTED.md](GETTING_STARTED.md)

## Implemented Models
Models are all trained with ResNet-50 backbone.
|  | epochs |official results | passl results | Backbone| Model |
| ---|--- | ----  | ---- | ----| ---- |
| MoCo  | 200 |  60.6| 60.64| ResNet-50 | [download](https://passl.bj.bcebos.com/models/moco_v1_r50_e200_ckpt.pdparams)|
| MoCo v2 | 200 | 67.7 | 67.72| ResNet-50 | [download](https://passl.bj.bcebos.com/models/moco_v2_r50_e200_ckpt.pdparams)|

## Getting Started

### 1. Train MoCo [v1,v2]

#### single gpu
```
python tools/train.py -c configs/moco/moco_v[1,2]_r50.yaml
```

#### multiple gpus

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/moco/moco_v[1,2]_r50.yaml
```

Pretraining models with 200 epochs can be found at [MoCo v1](https://passl.bj.bcebos.com/models/moco_v1_r50_e200_ckpt.pdparams) and [MoCo v2](https://passl.bj.bcebos.com/models/moco_v2_r50_e200_ckpt.pdparams)

Note: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```new_lr = old_lr * new_ngpus / old_ngpus```. Replacing v[1,2] to v1 or v2 according to your requriement.

### 2. Extract backbone weights

```
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE}
```

### 3. Evaluation on ImageNet Linear Classification

#### Train:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/moco/moco_clas_r50.yaml --pretrained ${WEIGHT_FILE}
```

#### Evaluate:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/moco/moco_clas_r50.yaml --load ${CLS_WEGHT_FILE} --evaluate-only
```

The trained linear weights in conjuction with the backbone weights can be found at [MoCo v1 linear](https://passl.bj.bcebos.com/models/moco_v1_r50_clas.pdparams) and [MoCo v2 linear](https://passl.bj.bcebos.com/models/moco_v2_r50_clas.pdparams)
