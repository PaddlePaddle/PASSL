# Train MoCo Model

## Introduction

PASSL reproduces [PixPro](https://arxiv.org/abs/2011.10043), which is an unsupervised visual feature learning approach by leveraging pixel-level pretext task.

## Installation
- See [INSTALL.md](INSTALL.md)

## Data Preparation
- See [GETTING_STARTED.md](GETTING_STARTED.md)

## Implemented Models
Models are all trained with ResNet-50 backbone.
|  | epochs |official results | passl results | Backbone| Model |
| ---|--- | ----  | ---- | ----| ---- |
| PixPro  | 100 | 55.1(fp16) | 57.2(fp32) | ResNet-50 | [download](https://passl.bj.bcebos.com/models/pixpro_r50_ep100_no_instance_with_linear.pdparams)|

## Getting Started

### 1. Train PixPro

#### single gpu
```
python tools/train.py -c configs/pixpro/pixpro_base_r50_100ep.yaml
```

#### multiple gpus

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/pixpro/pixpro_base_r50_100ep.yaml
```

Pretraining models with 100 epochs can be found at [pixpro](https://passl
.bj.bcebos.com/models/pixpro_r50_ep100_no_instance_with_linear.pdparams)

Note: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```new_lr = old_lr * new_ngpus / old_ngpus```.

### 2. Extract backbone weights

```
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE}
```

### 3. Evaluation on ImageNet Linear Classification

#### Train:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/pixpro/pixpro_base_r50_100ep_IM_clas.yaml --pretrained ${WEIGHT_FILE}
```

#### Evaluate:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/pixpro/pixpro_base_r50_100ep_IM_clas.yaml --load ${CLS_WEGHT_FILE} --evaluate-only
```

