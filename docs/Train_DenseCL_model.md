# Train DenseCL Model

## Introduction

PASSL reproduces [DenseCL](https://arxiv.org/abs/2011.09157), a self-supervised model for dense prediction tasks.

## Installation
- See [INSTALL.md](INSTALL.md)

## Data Preparation
- See [GETTING_STARTED.md](GETTING_STARTED.md)

## Implemented Models
Models are all trained with ResNet-50 backbone.
|  | epochs |official results | passl results | Backbone| Model |
| ---|--- | ----  | ---- | ----| ---- |
| DenseCL | 200 | 63.62 | 64.61 | ResNet-50 | [download](https://drive.google.com/file/d/1RWPO_g-fNJv8FsmCZ3LUbPTgPwtx-ybZ/view?usp=sharing) |

## Getting Started

### 1. Train DenseCL

#### single gpu
```
python tools/train.py -c configs/densecl/densecl_r50.yaml
```

#### multiple gpus

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/densecl/densecl_r50.yaml
```

Pretraining models with 200 epochs can be found at [DenseCL](https://drive.google.com/file/d/1RWPO_g-fNJv8FsmCZ3LUbPTgPwtx-ybZ/view?usp=sharing) 

Note: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```new_lr = old_lr * new_ngpus / old_ngpus```.

### 2. Extract backbone weights

```
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE} --remove_prefix
```

* Support PaddleClas

Convert the format of the extracted weights to the corresponding format of paddleclas to facilitate training on paddleclas

```
python tools/passl2ppclas/convert.py --type res50 --checkpoint ${CHECKPOINT} --output ${WEIGHT_FILE}
```

> Note: It must be ensured that the weights are extracted

### 3. Evaluation on ImageNet Linear Classification

#### Train:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/moco/moco_clas_r50.yaml --pretrained ${WEIGHT_FILE}
```

#### Evaluate:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/moco/moco_clas_r50.yaml --load ${CLS_WEGHT_FILE} --evaluate-only
```

The trained linear weights in conjuction with the backbone weights can be found at [DenseCL linear](https://drive.google.com/file/d/1XJeDY8clKfhUeXw4JcCa1QgG2G-Ibr4m/view?usp=sharing)
