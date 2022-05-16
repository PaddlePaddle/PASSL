# Train SimSiam Model

## Introduction

PASSL reproduces [SimSiam](https://arxiv.org/abs/2011.10566), which is a simsiam network for unsupervised visual representation learning.

## Installation
- See [INSTALL.md](INSTALL.md)

## Data Preparation
- See [GETTING_STARTED.md](GETTING_STARTED.md)

## Implemented Models
Models are all trained with ResNet-50 backbone.
|  | epochs |official results | passl results | Backbone| Model |
| ---|--- | ----  | ---- | ----| ---- |
| SimSiam | 100 | 68.3 | 68.4          | ResNet-50 | [download](https://drive.google.com/file/d/1kaAm8-tlvB570kzI4fo9h4dwGQFf_4FE/view?usp=sharing) |


## Getting Started

### 1. Train SimSiam

#### single gpu
```
python tools/train.py -c configs/simsiam/simsiam_r50.yaml
```

#### multiple gpus

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/simsiam/simsiam_r50.yaml
```

Pretraining models with 100 epochs can be found at [simsiam](https://drive.google.com/file/d/1kaAm8-tlvB570kzI4fo9h4dwGQFf_4FE/view?usp=sharing)

Note: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```new_lr = old_lr * new_ngpus / old_ngpus```.

### 2. Extract backbone weights

```
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE} --prefix encoder --remove_prefix
```

### 3. Evaluation on ImageNet Linear Classification

#### Train:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/simsiam/simsiam_clas_r50.yaml --pretrained ${WEIGHT_FILE}
```

#### Evaluate:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/simsiam/simsiam_clas_r50.yaml --load ${CLS_WEGHT_FILE} --evaluate-only
```

The trained linear weights in conjuction with the backbone weights can be found at [simsiam linear](https://drive.google.com/file/d/19smHZGhBEPWeyLjKIGhM7KPngr-8BOUl/view?usp=sharing)
