# Train SwAV Model

## Introduction

PASSL reproduces [SwAV](https://arxiv.org/abs/2006.09882). SwAV is an online algorithm that takes advantage of contrastive methods without requiring to compute pairwise comparisons. Compared to previous contrastive methods, SwAV is more memory efficient since it does not require a large memory bank or a special momentum network

## Installation
- See [INSTALL.md](INSTALL.md)

## Data Preparation
- See [GETTING_STARTED.md](GETTING_STARTED.md)

## Implemented Models
Models are all trained with ResNet-50 backbone.
|  | epochs |official results | passl results | Backbone| Model |
| ---|--- | ----  | ---- | ----| ---- |
| SwAV | 100 | 72.1 | 72.4      | ResNet-50 | [download](https://drive.google.com/file/d/1budFSoQqZz1Idyej-R4E6kUnL8CGtdyu/view?usp=sharing) |


## Getting Started

### 1. Train SwAV

#### single gpu
```
python tools/train.py -c configs/swav/swav_r50_100ep.yaml
```

#### multiple gpus

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/swav/swav_r50_100ep.yaml
```

Pretraining models with 100 epochs can be found at [swav](https://drive.google.com/file/d/1budFSoQqZz1Idyej-R4E6kUnL8CGtdyu/view?usp=sharing)

Note: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```new_lr = old_lr * new_ngpus / old_ngpus```.

### 2. Extract backbone weights

```
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE} --remove_prefix
```

### 3. Evaluation on ImageNet Linear Classification

#### Train:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/swav/swav_clas_r50.yaml --pretrained ${WEIGHT_FILE}
```

#### Evaluate:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/swav/swav_clas_r50.yaml --load ${CLS_WEGHT_FILE} --evaluate-only
```

The trained linear weights in conjuction with the backbone weights can be found at [swav linear](https://drive.google.com/file/d/1uduDAqJqK1uFclhQSK0d9RjzGNYR_Tj2/view?usp=sharing)
