# Train MoCo Model

## Introduction

[BYOL](https://arxiv.org/abs/2006.07733), a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks.it does not rely on negative pairs,which is a novel idea.
the paddle reproduction is according to the [jax code](https://github.com/deepmind/deepmind-research/tree/master/byol), and we found it seemed the largest affected by image augmentations.you can find the [jax code](https://github.com/deepmind/deepmind-research/tree/master/byol) focusing more on image augmentations.
## Installation
- See [INSTALL.md](INSTALL.md)

## Data Preparation
- See [GETTING_STARTED.md](GETTING_STARTED.md)


## Getting Started

### 1. Train BYOL

#### single gpu
```
python tools/train.py -c configs/byol/byol_r50_IM.yaml
```

#### multiple gpus

```
python tools/train.py -c configs/byol/byol_r50_IM.yaml --num-gpus 8
```

Pretraining models with 300 epochs can be found at [BYOL](https://arxiv.org/abs/2006.07733).

Note: The default learning rate in config files is for 32 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```end_lr = base_lr * total_batch / 256```.

### 2. Extract backbone weights

```
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE}
```

### 3. Evaluation on ImageNet Linear Classification

#### Train:
```
python tools/train.py -c configs/byol/byol_clas_r50.yaml --pretrained ${WEIGHT_FILE} --num-gpus 8
```

#### Evaluate:
```
python tools/train.py -c configs/byol/byol_clas_r50.yaml --load ${CLS_WEGHT_FILE} --evaluate-only --num-gpus 8
```

The trained linear weights in conjuction with the backbone weights can be found at [BYOL linear](https://passl.bj.bcebos.com/models/moco_v1_r50_clas.pdparams) 

