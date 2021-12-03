# Train BYOL Model

## Introduction

[BYOL](https://arxiv.org/abs/2006.07733), a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks.it does not rely on negative pairs,which is a novel idea.  
the paddle reproduction is according to the [jax code](https://github.com/deepmind/deepmind-research/tree/master/byol), and we found it seemed the largest affected by image augmentations.you can find the [jax code](https://github.com/deepmind/deepmind-research/tree/master/byol) focusing more on image augmentations.at present,the paddle accuracy is about 71.62% at 300 epoch, while the paper shows its accuracy is about 72.5%.
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
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/byol/byol_r50_IM.yaml
```

Pretraining models with 300 epochs can be found at [BYOL](https://passl.bj.bcebos.com/models/byol_r50_300.pdparams).

Note: The default learning rate in config files is for 32 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```end_lr = base_lr * total_batch / 256```.

### 2. Extract backbone weights

```
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE}
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
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/byol/byol_clas_r50.yaml --pretrained ${WEIGHT_FILE}
```

#### Evaluate:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/byol/byol_clas_r50.yaml --load ${CLS_WEGHT_FILE} --evaluate-only
```

The trained linear weights in conjuction with the backbone weights can be found at [BYOL linear](https://passl.bj.bcebos.com/models/byol_r50_clas.pdparams)

