Train SimCLR Model

## Introduction
SimCLR is a simple framework for contrastive learning of visual representations. SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space(https://arxiv.org/abs/2002.05709).
## Installation
- See [INSTALL.md](INSTALL.md)

## Data Preparation
- See [GETTING_STARTED.md](GETTING_STARTED.md)

## Implemented Models
Models are all trained with ResNet-50 backbone.
|  | epochs |official results | passl results | Backbone| Model |
| ---|--- | ----  | ---- | ----| ---- |
| SimCLR  | 100 |  64.5| 64.8| ResNet-50 | [download](https://passl.bj.bcebos.com/models/simclr_r50_ep100_ckpt.pdparam)|

## Getting Started

### 1. Train SimCLR

#### single gpu

##### ImageNet
```
python3 tools/train.py -c configs/simclr/simclr_r50_IM.yaml
```

##### Cifar10

```
python3 tools/train.py -c configs/simclr/simclr_r18_cifar10.yaml
```

#### multiple gpus
##### ImageNet

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/simclr/simclr_r50_IM.yaml
```

##### Cifar10

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/simclr/simclr_r50_IM.yaml
```



Pretraining models with 200 epochs can be found at [SimCLR](https://passl.bj.bcebos.com/models/simclr_r50_ep100_ckpt.pdparam).
Note: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```new_lr = old_lr * new_ngpus / old_ngpus```. 
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

The trained linear weights in conjuction with the backbone weights can be found at [SimCLR linear](https://passl.bj.bcebos.com/models/simclr_r50_ep100_ckpt.pdparam).

