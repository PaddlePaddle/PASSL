# PASSL

## Introduction
PASSL is a Paddle based vision library for state-of-the-art Self-Supervised Learning research with [PaddlePaddle](https://www.paddlepaddle.org.cn/). PASSL aims to accelerate research cycle in self-supervised learning: from designing a new self-supervised task to evaluating the learned representations.
- **Reproducible implementation of SOTA in Self-Supervision**: Existing SOTA in Self-Supervision are implemented - [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo(v1)](https://arxiv.org/abs/1911.05722),[MoCo(v2)](https://arxiv.org/abs/1911.05722), [MoCo-BYOL](), [CLIP](https://arxiv.org/abs/2103.00020). [BYOL](https://arxiv.org/abs/2006.07733) is coming soon. Also supports supervised trainings.
- **Modular**: Easy to build new tasks and reuse the existing components from other tasks (Trainer, models and heads, data transforms, etc.).

## Installation
- See [INSTALL.md](https://github.com/PaddlePaddle/PASSL/blob/main/docs/INSTALL.md).

## Implemented Models
Models are all trained with ResNet-50 backbone.
|  | epochs |official results | passl results | Backbone| Model |
| ---|--- | ----  | ---- | ----| ---- |
| MoCo  | 200 |  60.6| 60.64| ResNet-50 | [download](https://passl.bj.bcebos.com/models/moco_v1_r50_e200_ckpt.pdparams)|
| SimCLR | 100 | 64.5 | 65.3 | ResNet-50 | [download](https://passl.bj.bcebos.com/models/simclr_r50_ep100_ckpt.pdparams)|
| MoCo v2 | 200 | 67.7 | 67.72| ResNet-50 | [download](https://passl.bj.bcebos.com/models/moco_v2_r50_e200_ckpt.pdparams)|
| MoCo-BYOL | 300 | 71.56 | 72.10| ResNet-50 | [download](https://passl.bj.bcebos.com/models/mocobyol_r50_ep300_ckpt.pdparams)|
| BYOL | 300 | 72.50 | 71.62| ResNet-50 | [download](https://passl.bj.bcebos.com/models/byol_r50_300.pdparams)|

## Getting Started
Please see [GETTING_STARTED.md](https://github.com/PaddlePaddle/PASSL/blob/main/docs/GETTING_STARTED.md) for the basic usage of PASSL.

## Tutorials
- [Train SimCLR model](docs/Train_SimCLR_model.md)
- [Train MoCo(v1,v1) model](docs/Train_MoCo_model.md)
- [Train MoCo-BYOL model](docs/Train_MoCo-BYOL_model.md)
- [Train BYOL model](docs/Train_BYOL_model.md)
- [Train CLIP model](docs/Train_CLIP_model.md)
