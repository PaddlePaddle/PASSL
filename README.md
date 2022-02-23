⚙️ English | [简体中文](./README_cn.md)

<p align="center">
  <img src="./docs/imgs/passl_logo.svg" width="60%" height="60%"/>
</p>
<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href="https://github.com/PaddlePaddle/PASSL/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PASSL?color=ccf"></a>
    <a href=""><img src="https://camo.githubusercontent.com/abb97269de2982c379cbc128bba93ba724d8822bfbe082737772bd4feb59cb54/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f643733303566333864323966656437386661383536353265336136336531353464643865383832392f6d656469612f62616467652e737667"></a>
  <a href="https://aistudio.baidu.com/aistudio/personalcenter/thirdview/940489"><img src="https://img.shields.io/badge/Tutorial-AI Studio-blue.svg"></a>
</p>

## Introduction

PASSL is a Paddle based vision library for state-of-the-art Self-Supervised Learning research with [PaddlePaddle](https://www.paddlepaddle.org.cn/). PASSL aims to accelerate research cycle in self-supervised learning: from designing a new self-supervised task to evaluating the learned representations.

Key features of PASSL：

- Reproducible implementation of SOTA in Self-Supervision

  Existing SOTA in Self-Supervision are implemented - [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo(v1)](https://arxiv.org/abs/1911.05722), [MoCo(v2)](https://arxiv.org/abs/1911.05722), [MoCo-BYOL](docs/Train_MoCo-BYOL_model.md), [BYOL](https://arxiv.org/abs/2006.07733), [BEiT](https://arxiv.org/abs/2106.08254). Supervised classification training is also supported.

- Modular Design

  Easy to build new tasks and reuse the existing components from other tasks (Trainer, models and heads, data transforms, etc.)

🛠️ The ultimate goal of PASSL is to use self-supervised learning to provide more appropriate pre-training weights for downstream tasks while significantly reducing the cost of data annotation.

**📣 Recent Update:**

* (2022-2-9): Refactoring README
* 🔥 Now：

## Implemented Models

* **Self-Supervised Learning Models**

PASSL implements a series of self-supervised learning algorithms, See **Document** for details on its use

|           | Epochs | Official results | PASSL results | Backbone  | Model                                                        | Document                                         |
| --------- | ------ | ---------------- | ------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------ |
| MoCo      | 200    | 60.6             | 60.64         | ResNet-50 | [download](https://passl.bj.bcebos.com/models/moco_v1_r50_e200_ckpt.pdparams) | [Train MoCo](docs/Train_MoCo_model.md)           |
| SimCLR    | 100    | 64.5             | 65.3          | ResNet-50 | [download](https://passl.bj.bcebos.com/models/simclr_r50_ep100_ckpt.pdparams) | [Train SimCLR](docs/Train_SimCLR_model.md)       |
| MoCo v2   | 200    | 67.7             | 67.72         | ResNet-50 | [download](https://passl.bj.bcebos.com/models/moco_v2_r50_e200_ckpt.pdparams) | [Train MoCo](docs/Train_MoCo_model.md)           |
| MoCo-BYOL | 300    | 71.56            | 72.10         | ResNet-50 | [download](https://passl.bj.bcebos.com/models/mocobyol_r50_ep300_ckpt.pdparams) | [Train MoCo-BYOL](docs/Train_MoCo-BYOL_model.md) |
| BYOL      | 300    | 72.50            | 71.62         | ResNet-50 | [download](https://passl.bj.bcebos.com/models/byol_r50_300.pdparams) | [Train BYOL](docs/Train_BYOL_model.md)           |
| PixPro    | 100    | 55.1(fp16)       | 57.2(fp32)    | ResNet-50 | [download](https://passl.bj.bcebos.com/models/pixpro_r50_ep100_no_instance_with_linear.pdparams) | [Train PixPro](docs/Train_PixPro_model.md)       |

> Benchmark Linear Image Classification on ImageNet-1K.

Comming Soon：More algorithm implementations are already in our plans ... 

* **Classification Models**

PASSL implements influential image classification algorithms such as Visual Transformer, and provides corresponding pre-training weights. Designed to support the construction and research of self-supervised, multimodal, large-model algorithms. See [Classification_Models_Guide.md](docs/Classification_Models_Guide.md) for more usage details

|                  | Detail                      | Tutorial                                                     |
| ---------------- | --------------------------- | ------------------------------------------------------------ |
| ViT              | /                           | [PaddleEdu](https://aistudio.baidu.com/aistudio/projectdetail/2293050) |
| Swin Transformer | /                           | [PaddleEdu](https://aistudio.baidu.com/aistudio/projectdetail/2280436) |
| CaiT             | [config](configs/cait)      | [PaddleFleet](https://aistudio.baidu.com/aistudio/projectdetail/3401469) |
| T2T-ViT          | [config](configs/t2t_vit)   | [PaddleFleet](https://aistudio.baidu.com/aistudio/projectdetail/3401348) |
| CvT              | [config](configs/cvt)       | [PaddleFleet](https://aistudio.baidu.com/aistudio/projectdetail/3401386) |
| BEiT             | [config](configs/beit)      | [unofficial](https://aistudio.baidu.com/aistudio/projectdetail/2417241) |
| MLP-Mixer        | [config](configs/mlp_mixer) | [PaddleFleet](https://aistudio.baidu.com/aistudio/projectdetail/3401295) |
| ConvNeXt         | [config](configs/convnext)  | [PaddleFleet](https://aistudio.baidu.com/aistudio/projectdetail/3407445) |

🔥 PASSL provides a detailed dissection of the algorithm, see **Tutorial** for details.

## Installation

See [INSTALL.md](https://github.com/PaddlePaddle/PASSL/blob/main/docs/INSTALL.md).

## Getting Started

Please see [GETTING_STARTED.md](https://github.com/PaddlePaddle/PASSL/blob/main/docs/GETTING_STARTED.md) for the basic usage of PASSL.

## Awesome SSL

Self-Supervised Learning (SSL) is a rapidly growing field, and some influential papers are listed here for research use.PASSL seeks to implement self-supervised algorithms with application potential

* *[Masked Feature Prediction for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2112.09133)* by Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, Christoph Feichtenhofer.
* *[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)* by Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.
* *[Corrupted Image Modeling for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2202.03382)* by Yuxin Fang, Li Dong, Hangbo Bao, Xinggang Wang, Furu Wei.
* *[Are Large-scale Datasets Necessary for Self-Supervised Pre-training?](https://arxiv.org/abs/2112.10740)* by Alaaeldin El-Nouby, Gautier Izacard, Hugo Touvron, Ivan Laptev, Hervé Jegou, Edouard Grave.
* *[PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers](https://arxiv.org/abs/2111.12710)* by Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu.
* *[SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)* by Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, Han Hu.

## Contributing

PASSL is still young. It may contain bugs and issues. Please report them in our bug track system. Contributions are welcome. Besides, if you have any ideas about PASSL, please let us know.

## Citation

If PASSL is helpful to your research, feel free to cite

```
@misc{=passl,
    title={PASSL: A visual Self-Supervised Learning Library},
    author={PASSL Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PASSL}},
    year={2022}
}
```

## License

As shown in the LICENSE.txt file, PASSL uses the Apache 2.0 copyright agreement.
