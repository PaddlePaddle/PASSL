# MLP-Mixer: An all-MLP Architecture for Vision ([arxiv](https://arxiv.org/abs/2105.01601))

* **(Update 2021-11-15)**  Code is released and ported weights are uploaded

## Introduction

Convolutional Neural Networks (CNNs) are the go-to model for computer vision. Recently, attention-based networks, such as the Vision Transformer, have also become popular. In this paper we show that while convolutions and attention are both sufficient for good performance, neither of them are necessary. We present MLP-Mixer, an architecture based exclusively on multi-layer perceptrons (MLPs). MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information). When trained on large datasets, or with modern regularization schemes, MLP-Mixer attains competitive scores on image classification benchmarks, with pre-training and inference cost comparable to state-of-the-art models. We hope that these results spark further research beyond the realms of well established CNNs and Transformers.

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/fdaaa4ba00a24841b6d0875820672a59613216fb96464c678f8801dfd4edcfdf" alt="drawing" width="80%" height="80%"/>
</p>

For details see [An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601v4.pdf) by Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Jiang, Zi-Hang and Tay, Francis E.H. and Feng, Jiashi and Yan, Shuicheng

## Model Zoo

The results are evaluated on ImageNet2012 validation set

| Arch              | Weight                                                       | Top-1 Acc | Top-5 Acc | Crop ratio | # Params |
| ----------------- | ------------------------------------------------------------ | --------- | --------- | ---------- | -------- |
| mlp_mixer_b16_224 | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/mlp_mixer/mlp-mixer_b16_224.pdparams) | 76.60     | 92.23     | 0.875      | 60.0M    |
| mlp_mixer_l16_224 | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/mlp_mixer/mlp-mixer_l16_224.pdparams) | 72.06     | 87.67     | 0.875      | 208.2M   |

Note: pretrain 1k is trained directly on the ImageNet-1k dataset

## Usage

```python
from passl.modeling.backbones import build_backbone
from passl.modeling.heads import build_head
from passl.utils.config import get_config


class Model(nn.Layer):
    def __init__(self, cfg_file):
        super().__init__()
        cfg = get_config(cfg_file)
        self.backbone = build_backbone(cfg.model.architecture)
        self.head = build_head(cfg.model.head)

    def forward(self, x):

        x = self.backbone(x)
        x = self.head(x)
        return x


cfg_file  = 'configs/mlp_mixer/mlp-mixer_b16_224.yaml'
m = Model(cfg_file)
```

## Reference

```
@article{tolstikhin2021mlp,
  title={Mlp-mixer: An all-mlp architecture for vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and others},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}
```
