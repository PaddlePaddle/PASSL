# CvT: Introducing Convolutions to Vision Transformers

* (Update 2021-11-20) Code is released and ported weights are uploaded

## Introduction

We present in this paper a new architecture, named Convolutional vision Transformer (CvT), that improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs. This is accomplished through two primary modifications: a hierarchy of Transformers containing a new convolutional token embedding, and a convolutional Transformer block leveraging a convolutional projection. These changes introduce desirable properties of convolutional neural networks (CNNs) to the ViT architecture (\ie shift, scale, and distortion invariance) while maintaining the merits of Transformers (\ie dynamic attention, global context, and better generalization).

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/737455cc751c48aeb61f6cb359a6e91b45950b7eccb74335b709a6cc36d2e2e0" alt="drawing" width="90%" height="90%"/>
</p>

For details see [Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) by Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei.

## Model Zoo

The results are evaluated on ImageNet2012 validation set

| Arch        | Weight                                                       | Top-1 Acc | Top-5 Acc | Crop ratio | # Params |
| ----------- | ------------------------------------------------------------ | --------- | --------- | ---------- | -------- |
| cvt_13_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s24_224.pdparams) | 83.45     | 96.57     | 0.875      | 46.8M    |
| cvt_13_384  | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_x24_384.pdparams) | 84.06     | 96.89     | 1.0        | 26.5M    |
| cvt_21_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s24_384.pdparams) | 85.05     | 97.34     | 0.875      | 46.8M    |
| cvt_21_384  | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s36_384.pdparams) | 85.45     | 97.48     | 1.0        | 68.1M    |
| cvt_w24_384 | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_m36_384.pdparams) | 86.06     | 97.73     | 1.0        | 270.7M   |

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


cfg_file = "configs/cvt/cvt_13_224.yaml"
m = Model(cfg_file)
```

## Reference

```
@article{wu2021cvt,
  title={Cvt: Introducing convolutions to vision transformers},
  author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},
  journal={arXiv preprint arXiv:2103.15808},
  year={2021}
}
```
