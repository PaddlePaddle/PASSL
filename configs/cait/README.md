# CaiT: Going deeper with Image Transformers ([arxiv](https://arxiv.org/abs/2103.17239))

## Introduction

CaiT, or Class-Attention in Image Transformers, is a type of vision transformer with several design alterations upon the original ViT. First a new layer scaling approach called LayerScale is used, adding a learnable diagonal matrix on output of each residual block, initialized close to (but not at) 0, which improves the training dynamics. Secondly, class-attention layers are introduced to the architecture. This creates an architecture where the transformer layers involving self-attention between patches are explicitly separated from class-attention layers -- that are devoted to extract the content of the processed patches into a single vector so that it can be fed to a linear classifier.

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/a02234f9ccdc4100b027748c05052942cb9512a8604b49128b231f7cfc172302" alt="drawing" width="60%" height="60%"/>
</p>

For details see [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239) by Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve and Hervé Jégou

## Update

* **(10 Nov 2021)** Code is released and ported weights are uploaded

## Model Zoo

The results are evaluated on ImageNet2012 validation set

| Arch          | Weight                                                       | Top-1 Acc | Top-5 Acc | Crop ratio | # Params |
| ------------- | ------------------------------------------------------------ | --------- | --------- | ---------- | -------- |
| cait_s24_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s24_224.pdparams) | 83.45     | 96.57     | 1.0        | 46.8M    |
| cait_xs24_384 | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_x24_384.pdparams) | 84.06     | 96.89     | 1.0        | 26.5M    |
| cait_s24_384  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s24_384.pdparams) | 85.05     | 97.34     | 1.0        | 46.8M    |
| cait_s36_384  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s36_384.pdparams) | 85.45     | 97.48     | 1.0        | 68.1M    |
| cait_m36_384  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_m36_384.pdparams) | 86.06     | 97.73     | 1.0        | 270.7M   |
| cait_m48_448  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_m48_448.pdparams) | 86.49     | 97.75     | 1.0        | 355.8M   |

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


cfg_file = "configs/cait/cait_s24_224.yaml"
m = Model(cfg_file)
```

## Reference

```
@article{touvron2021cait,
  title={Going deeper with Image Transformers},
  author={Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Herv\'e J\'egou},
  journal={arXiv preprint arXiv:2103.17239},
  year={2021}
}
```
