# Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet ([arxiv](https://arxiv.org/abs/2101.11986))

* **(Update 2021-11-10)**  Code is released and ported weights are uploaded

## Introduction

T2T-ViT (Tokens-To-Token Vision Transformer) is a type of Vision Transformer which incorporates 1) a layerwise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token (Tokens-to-Token), such that local structure represented by surrounding tokens can be modeled and tokens length can be reduced; 2) an efficient backbone with a deep-narrow structure for vision transformer motivated by CNN architecture design after empirical study.

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/16918b285967476691550c5769d1bda13add38a925704ac3822ea1df042f8fae" alt="drawing" width="80%" height="80%"/>
</p>


For details see [Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986) by Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Jiang, Zi-Hang and Tay, Francis E.H. and Feng, Jiashi and Yan, Shuicheng

## Model Zoo

The results are evaluated on ImageNet2012 validation set

| Arch         | Weight                                                       | Top-1 Acc | Top-5 Acc | Crop ratio | # Params |
| ------------ | ------------------------------------------------------------ | --------- | --------- | ---------- | -------- |
| t2t_vit_14   | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_14.pdparams) | 81.50     | 95.67     | 0.9        | 21.5M    |
| t2t_vit_19   | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_19.pdparams) | 81.93     | 95.74     | 0.9        | 39.1M    |
| t2t_vit_24   | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_24.pdparams) | 82.28     | 95.89     | 0.9        | 64.0M    |
| t2t_vit_t_14 | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_t_14.pdparams) | 81.69     | 95.85     | 0.9        | 21.5M    |
| t2t_vit_t_19 | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_t_19.pdparams) | 82.44     | 96.08     | 0.9        | 39.1M    |
| t2t_vit_t_24 | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_t_24.pdparams) | 82.55     | 96.07     | 0.9        | 64.0M    |

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


cfg_file = "configs/t2t_vit/t2t_vit_14.yaml"
m = Model(cfg_file)
```

## Reference

```
@InProceedings{Yuan_2021_ICCV,
    author    = {Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Jiang, Zi-Hang and Tay, Francis E.H. and Feng, Jiashi and Yan, Shuicheng},
    title     = {Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {558-567}
}
```
