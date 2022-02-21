## XCiT: Cross-Covariance Image Transformers ([arxiv](https://arxiv.org/abs/2106.09681))

## Introduction

Following tremendous success in natural language processing, transformers have recently shown much promise for computer vision. The self-attention operation underlying transformers yields global interactions between all tokens, i.e. words or image patches, and enables flexible modelling of image data beyond the local interactions of convolutions. This flexibility, however, comes with a quadratic complexity in time and memory, hindering application to long sequences and high-resolution images. We propose a *transposed* version of self-attention that operates across feature channels rather than tokens, where the interactions are based on the cross-covariance matrix between keys and queries. The resulting cross-covariance attention (XCA) has linear complexity in the number of tokens, and allows efficient processing of high-resolution images. Our cross-covariance image transformer (XCiT) – built upon XCA – combines the accuracy of conventional transformers with the scalability of convolutional architectures. We validate the effectiveness and generality of XCiT by reporting excellent results on multiple vision benchmarks, includ- ing (self-supervised) image classification on ImageNet-1k, object detection and instance segmentation on COCO, and semantic segmentation on ADE20k.

![XCiT](https://user-images.githubusercontent.com/42234328/154954202-e51e6c9d-68af-4f42-b466-2db3a82fd19a.png)

## Getting Started

#### Train with single gpu
```bash
python tools/train.py -c configs/xcit/${XCIT_ARCH}.yaml
```
#### Train with multiple gpus

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/xcit/${XCIT_ARCH}.yaml
```
#### Evaluate
```bash
python tools/train.py -c configs/xcit/${XCIT_ARCH}.yaml --load ${XCIT_WEGHT_FILE} --evaluate-only
```

## Model Zoo

The results are evaluated on ImageNet2012 validation set
| Arch               | Weight                                                       | Top-1 Acc | Top-5 Acc | Crop ratio | # Params |
| ------------------ | ------------------------------------------------------------ | --------- | --------- | ---------- | -------- |
| xcit_nano_12_p8_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_nano_12_p8_224.pdparams) | 73.90   | 92.13 | 1.0 | 3.05M |
| xcit_tiny_12_p8_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_tiny_12_p8_224.pdparams) | 79.68   | 95.04 | 1.0 | 6.71M |
| xcit_tiny_24_p8_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_tiny_24_p8_224.pdparams) | 81.87   | 95.97 | 1.0 | 12.11M |
| xcit_small_12_p8_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_small_12_p8_224.pdparams) | 83.36   | 96.51 | 1.0 | 26.21M |
| xcit_small_24_p8_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_small_24_p8_224.pdparams) | 83.82   | 96.65 | 1.0 | 47.63M |
| xcit_medium_24_p8_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_medium_24_p8_224.pdparams ) | 83.73 | 96.39 | 1.0 | 84.32M |
| xcit_large_24_p8_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_large_24_p8_224.pdparams) | 84.42  | 96.65 | 1.0 | 188.93M |
| xcit_nano_12_p16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_nano_12_p16_224.pdparams) | 70.01 | 89.82 | 1.0 | 3.05M |
| xcit_tiny_12_p16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_tiny_12_p16_224.pdparams) | 77.15    | 93.72 | 1.0 | 6.72M |
| xcit_tiny_24_p16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_tiny_24_p16_224.pdparams) | 79.42    | 94.86 | 1.0 | 12.12M |
| xcit_small_12_p16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_small_12_p16_224.pdparams) | 81.89 | 95.83 | 1.0 | 26.25M |
| xcit_small_24_p16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_small_24_p16_224.pdparams) | 82.51   | 95.97 | 1.0 | 47.67M |
| xcit_medium_24_p16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_medium_24_p16_224.pdparams) | 82.67   | 95.91 | 1.0 | 84.40M |
| xcit_large_24_p16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/pvt_v2/xcit_large_24_p16_224.pdparams) | 82.89   | 95.89 | 1.0 | 189.10M |


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


cfg_file = "configs/xcit/xcit_nano_12_p8_224.yaml"
m = Model(cfg_file)
```

## Reference

```
@article{xcit,
      title={{XCiT}: Cross-Covariance Image Transformers}, 
      author={Alaaeldin El-Nouby and Hugo Touvron and Mathilde Caron and Piotr Bojanowski and Matthijs Douze and Armand Joulin and Ivan Laptev and Natalia Neverova and Gabriel Synnaeve and Jakob Verbeek and Hervé Jegou},
      year={2021},
      eprint={2106.09681},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```