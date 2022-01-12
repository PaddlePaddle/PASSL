# ConvNeXt: A ConvNet for the 2020s

* (Update 2022-1-12) Code is released and ported weights are uploaded

## Introduction

In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually“modernize” a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/9d72cedda16846cbac5af5a7011afad069b6e7dd866746b380b68d82cf60847b" alt="drawing" width="50%" height="50%"/>
</p>


For details see [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf) by Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie.

## Model Zoo

The results are evaluated on ImageNet2012 validation set

| Arch               | Weight                                                       | Top-1 Acc | Top-5 Acc | Crop ratio | # Params | FLOPs |
| ------------------ | ------------------------------------------------------------ | --------- | --------- | ---------- | -------- | ----- |
| ConvNeXt_tiny_224  | [pretrain 1k](https://passl.bj.bcebos.com/models/convnext_small_1k_224.pdparams) | 82.1      | 95.67     | 0.875      | 28M      | -     |
| ConvNeXt_small_224 | [pretrain 1k](https://passl.bj.bcebos.com/models/convnext_small_1k_224.pdparams) | 83.1      | 96.92     | 0.875      | 50M      | -     |

Note: pretrain 1k is trained directly on the ImageNet-1k dataset

## Usage

Run the following code in the home directory

```python
import paddle
from passl.modeling.backbones import build_backbone
from passl.modeling.heads import build_head
from passl.utils.config import get_config


class Model(paddle.nn.Layer):
    def __init__(self, cfg_file):
        super().__init__()
        cfg = get_config(cfg_file)
        self.backbone = build_backbone(cfg.model.architecture)
        self.head = build_head(cfg.model.head)

    def forward(self, x):

        x = self.backbone(x)
        x = self.head(x)
        return x


cfg_file = "configs/convnext/convnext_tiny_224.yaml"
m = Model(cfg_file)


# infer test
x = paddle.randn([2, 3, 224, 224])
out = m(x)  # forward

loss = out.sum()
loss.backward()  # backward
print('Single iteration completed successfully')
```

## Reference

```
@Article{liu2021convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {arXiv preprint arXiv:2201.03545},
  year    = {2022},
}
```
