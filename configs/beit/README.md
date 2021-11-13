## BEiT: BERT Pre-Training of Image Transformers ([arxiv](https://arxiv.org/abs/2106.08254))

* **(Update 2021-11-11)**  Code is released and ported weights are uploaded

## Introduction

We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. Experimental results on image classification and semantic segmentation show that our model achieves competitive results with previous pre-training methods

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/498f6488c7174b428bd8a8a23fc00461ebdf9a094aff407daaec35f9cc307e62" alt="drawing" width="90%" height="90%"/>
</p>


For details see [BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) by Hangbo Bao and Li Dong and Furu Wei

## Model Zoo

The results are evaluated on ImageNet2012 validation set

| Arch               | Weight                                                       | Top-1 Acc | Top-5 Acc | Crop ratio | # Params |
| ------------------ | ------------------------------------------------------------ | --------- | --------- | ---------- | -------- |
| beit_base_p16_224  | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_base_p16_224_ft.pdparams) | 85.21     | 97.66     | 0.9        | 87M      |
| beit_base_p16_384  | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_base_p16_384_ft.pdparams) | 86.81     | 98.14     | 1.0        | 87M      |
| beit_large_p16_224 | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_large_p16_224_ft.pdparams) | 87.48     | 98.30     | 0.9        | 304M     |
| beit_large_p16_384 | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_large_p16_384_ft.pdparams) | 88.40     | 98.60     | 1.0        | 304M     |
| beit_large_p16_512 | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_large_p16_512_ft.pdparams) | 88.60     | 98.66     | 1.0        | 304M     |

Note: ft 22k to 1k is pre-trained on imagenet22K and then fine-tuned to 1K

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


cfg_file = "configs/beit/beit_base_p16_224.yaml"
m = Model(cfg_file)
```

## Reference

```
@article{beit,
      title={{BEiT}: {BERT} Pre-Training of Image Transformers}, 
      author={Hangbo Bao and Li Dong and Furu Wei},
      year={2021},
      eprint={2106.08254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
