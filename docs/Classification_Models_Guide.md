# How to use ViTs in PASSL 

PASSL provides developers with a number of implementations of Transformer classification models for the vision domain, each of which can be invoked through PASSL's configuration files so that users can quickly implement research experiments, and provides model pre-training weights that can be used to fine-tune their own datasets

## Included Model

* **Vision Transformer**:  [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by y Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
* **Swin Transformer**: [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) by by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
* **CaiT**: [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239) by Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Herv\'e J\'egou.
* **T2T ViT**: [Training Vision Transformers from Scratch on ImageNet ](https://arxiv.org/abs/2101.11986) by Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Jiang, Zi-Hang and Tay, Francis E.H. and Feng, Jiashi and Yan, Shuicheng.

* **CvT**: [Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) by by Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei.
* **BEiT**: [BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) by Hangbo Bao and Li Dong and Furu Wei.
* **MLP Mixer**: [An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) by Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and others.
* **XCiT**: [Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681) by Alaaeldin El-Nouby, Hugo Touvron, Mathilde Caron, Piotr Bojanowski, Matthijs Douze, Armand Joulin, Ivan Laptev, Natalia Neverova, Gabriel Synnaeve, Jakob Verbeek and Hervé Jegou.

## Weights Download

| Arch               | Weight                                                       | Top-1 Acc | Top-5 Acc | Crop ratio | # Params |
| ------------------ | ------------------------------------------------------------ | --------- | --------- | ---------- | -------- |
| cait_s24_224       | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s24_224.pdparams) | 83.45     | 96.57     | 1.0        | 46.8M    |
| cait_xs24_384      | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_x24_384.pdparams) | 84.06     | 96.89     | 1.0        | 26.5M    |
| cait_s24_384       | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s24_384.pdparams) | 85.05     | 97.34     | 1.0        | 46.8M    |
| cait_s36_384       | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_s36_384.pdparams) | 85.45     | 97.48     | 1.0        | 68.1M    |
| cait_m36_384       | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_m36_384.pdparams) | 86.06     | 97.73     | 1.0        | 270.7M   |
| cait_m48_448       | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cait/cait_m48_448.pdparams) | 86.49     | 97.75     | 1.0        | 355.8M   |
| t2t_vit_14         | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_14.pdparams) | 81.50     | 95.67     | 0.9        | 21.5M    |
| t2t_vit_19         | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_19.pdparams) | 81.93     | 95.74     | 0.9        | 39.1M    |
| t2t_vit_24         | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_24.pdparams) | 82.28     | 95.89     | 0.9        | 64.0M    |
| t2t_vit_t_14       | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_t_14.pdparams) | 81.69     | 95.85     | 0.9        | 21.5M    |
| t2t_vit_t_19       | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_t_19.pdparams) | 82.44     | 96.08     | 0.9        | 39.1M    |
| t2t_vit_t_24       | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/t2t/t2t_vit_t_24.pdparams) | 82.55     | 96.07     | 0.9        | 64.0M    |
| cvt_13_224         | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cvt/cvt_13_224_pt.pdparams) | 81.59     | 95.67     | 0.875      | 20.0M    |
| cvt_13_384         | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/cvt/cvt_13_384_ft.pdparams) | 82.90     | 96.92     | 1.0        | 20.0M    |
| cvt_21_224         | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/cvt/cvt_21_224_pt.pdparams) | 82.46     | 96.00     | 0.875      | 31.6M    |
| cvt_21_384         | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/cvt/cvt_21_384_ft.pdparams) | 84.63     | 97.54     | 1.0        | 31.6M    |
| cvt_w24_384        | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/cvt/cvt_w24_384_ft.pdparams) | 87.39     | 98.37     | 1.0        | 277.3M   |
| beit_base_p16_224  | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_base_p16_224_ft.pdparams) | 85.21     | 97.66     | 0.9        | 87M      |
| beit_base_p16_384  | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_base_p16_384_ft.pdparams) | 86.81     | 98.14     | 1.0        | 87M      |
| beit_large_p16_224 | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_large_p16_224_ft.pdparams) | 87.48     | 98.30     | 0.9        | 304M     |
| beit_large_p16_384 | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_large_p16_384_ft.pdparams) | 88.40     | 98.60     | 1.0        | 304M     |
| beit_large_p16_512 | [ft 22k to 1k](https://passl.bj.bcebos.com/vision_transformers/beit/beit_large_p16_512_ft.pdparams) | 88.60     | 98.66     | 1.0        | 304M     |
| mlp_mixer_b16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/mlp_mixer/mlp-mixer_b16_224.pdparams) | 76.60     | 92.23     | 0.875      | 60.0M    |
| mlp_mixer_l16_224  | [pretrain 1k](https://passl.bj.bcebos.com/vision_transformers/mlp_mixer/mlp-mixer_l16_224.pdparams) | 72.06     | 87.67     | 0.875      | 208.2M   |
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

The above metrics were tested on the ImageNet 2012 dataset.

>  Note：**pretrain 1k**  means that the model is trained directly on ImageNet1k, **ft 22k in 1k** means that the model is trained on ImageNet22k and then fine-tuned on ImageNet1K

## Usage 

Please install the necessary packages first to ensure the code can run, see [INSTALL.md](https://github.com/PaddlePaddle/PASSL/blob/main/docs/INSTALL.md)

You can run the following code in the `./PASSL` directory and you can change the `cfg_file` to select the model you want

You can download the appropriate weights for the model to load the pre-training weights

```python
import paddle
from passl.modeling.backbones import build_backbone
from passl.modeling.heads import build_head
from passl.utils.config import get_config


class CreatModel(paddle.nn.Layer):
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
model = CreatModel(cfg_file)

model_state_dict = paddle.load('cvt_13_224.pdparams')
model.set_dict(model_state_dict)
```

If you need to fine tune with the model，ou can modify the config file, such as changing the number of categories

```yaml
# configs/cvt/cvt_13_224.yaml
...

model:
  name: CvTWrapper
  architecture:
      name: CvT
      embed_dim: [64, 192, 384]
      depth: [1, 2, 10]
      num_heads: [1, 3, 6]
  head:
    name: CvTClsHead
    num_classes: 10   # Modify the number of categories to match your taxonomy data set
    in_channels: 384
    
...
```

## Coming Soon

model train 

model validate

## Contact

If you have any questions, please create an [issue](https://github.com/PaddlePaddle/PASSL/issues) on our Github.
