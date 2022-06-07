[简体中文](README_ch.md) | English

# All Tokens Matter: Token Labeling for Training Better Vision Transformers ([arxiv](https://arxiv.org/abs/2104.10858))

## Introduction

In this paper, we present token labeling—a new training objective for training
high-performance vision transformers (ViTs). Different from the standard training
objective of ViTs that computes the classification loss on an additional trainable
class token, our proposed one takes advantage of all the image patch tokens to compute the training loss in a dense manner. Specifically, token labeling reformulates
the image classification problem into multiple token-level recognition problems and
assigns each patch token with an individual location-specific supervision generated
by a machine annotator. Experiments show that token labeling can clearly and consistently improve the performance of various ViT models across a wide spectrum.
For a vision transformer with 26M learnable parameters serving as an example,
with token labeling, the model can achieve 84.4% Top-1 accuracy on ImageNet.
The result can be further increased to 86.4% by slightly scaling the model size up
to 150M, delivering the minimal-sized model among previous models (250M+)
reaching 86%. We also show that token labeling can clearly improve the generalization capability of the pretrained models on downstream tasks with dense prediction,
such as semantic segmentation.

![lvvit](https://raw.githubusercontent.com/zihangJiang/TokenLabeling/main/figures/Compare.png)


## Getting Started

### Prepare label data

We provide the label data in `.npy` format converted from the [official](https://github.com/zihangJiang/TokenLabeling#label-data) `.pth` format. 

The data is 6GB+ size and is split into 7 parts. 

```bash
# Download all parts of label data.
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.00
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.01
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.02
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.03
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.04
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.05
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.06

# After downloading them into a folder, use the `cat` to merge them. 
cat label_top5_train_nfnet.tgz.0* > label_top5_train_nfnet.tgz

# Then unzip it.
tar -zxvf label_top5_train_nfnet.tgz

# Finally Link to data folder.
cd data
ln -s /path/to/label_top5_train_nfnet/ label_top5_train_nfnet
```

### Train with multiple gpus

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/lvvit/lvvit_tiny.yaml
```

### Evaluate

```bash
python tools/train.py -c configs/lvvit/lvvit_tiny.yaml --load ${LVViT_WEGHT_FILE} --evaluate-only
```


## Reference

```
@inproceedings{NEURIPS2021_9a49a25d,
 author = {Jiang, Zi-Hang and Hou, Qibin and Yuan, Li and Zhou, Daquan and Shi, Yujun and Jin, Xiaojie and Wang, Anran and Feng, Jiashi},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {18590--18602},
 publisher = {Curran Associates, Inc.},
 title = {All Tokens Matter: Token Labeling for Training Better Vision Transformers},
 url = {https://proceedings.neurips.cc/paper/2021/file/9a49a25d845a483fae4be7e341368e36-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
