简体中文 | [English](README.md)

# All Tokens Matter: Token Labeling for Training Better Vision Transformers ([arxiv](https://arxiv.org/abs/2104.10858))

## 简介

在本文中，我们提出了Token Labeling：一种用于训练高性能视觉Transformer (ViT) 的新训练目标。与计算额外可训练类标记的分类损失的 ViT 的标准训练目标不同，我们提出的目标利用所有图像块标记以密集的方式计算训练损失。具体来说，标记标记将图像分类问题重新表述为多个标记级别的识别问题，并为每个patch标记分配由机器注释器生成的特定位置的单独监督。实验表明，Token Labeling可以在广泛的范围内清晰且一致地提高各种 ViT 模型的性能。以具有 26M 可学习参数的视觉Transformer为例，通过Token Labeling，该模型在 ImageNet 上可以达到 84.4% 的 Top-1 准确率。通过将模型大小稍微扩展到 150M，结果可以进一步增加到 86.4%，在以前的模型（250M+）中提供最小大小的模型达到 86%。我们还表明，标记标记可以明显提高预训练模型在具有密集预测的下游任务（例如语义分割）上的泛化能力。

![lvvit](https://raw.githubusercontent.com/zihangJiang/TokenLabeling/main/figures/Compare.png)


## 快速开始

### 准备label数据

我们提供了由[官方](https://github.com/zihangJiang/TokenLabeling#label-data)提供的`.pth`格式转换得到的`.npy`格式的label数据，

数据大小超过6GB，被分成了7部分。

```bash
# 下载全部7部分的label数据
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.00
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.01
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.02
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.03
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.04
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.05
wget https://passl.bj.bcebos.com/vision_transformers/TokenLabeling/label_top5_train_nfnet.tgz.06

# 将数据下载到同一个文件夹后, 用cat命令合并7个部分 
cat label_top5_train_nfnet.tgz.0* > label_top5_train_nfnet.tgz

# 然后解压数据
tar -zxvf label_top5_train_nfnet.tgz

# 最后将数据链接到data目录下
cd data
ln -s /path/to/label_top5_train_nfnet/ label_top5_train_nfnet
```

### 多卡训练

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/lvvit/lvvit_tiny.yaml
```

### 评估

```bash
python tools/train.py -c configs/lvvit/lvvit_tiny.yaml --load ${LVViT_WEGHT_FILE} --evaluate-only
```


## 参考

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
