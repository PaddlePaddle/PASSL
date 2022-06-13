简体中文 | [English](README.md)

# Unsupervised Learning of Visual Features by Contrasting Cluster Assignments ([arxiv](https://arxiv.org/abs/2006.09882))

## 简介

无监督图像表示显着缩小了与有监督预训练的差距，特别是最近对比学习方法的成就。这些对比方法通常在线工作，并且依赖于大量显式的成对特征比较，这在计算上具有挑战性。在本文中，我们提出了一种在线算法 SwAV，它利用对比方法而不需要计算成对比较。具体来说，我们的方法同时对数据进行聚类，同时强制为同一图像的不同增强（或视图）生成的聚类分配之间的一致性，而不是像对比学习中那样直接比较特征。简单地说，我们使用交换预测机制，从另一个视图的表示中预测一个视图的集群分配。我们的方法可以用大批量和小批量进行训练，并且可以扩展到无限量的数据。与以前的对比方法相比，我们的方法内存效率更高，因为它不需要大型内存库或特殊的动量网络。此外，我们还提出了一种新的数据增强策略 multi-crop，它使用具有不同分辨率的视图混合来代替两个全分辨率视图，而不会增加内存或计算需求。我们通过使用 ResNet-50 在 ImageNet 上实现 75.3% 的 top-1 准确率来验证我们的发现，并且在所有考虑的下游任务上都超过了监督预训练。

<p align="center">
  <img src="../../docs/imgs/swav.png" width="100%" height="100%"/>
</p>




## 快速开始

### 1. 训练SwAV

单卡训练

```bash
python tools/train.py -c configs/swav/swav_r50_100ep.yaml
```

多卡训练

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/swav/swav_r50_100ep.yaml
```

100 个 epoch 的预训练模型权重：[swav](https://drive.google.com/file/d/1budFSoQqZz1Idyej-R4E6kUnL8CGtdyu/view?usp=sharing)

### 2. 提取 backbone 权重

```bash
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE} --remove_prefix
```

### 3. ImageNet 线性分类评估

训练

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/swav/swav_clas_r50.yaml --pretrained ${WEIGHT_FILE}
```

评估

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/swav/swav_clas_r50.yaml --load ${CLS_WEGHT_FILE} --evaluate-only
```

主干网络以及线性权重：[swav linear](https://drive.google.com/file/d/1uduDAqJqK1uFclhQSK0d9RjzGNYR_Tj2/view?usp=sharing)

## 参考

```
@inproceedings{caron2020unsupervised,
  title={Unsupervised learning of visual features by contrasting cluster assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9912--9924},
  year={2020}
}
```
