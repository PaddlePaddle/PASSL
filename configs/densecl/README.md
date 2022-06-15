[简体中文](README_ch.md) | English

# Dense Contrastive Learning for Self-Supervised Visual Pre-Training ([arxiv](https://arxiv.org/abs/2011.09157))

## Introduction

To date, most existing self-supervised learning methods are designed and optimized for image classification. These pre-trained models can be sub-optimal for dense prediction tasks due to the discrepancy between image-level prediction and pixel-level prediction. To fill this gap, we aim to design an effective, dense self-supervised learning method that directly works at the level of pixels (or local features) by taking into account the correspondence between local features. We present dense contrastive learning, which implements self-supervised learning by optimizing a pairwise contrastive (dis)similarity loss at the pixel level between two views of input images. Compared to the baseline method MoCo-v2, our method introduces negligible computation overhead (only <1% slower), but demonstrates consistently superior performance when transferring to downstream dense prediction tasks including object detection, semantic segmentation and instance segmentation; and outperforms the state-of-the-art methods by a large margin. Specifically, over the strong MoCo-v2 baseline, our method achieves significant improvements of 2.0% AP on PASCAL VOC object detection, 1.1% AP on COCO object detection, 0.9% AP on COCO instance segmentation, 3.0% mIoU on PASCAL VOC semantic segmentation and 1.8% mIoU on Cityscapes semantic segmentation.

<p align="center">
  <img src="../../docs/imgs/densecl.png" width="100%" height="100%"/>
</p>


## Getting Started

### 1. Train DenseCL

#### single gpu
```
python tools/train.py -c configs/densecl/densecl_r50.yaml
```

#### multiple gpus

```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/densecl/densecl_r50.yaml
```

Pretraining models with 200 epochs can be found at [densecl](https://drive.google.com/file/d/1RWPO_g-fNJv8FsmCZ3LUbPTgPwtx-ybZ/view?usp=sharing)

Note: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following ```new_lr = old_lr * new_ngpus / old_ngpus```.

### 2. Extract backbone weights

```
python tools/extract_weight.py ${CHECKPOINT} --output ${WEIGHT_FILE} --remove_prefix
```

### 3. Evaluation on ImageNet Linear Classification

#### Train:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/moco/moco_clas_r50.yaml --pretrained ${WEIGHT_FILE}
```

#### Evaluate:
```
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c configs/moco/moco_clas_r50.yaml --load ${CLS_WEGHT_FILE} --evaluate-only
```

The trained linear weights in conjuction with the backbone weights can be found at [densecl linear](https://drive.google.com/file/d/1XJeDY8clKfhUeXw4JcCa1QgG2G-Ibr4m/view?usp=sharing)

## Reference

```
@inproceedings{wang2021dense,
  title={Dense contrastive learning for self-supervised visual pre-training},
  author={Wang, Xinlong and Zhang, Rufeng and Shen, Chunhua and Kong, Tao and Li, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3024--3033},
  year={2021}
}
```
