# MoCo
![MoCo](https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png)

This is a PaddlePaddle implementation of the 
[MoCov2](https://arxiv.org/abs/2003.04297).


## Install Preparation

MoCo requires `PaddlePaddle >= 2.4`.
```shell
# git clone https://github.com/PaddlePaddle/PASSL.git
# cd /path/to/PASSL
```

All commands are executed in the `PASSL` root directory.

```shell
# python setup.py install
```

## Data Preparation

The imagenet 1k dataset needs to be prepared first and will be organized into the following directory structure.

```shell
ILSVRC2012
├── train/
├── xxx
├── val/
└── xxx
```

Then configure the path.

```shell
mkdir -p dataset
ln -s /path/to/ILSVRC2012 dataset/ILSVRC2012
```

## Unsupervised Training

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, you can run the script: 

### MoCo V2 (Single Node with 8 GPUs)
```shell
# sh pretrain.sh
```

The differences between MoCo v2 and MoCo v1 are as follows:
* MoCo v2 has a projector
* Data augmentation
* Softmax temperature
* Learning rate scheduler

## Linear Classification

When the unsupervised pre-training is complete, or directly download the provided pre-training checkpoint, you can use the following script to train a supervised linear classifier.

#### Linear Classification Training (Single Node with 8 GPUs)

```shell
# sh linearprobe.sh
```

### MoCo v2

#### [Optional] Download checkpoint & Modify yaml  configure
```shell
mkdir -p pretrained/moco/
wget -O ./pretrained/moco/mocov2_pt_imagenet2012_resnet50.pdparams https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_pt_imagenet2012_resnet50.pdparams
```

#### Linear Classification Training (Single Node with 8 GPUs)

```shell
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/mocov2_resnet50_lp_in1k_1n8c.yaml
```

## Models

| Model   | Phase                 | Epochs | Top1 Acc | Checkpoint                                                   | Log                                                          |
| ------- | --------------------- | ------ | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MoCo v2 | Unsupervised Training | 200    | -        | [download](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_pt_imagenet2012_resnet50.pdparams) | [log](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_pt_imagenet2012_resnet50.log) |
| MoCo v2 | Linear Classification | 100    | 0.676595 | [download](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_lincls_imagenet2012_resnet50.pdparams) | [log](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_lincls_imagenet2012_resnet50.log) |


## Citations

```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```
