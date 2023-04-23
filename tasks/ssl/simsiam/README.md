## SimSiam


PaddlePaddle reimplementation of [facebookresearch's repository for the SimSiam model](https://github.com/facebookresearch/simsiam) that was released with the paper [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566).

## Requirements
To enjoy some new features, PaddlePaddle 2.4 is required. For more installation tutorials
refer to [installation.md](../../../tutorials/get_started/installation.md)

## Data Preparation

Prepare the data into the following directory:
```text
dataset/
└── ILSVRC2012
    ├── train
    └── val
```


## How to Self-supervised Pre-Training

The pretrain of SimSiam is trained with a single 8-GPU node:

```bash
unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.0:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/simsiam_resnet50_pt_in1k_1n8c_dp_fp32.yaml
```

## How to Linear Classification

- Download pretrained model
```bash
mkdir -p pretrained/simsiam
wget -O ./pretrained/simsiam/simsiam_resnet50_in1k_100ep_bz512_pretrained.pdparams https://passl.bj.bcebos.com/models/simsiam/simsiam_resnet50_in1k_100ep_bz512_pretrained.pdparams
```

- Train linear classification model
By default, we use LARS optimizer and a batch size of 4096 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

```bash
unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.0:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/simsiam_resnet50_lp_in1k_1n8c_dp_fp32.yaml \
    -o Global.pretrained_model=./pretrained/simsiam/simsiam_resnet50_in1k_100ep_bz512_pretrained

```

## Other Configurations
We provide more directly runnable configurations, see [SimSiam Configurations](./configs/).

## Models

| Model   | Phase       | Dataset      | Configs                                                        | GPUs      | Epochs | Batch | Top1 Acc | official Top1 Acc |Checkpoint                                                                                      | Train Log                                                                                              |
|---------|-------------| ------------ |----------------------------------------------------------------|-----------|--------|-------|----------|-------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| simsiam | pre-train   | ImageNet2012 | [config](./configs/simsiam_resnet50_pt_in1k_1n8c_dp_fp32.yaml) | A100*N1C8 | 100    | 512   | -        | -                 | [download](https://passl.bj.bcebos.com/models/simsiam/simsiam_resnet50_in1k_100ep_bz512_pretrained.pdparams)     | [log](https://passl.bj.bcebos.com/models/simsiam/simsiam_resnet50_in1k_100ep_bz512_pretrained.log)     |
| simsiam | linear-prob | ImageNet2012 | [config](./configs/simsiam_resnet50_lp_in1k_1n8c_dp_fp32.yaml) | A100*N1C8 | 90     | 4096  | 68.2     | 68.1              | [download](https://passl.bj.bcebos.com/models/simsiam/simsiam_resnet50_in1k_pt_100ep_bz512_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/simsiam/simsiam_resnet50_in1k_pt_100ep_bz512_linearprobe.log) |                                                                                            |

## Citations

```bibtex
@Article{chen2020simsiam,
  author  = {Xinlei Chen and Kaiming He},
  title   = {Exploring Simple Siamese Representation Learning},
  journal = {arXiv preprint arXiv:2011.10566},
  year    = {2020},
}
```
