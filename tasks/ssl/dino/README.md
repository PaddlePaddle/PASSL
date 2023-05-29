## Emerging Properties in Self-Supervised Vision Transformers (DINOv1)


PaddlePaddle reimplementation of [facebookresearch's repository for the DINO model](https://github.com/facebookresearch/dino) that was released with the paper [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294).

## Requirements
To enjoy some new features, PaddlePaddle develop is required. For more installation tutorials
refer to [installation.md](../../../tutorials/get_started/installation.md)

## Data Preparation

Prepare the data into the following directory:
```text
dataset/
└── ILSVRC2012
    ├── train
    └── val
```


## How to Linear Classification

By default, we use momentum-SGD and a batch size of 256 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

```bash
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_stop_check_timeout=3600

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/dino_deit_small_patch16_224_lp_in1k_1n8c_dp_fp16o1.yaml \
    -o Global.pretrained_model=./pretrained/dino/dino_deitsmall8_pretrain
```

## Other Configurations
We provide more directly runnable configurations, see [DINO Configurations](./configs/).

## Models

### DINO

| Model   | Phase       | Dataset      |  Configs   | GPUs | Epochs | BatchSize | Top1 Acc (%) | Checkpoint  | Train Log |
|---------|-------------| ------------ |------------|------|--------|-----------|----------|-------------|-----------|
| ViT-S/16 | pretrain    | ImageNet2012 | [config]() | A100*N4C32 |   500  |  1024  |   -  | [model]()   | [log]()   |
| ViT-S/16 | linear prob | ImageNet2012 | [config](./configs/dino_deit_small_patch16_224_lp_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 100 | 256  | 77.0 | [model](https://passl.bj.bcebos.com/models/dino/dino_deit_small_patch16_224_in1k_pt_100ep_bz256_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/dino/dino_deit_small_patch16_224_in1k_pt_100ep_bz256_linearprobe.log) |
| ViT-S/8 | linear prob | ImageNet2012 | [config](./configs/dino_deit_small_patch8_224_lp_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 100 | 256  | 79.6 | [model](https://passl.bj.bcebos.com/models/dino/dino_deit_small_patch8_224_in1k_pt_100ep_bz512_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/dino/dino_deit_small_patch8_224_in1k_pt_100ep_bz512_linearprobe.log) |
| ViT-B/16| linear prob | ImageNet2012 | [config](./configs/dino_vit_base_patch16_224_lp_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 100 | 768  | 78.1 | [model](https://passl.bj.bcebos.com/models/dino/dino_vit_base_patch16_224_in1k_pt_100ep_bz768_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/dino/dino_vit_base_patch16_224_in1k_pt_100ep_bz768_linearprobe.log) |
| ViT-B/8 | linear prob | ImageNet2012 | [config](./configs/dino_vit_base_patch8_224_lp_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 100 | 768  | 79.9 | [model](https://passl.bj.bcebos.com/models/dino/dino_vit_base_patch8_224_in1k_pt_100ep_bz768_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/dino/dino_vit_base_patch8_224_in1k_pt_100ep_bz768_linearprobe.log) |


## Citations

```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
