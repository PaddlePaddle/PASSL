## MoCo v3 for Self-supervised ResNet and ViT


PaddlePaddle reimplementation of [facebookresearch's repository for the MoCo v3 model](https://github.com/facebookresearch/moco-v3) that was released with the paper [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057).

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

With a batch size of 4096, ViT-Base is trained with 4 nodes:

```bash
# Note: Set the following environment variables
# and then need to run the script on each node.
unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=4
export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_stop_check_timeout=3600

IMAGENET_DIR=./dataset/ILSVRC2012/
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    main_moco.py \
    -a moco_vit_base \
    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
    --epochs=300 --warmup-epochs=40 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    ${IMAGENET_DIR}
```

## How to Linear Classification

By default, we use momentum-SGD and a batch size of 1024 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

```bash
unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_stop_check_timeout=3600

IMAGENET_DIR=./dataset/ILSVRC2012/
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    main_lincls.py \
    -a moco_vit_base \
    --lr=3 \
    --pretrained pretrained/checkpoint_0299.pd \
    ${IMAGENET_DIR}
```

## How to End-to-End Fine-tuning
To perform end-to-end fine-tuning for ViT, use our script to convert the pre-trained ViT checkpoint to PASSL DeiT format:

```bash
python extract_weight.py \
  --input pretrained/checkpoint_0299.pd \
  --output pretrained/moco_vit_base.pdparams
```

Then run the training with the converted PASSL format checkpoint:

```bash
unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_stop_check_timeout=3600

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/DeiT_base_patch16_224_in1k_1n8c_dp_fp16o1.yaml \
    -o Global.epochs=150 \
    -o Global.pretrained_model=pretrained/moco_vit_base \
    -o Global.finetune=True
```

## Models

### ViT-Base
| Model         | Phase       | Dataset      | Configs                                                      | GPUs       | Epochs | Top1 Acc | Checkpoint                                                   |
| ------------- | ----------- | ------------ | ------------------------------------------------------------ | ---------- | ------ | -------- | ------------------------------------------------------------ |
| moco_vit_base | pretrain    | ImageNet2012 | -                                                            | A100*N4C32 | 300    | -        | [download](https://plsc.bj.bcebos.com/models/mocov3/v2.4/moco_vit_base_in1k_300ep.pd) |
| moco_vit_base | linear prob | ImageNet2012 | -                                                            | A100*N1C8  | 90     | 0.7662   |                                                              |
| moco_vit_base | finetune    | ImageNet2012 | [config](./configs/DeiT_base_patch16_224_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 150    | 0.8288   |                                                              |

## Citations

```bibtex
@Article{chen2021mocov3,
  author  = {Xinlei Chen* and Saining Xie* and Kaiming He},
  title   = {An Empirical Study of Training Self-Supervised Vision Transformers},
  journal = {arXiv preprint arXiv:2104.02057},
  year    = {2021},
}
```
