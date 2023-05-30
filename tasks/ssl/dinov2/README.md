## DINOv2: Learning Robust Visual Features without Supervision


PaddlePaddle reimplementation of [facebookresearch's repository for the DINOv2 model](https://github.com/facebookresearch/dinov2) that was released with the paper [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).

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

By default, we use momentum-SGD and a batch size of 128 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

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
    -c ./configs/dinov2_vit_small_patch14_224_lp_in1k_1n8c_dp_fp16o1.yaml \
    -o Global.pretrained_model=./pretrained/dinov2/dinov2_vits14_pretrain
```

## Other Configurations
We provide more directly runnable configurations, see [DINOv2 Configurations](./configs/).

## Models

### DINOv2

| Model   | Phase       | Dataset      |  Configs   | GPUs | Epochs | BatchSize | Top1 Acc (%) | Checkpoint  | Train Log |
|---------|-------------| ------------ |------------|------|--------|-----------|----------|-------------|-----------|
| ViT-S/14 | pretrain    | ImageNet2012 | [config]() | A100*N4C32 |   500  |  1024  |   -  | [model]()   | [log]()   |
| ViT-S/14 | linear prob | ImageNet2012 | [config](./configs/dinov2_vit_small_patch14_224_lp_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 10 | 128  | 80.9 | [model](https://passl.bj.bcebos.com/models/dinov2/dinov2_vit_small_patch14_224_in1k_pt_10ep_bz128_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/dinov2/dinov2_vit_small_patch14_224_in1k_pt_10ep_bz128_linearprobe.log) |                                                                                            |
| ViT-B/14 | linear prob | ImageNet2012 | [config](./configs/dinov2_vit_base_patch14_224_lp_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 10 | 128  | 84.1 | [model](https://passl.bj.bcebos.com/models/dinov2/dinov2_vit_base_patch14_224_in1k_pt_10ep_bz128_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/dinov2/dinov2_vit_base_patch14_224_in1k_pt_10ep_bz128_linearprobe.log) |
| ViT-L/14 | linear prob | ImageNet2012 | [config](./configs/dinov2_vit_large_patch14_224_lp_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 10 | 128  | 85.9 | [model](https://passl.bj.bcebos.com/models/dinov2/dinov2_vit_large_patch14_224_in1k_pt_10ep_bz128_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/dinov2/dinov2_vit_large_patch14_224_in1k_pt_10ep_bz128_linearprobe.log) |
| ViT-G2/14 | linear prob | ImageNet2012 | [config](./configs/dinov2_vit_gaint2_patch14_224_lp_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8  | 10 | 128 | 86.4 | [model](https://passl.bj.bcebos.com/models/dinov2/dinov2_vit_gaint2_patch14_224_in1k_pt_10ep_bz128_linearprobe.pdparams) | [log](https://passl.bj.bcebos.com/models/dinov2/dinov2_vit_gaint2_patch14_224_in1k_pt_10ep_bz128_linearprobe.log) |


## Citations

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
