## SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments


PaddlePaddle reimplementation of [facebookresearch's repository for the SwAV model](https://github.com/facebookresearch/swav) that was released with the paper [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882).

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


## How to Self-supervised Pre-Training

With a batch size of 4096, SwAV is trained with 4 nodes:

```bash
# Note: Set the following environment variables
# and then need to run the script on each node.
unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=4
export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_stop_check_timeout=3600

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/swav_resnet50_224_pt_in1k_4n32c_dp_fp16o1.yaml
```

## How to Linear Classification

By default, we use momentum-SGD and a batch size of 256 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

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
    -c ./configs/swav_resnet50_224_lp_in1k_1n8c_dp_fp16o1.yaml
```

## How to End-to-End Fine-tuning
To perform end-to-end fine-tuning for SwAV:

* First download the data split text file with following commands:
    ```bash
    cd PASSL

    wget "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/10percent.txt"

    wget "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/1percent.txt"
    ```

* Then, download the pretrained models to `./pretrained/swav/swav_resnet50_in1k_800ep_pretrained.pdparams`

* Finally, run the training with the trained PASSL format checkpoint:
    ```bash
    unset PADDLE_TRAINER_ENDPOINTS
    export PADDLE_NNODES=1
    export PADDLE_MASTER="127.0.0.1:12538"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export FLAGS_stop_check_timeout=3600

    python -m paddle.distributed.launch \
        --nnodes=$PADDLE_NNODES \
        --master=$PADDLE_MASTER \
        --devices=$CUDA_VISIBLE_DEVICES \
        passl-train \
        -c ./configs/swav_resnet50_224_ft_in1k_1n4c_dp_fp16o1.yaml
        -o Global.pretrained_model=./pretrained/swav/swav_resnet50_in1k_800ep_pretrained
    ```

## Other Configurations
We provide more directly runnable configurations, see [SwAV Configurations](./configs/).

## Models

### ViT-Base
| Model         | Phase       | Dataset      | Configs                                                      | GPUs       | Epochs | Top1 Acc (%) | Links                                                   |
| ------------- | ----------- | ------------ | ------------------------------------------------------------ | ---------- | ------ | -------- | ------------------------------------------------------------ |
| resnet50 | pretrain    | ImageNet2012 | [config](./configs/swav_resnet50_224_pt_in1k_4n32c_dp_fp16o1.yaml) | A100*N4C32 | 800    | -        | [model]() \| [log]() |
| resnet50 | linear probe | ImageNet2012 | [config](./configs/swav_resnet50_224_lp_in1k_4n32c_dp_fp16o1.yaml) | A100*N1C8  | 75.3    | 0.7662   |        [model]() \| [log]() |
| resnet50 | finetune    | ImageNet2012 | [config](./configs/swav_resnet50_224_ft_in1k_1n4c_dp_fp16o1.yaml) | A100*N1C4  | 100    | 69.0   | [model]() \| [log]() |

## Citations

```bibtex
@misc{caron2021unsupervised,
      title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
      author={Mathilde Caron and Ishan Misra and Julien Mairal and Priya Goyal and Piotr Bojanowski and Armand Joulin},
      year={2021},
      eprint={2006.09882},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
