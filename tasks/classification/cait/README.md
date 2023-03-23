# CaiT

PaddlePaddle reimplementation of [facebookresearch's repository for the cait model](https://github.com/facebookresearch/deit) that was released with the paper [CaiT: Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).

## Requirements
To enjoy some new features, PaddlePaddle 2.4 is required. For more installation tutorials
refer to [installation.md](../../../tutorials/get_started/installation.md)

## How to Train

```bash
# Note: Set the following environment variables
# and then need to run the script on each node.
export PADDLE_NNODES=1
export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/cait_s24_224_in1k_1n8c_dp_fp16o2.yaml
```

## How to Evaluation

```bash
# [Optional] Download checkpoint
mkdir -p pretrained/
wget -O ./pretrained/cait_s24_224_in1k_1n8c_dp_fp16o2.pdparams https://plsc.bj.bcebos.com/models/cait/v2.4/cait_s24_224_in1k_1n8c_dp_fp16o2.pdparams

```

```bash
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
  --nnodes=$PADDLE_NNODES \
  --master=$PADDLE_MASTER \
  --devices=$CUDA_VISIBLE_DEVICES \
  passl-eval \
  -c ./configs/cait_s24_224_in1k_1n8c_dp_fp16o2.yaml \
  -o Global.pretrained_model=pretrained/cait_s24_224_in1k_1n8c_dp_fp16o2 \
  -o Global.finetune=False
```

## Other Configurations
We provide more directly runnable configurations, see [CaiT Configurations](./configs/).


## Models

| Model        | Phase    | Dataset      | Configs                                                      | GPUs      | Img/sec | Top1 Acc | Pre-trained checkpoint                                       | Fine-tuned checkpoint | Log                                                          |
| ------------ | -------- | ------------ | ------------------------------------------------------------ | --------- | ------- | -------- | ------------------------------------------------------------ | --------------------- | ------------------------------------------------------------ |
| cait_s24_224 | pretrain | ImageNet2012 | [config](./configs/cait_s24_224_in1k_1n8c_dp_fp16o2.yaml) | A100*N1C8 | 2473    | 0.82628  | [download](https://plsc.bj.bcebos.com/models/cait/v2.4/cait_s24_224_in1k_1n8c_dp_fp16o2.pdparams) |                       | [log](https://plsc.bj.bcebos.com/models/cait/v2.4/cait_s24_224_in1k_1n8c_dp_fp16o2.log) |



## Citations

```bibtex
@InProceedings{Touvron_2021_ICCV,
    author    = {Touvron, Hugo and Cord, Matthieu and Sablayrolles, Alexandre and Synnaeve, Gabriel and J\'egou, Herv\'e},
    title     = {Going Deeper With Image Transformers},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {32-42}
}
```
