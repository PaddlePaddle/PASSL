# DeiT

PaddlePaddle reimplementation of [facebookresearch's repository for the deit model](https://github.com/facebookresearch/deit) that was released with the paper [Training data-efficient image transformers &amp; distillation through attention](https://arxiv.org/abs/2012.12877). We only reimplementation training for ViT, excluding distillation.

## Requirements
To enjoy some new features, PaddlePaddle 2.4 is required. For more installation tutorials
refer to [installation.md](../../../tutorials/get_started/installation.md)

## How to Train

```bash
# Note: Set the following environment variables
# and then need to run the script on each node.
export PADDLE_NNODES=2
export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/DeiT_base_patch16_224_in1k_2n16c_dp_fp16o2.yaml
```

## How to Evaluation

```bash
# [Optional] Download checkpoint
mkdir -p pretrained/
wget -O ./pretrained/DeiT_base_patch16_224_in1k_2n16c_dp_fp16o2.pdparams https://plsc.bj.bcebos.com/models/deit/v2.4/DeiT_base_patch16_224_in1k_2n16c_dp_fp16o2.pdparams

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
  -c ./configs/DeiT_base_patch16_224_in1k_2n16c_dp_fp16o2.yaml \
  -o Global.pretrained_model=pretrained/DeiT_base_patch16_224_in1k_2n16c_dp_fp16o2 \
  -o Global.finetune=False
```

## Other Configurations
We provide more directly runnable configurations, see [DeiT Configurations](./configs/).


## Models

| Model        | DType   | Phase    | Dataset      | Configs                                                      | GPUs       | Img/sec | Top1 Acc | Pre-trained checkpoint                                       | Fine-tuned checkpoint | Log                                                          |
| ------------ | ------- | -------- | ------------ | ------------------------------------------------------------ | ---------- | ------- | -------- | ------------------------------------------------------------ | --------------------- | ------------------------------------------------------------ |
| ViT-B_16_224 | FP32    | pretrain | ImageNet2012 | [config](./configs/DeiT_base_patch16_224_in1k_1n8c_dp_fp32.yaml) | A100*N1C8  | 2780    | 0.81870  | [download](https://plsc.bj.bcebos.com/models/deit/v2.4/DeiT_base_patch16_224_in1k_1n8c_dp_fp32.pdparams) |                       | [log](https://plsc.bj.bcebos.com/models/deit/v2.4/DeiT_base_patch16_224_in1k_1n8c_dp_fp32.log) |
| ViT-B_16_224 | FP16_O2 | pretrain | ImageNet2012 | [config](./configs/DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2.yaml) | A100*N1C8  | 3169    | 0.82098  | [download](https://plsc.bj.bcebos.com/models/deit/v2.4/DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2.pdparams) |                       | [log](https://plsc.bj.bcebos.com/models/deit/v2.4/DeiT_base_patch16_224_in1k_1n8c_dp_fp16o2.log) |
| ViT-B_16_224 | FP16_O2 | pretrain | ImageNet2012 | [config](./configs/DeiT_base_patch16_224_in1k_2n16c_dp_fp16o2.yaml) | A100*N2C16 | 6514    | 0.81831  | [download](https://plsc.bj.bcebos.com/models/deit/v2.4/DeiT_base_patch16_224_in1k_2n16c_dp_fp16o2.pdparams) | -                     | [log](https://plsc.bj.bcebos.com/models/deit/v2.4/DeiT_base_patch16_224_in1k_2n16c_dp_fp16o2.log) |



## Citations

```bibtex
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```
