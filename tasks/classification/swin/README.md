# Swin Transformer

PaddlePaddle reimplementation of [microsoft's repository for the Swin-Transformer model](https://github.com/microsoft/Swin-Transformer) that was released with the paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf).

Swin Transformer (the name Swin stands for Shifted window) capably serves as a general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection.

![teaser](https://github.com/microsoft/Swin-Transformer/blob/main/figures/teaser.png?raw=true)

## Requirements
To enjoy some new features, a higher version of PaddlePaddle is required. For more installation tutorials
refer to [installation.md](../../../README.md#installation)

## How to Train

```bash
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/swin_base_patch4_window7_224_in1k_1n8c_dp_fp16o2.yaml
```

## How to Evaluation

```bash
# [Optional] Download checkpoint
mkdir -p pretrained/
wget -O ./pretrained/swin_base_patch4_window7_224_fp16o2.pdparams https://plsc.bj.bcebos.com/models/swin/v2.5/swin_base_patch4_window7_224_fp16o2.pdparams

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
  -c ./configs/swin_base_patch4_window7_224_in1k_1n8c_dp_fp16o2.yaml \
  -o Global.pretrained_model=pretrained/swin_base_patch4_window7_224_fp16o2 \
  -o Global.finetune=False
```

## Other Configurations
We provide more directly runnable configurations, see [Swin Configurations](./configs/).


## Models

| Model  | DType   | Pretrain     | Resolution | Configs                                                      | GPUs      | Img/sec | Top1 Acc | Official | Checkpoint                                                   | Log                                                          |
| ------ | ------- | ------------ | ---------- | ------------------------------------------------------------ | --------- | ------- | -------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Swin-B | FP16 O1 | ImageNet2012 | 224x224    | [config](./configs/swin_base_patch4_window7_224_in1k_1n8c_dp_fp16o1.yaml) | A100*N1C8 | 2155    | 0.83362  | 0.835    | [download](https://plsc.bj.bcebos.com/models/swin/v2.5/swin_base_patch4_window7_224_fp16o1.pdparams) | [log](https://plsc.bj.bcebos.com/models/swin/v2.5/swin_base_patch4_window7_224_fp16o1.log) |
| Swin-B | FP16 O2 | ImageNet2012 | 224x224    | [config](./configs/swin_base_patch4_window7_224_in1k_1n8c_dp_fp16o2.yaml) | A100*N1C8 | 3006    | 0.83223  | 0.835    | [download](https://plsc.bj.bcebos.com/models/swin/v2.5/swin_base_patch4_window7_224_fp16o2.pdparams) | [log](https://plsc.bj.bcebos.com/models/swin/v2.5/swin_base_patch4_window7_224_fp16o2.log) |



## Citations

```bibtex
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
