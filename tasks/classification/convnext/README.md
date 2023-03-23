# ConvNeXt

PaddlePaddle reimplementation of [facebookresearch's repository for the ConvneXt model](https://github.com/facebookresearch/ConvNeXt) that was released with the paper [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545).

## Requirements
To enjoy some new features, PaddlePaddle 2.4 is required. For more installation tutorials
refer to [installation.md](../../../tutorials/get_started/installation.md)

## How to Train

```bash
# Note: Set the following environment variables
# and then need to run the script on each node.
#export PADDLE_NNODES=4
#export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/ConvNeXt_base_224_in1k_4n32c_dp_fp32.yaml
```

## How to Evaluation

```bash
# [Optional] Download checkpoint
mkdir -p pretrained/
wget -O ./pretrained/ConvNeXt_base_224_in1k_dp_fp32.pdparams https://plsc.bj.bcebos.com/models/convnext/v2.5/ConvNeXt_base_224_in1k_dp_fp32.pdparams

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
  -c ./configs/ConvNeXt_base_224_in1k_1n8c_dp_fp32.yaml \
  -o Global.pretrained_model=pretrained/ConvNeXt_base_224_in1k_dp_fp32 \
  -o Global.finetune=False
```

## Other Configurations
We provide more directly runnable configurations, see [ConvNeXt Configurations](./configs/).


## Models

| Model         | DType | Phase    | Dataset      | Configs                                                       | GPUs       | Img/sec | Top1 Acc | Pre-trained checkpoint | Log         |
|---------------|-------|----------| ------------ |---------------------------------------------------------------|------------|--------|----------|------------------------|-------------|
| convnext_base | FP32  | pretrain | ImageNet2012 | [config](./configs/ConvNeXt_base_224_in1k_4n32c_dp_fp32.yaml) | A100*N4C32 | 7800   | 0.838    | [download](https://plsc.bj.bcebos.com/models/convnext/v2.5/ConvNeXt_base_224_in1k_dp_fp32.pdparams)       | [log](https://plsc.bj.bcebos.com/models/convnext/v2.5/ConvNeXt_base_224_in1k_dp_fp32.log) |



## Citations

```bibtex
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```
