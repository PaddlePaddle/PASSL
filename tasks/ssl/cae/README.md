## [CAE](https://github.com/PaddlePaddle/VIMER/tree/main/CAE)

CAE is a novel masked image modeling (MIM) approach for self-supervised representation pretraining. The goal is to pretrain an encoder by solving the pretext task: estimate the masked patches from the visible patches in an image. We first feed the visible patches into the encoder, extracting the representations. Then, we make predictions from visible patches to masked patches in the encoded representation space. We introduce an alignment constraint, encouraging that the representations for masked patches, predicted from the encoded representations of visible patches, are aligned with the masked patch presentations computed from the encoder. In other words, the predicted representations are expected to lie in the encoded representation space, which empirically shows the benefit to representation learning. Last, the predicted masked patch representations are mapped to the targets of the pretext task through a decoder.
<br />
In comparison to previous MIM methods (e.g., BEiT) that couple the encoding and pretext task completion roles, our approach benefits the separation of the representation learning (encoding) role and the pretext task completion role, improving the representation learning capacity and accordingly helping more on downstream tasks. In addition, we present the explanations about why contrastive pretraining and supervised pretraining perform similarly and why MIM potentially performs better. We demonstrate the effectiveness of our CAE through superior transfer performance in downstream tasks: semantic segmentation, and object detection and instance segmentation.

<div align="center">
  <img src="https://github.com/PaddlePaddle/VIMER/blob/main/CAE/figs/CAE2.png?raw=true" width="480">
</div>

This is a PaddlePaddle/GPU re-implementation of the paper [Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/abs/2202.03026).

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

## How to train

### Pre-Training

A typical command to pre-train Vit-Base (recommended by default) with multi-nodes distributed training, run the following on **4 nodes** with 8 GPUs each::

```
# unset PADDLE_TRAINER_ENDPOINTS
# export PADDLE_NNODES=4
# export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PADDLE_JOB_ID=CAE

tmp_my_name=ep800_fp16o1
my_name=${tmp_my_name%.*}
OUTPUT_DIR='./output/'$my_name
echo $OUTPUT_DIR
DATA_PATH='./dataset/ILSVRC2012/'
TOKENIZER_PATH=dalle-weights
FLAGS_cudnn_exhaustive_search=True
export FLAGS_gemm_use_half_precision_compute_type=False

python -m paddle.distributed.launch  \
  --nnodes=$PADDLE_NNODES \
  --master=$PADDLE_MASTER \
  --devices=$CUDA_VISIBLE_DEVICES \
  main_pretrain.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --batch_size 64 --lr 1.5e-3 --warmup_epochs 10 --epochs 800 \
  --clip_grad 3.0 --layer_scale_init_value 0.1 \
  --imagenet_default_mean_and_std \
  --color_jitter 0 \
  --drop_path 0 \
  --sincos_pos_emb \
  --mask_generator block \
  --num_mask_patches 75 \
  --decoder_layer_scale_init_value 0.1 \
  --no_auto_resume \
  --save_ckpt_freq 50 \
  --exp_name $my_name \
  --regressor_depth 4 \
  --seed 0 \
  --log_dir vdl \
  --num_decoder_self_attention 4 \
  --dual_loss_weight 2 \
  --dual_path_ema 0
```
**Note**: The `dalle-weights` can be download from [encoder_weight](https://vimer.bj.bcebos.com/CAE/encoder_weight.pd) and [decoder_weight](https://vimer.bj.bcebos.com/CAE/decoder_weight.pd)

### Fine-tuning

A typical command to finetune of Vit-Base (recommended by default) with multi-nodes distributed training, run the following on **4 nodes** with 8 GPUs each:

```
# unset PADDLE_TRAINER_ENDPOINTS
# export PADDLE_NNODES=4
# export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PADDLE_JOB_ID=CAE

tmp_my_name=finetune_ep100_fp16o1
my_name=${tmp_my_name%.*}
OUTPUT_DIR='./output/'$my_name
echo $OUTPUT_DIR
DATA_PATH='./dataset/ILSVRC2012/'
MODEL_PATH='output/ep800_fp16o1/ep800_fp16o1_checkpoint-799.pd'
FLAGS_cudnn_exhaustive_search=True
export FLAGS_gemm_use_half_precision_compute_type=False

python -m paddle.distributed.launch  \
  --nnodes=$PADDLE_NNODES \
  --master=$PADDLE_MASTER \
  --devices=$CUDA_VISIBLE_DEVICES \
  main_finetune.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_base_patch16_224 \
  --finetune $MODEL_PATH \
  --nb_classes 1000 \
  --batch_size 128 \
  --lr 8e-3 \
  --accum_iter 1 \
  --warmup_epochs 5 \
  --epochs 100 \
  --layer_decay 0.65 \
  --drop_path 0.1 \
  --weight_decay 0.05 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --sin_pos_emb \
  --dist_eval \
  --no_auto_resume \
  --exp_name $my_name
```


### Linear Probing

A typical command to run Linear Probing of Vit-Base (recommended by default) with multi-nodes distributed training, run the following on **4 nodes** with 8 GPUs each:

```
# unset PADDLE_TRAINER_ENDPOINTS
# export PADDLE_NNODES=4
# export PADDLE_MASTER="xxx.xxx.xxx.xxx:12538"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PADDLE_JOB_ID=CAE

tmp_my_name=linprobe_ep90_fp16o1
my_name=${tmp_my_name%.*}
OUTPUT_DIR='./output/'$my_name
echo $OUTPUT_DIR
DATA_PATH='./dataset/ILSVRC2012/'
MODEL_PATH='output/ep800_fp16o1/ep800_fp16o1_checkpoint-799.pd'
FLAGS_cudnn_exhaustive_search=True
export FLAGS_gemm_use_half_precision_compute_type=False

python -m paddle.distributed.launch  \
  --nnodes=$PADDLE_NNODES \
  --master=$PADDLE_MASTER \
  --devices=$CUDA_VISIBLE_DEVICES \
  main_linprobe.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_base_patch16_224 \
  --finetune $MODEL_PATH \
  --nb_classes 1000 \
  --batch_size 512 \
  --epochs 90 \
  --blr 0.1 \
  --weight_decay 0.0 \
  --dist_eval \
  --log_dir $OUTPUT_DIR \
  --enable_linear_eval \
  --use_cls \
  --save_freq 50 \
  --disable_rel_pos_bias \
  --linear_type standard \
  --exp_name $my_name
```

### Models

| Model                         | Phase    | Dataset      | Epochs | GPUs       | Img/sec | Top1 acc@1(%) | Official | Checkpoint                                                   | Log                                                          |
| ----------------------------- | -------- | ------------ | ------ | ---------- | ------- | ------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| cae_base_patch16_224_8k_vocab | pretrain | ImageNet2012 | 800    | A100*N4C32 | 4936    | -             | -        | [download](https://plsc.bj.bcebos.com/models/cae/v2.5/cae_base_patch16_224_8k_vocab_pretrained_800ep.pd) | [download](https://plsc.bj.bcebos.com/models/cae/v2.5/cae_base_patch16_224_8k_vocab_pretrained_800ep.log) |
| cae_base_patch16_224          | finetune | ImageNet2012 | 100    | A100*N4C32 | 1729    | 83.62         | 83.61    | [download](https://plsc.bj.bcebos.com/models/cae/v2.5/cae_base_patch16_224_finetuned.pd) | [download](https://plsc.bj.bcebos.com/models/cae/v2.5/cae_base_patch16_224_finetuned.log) |
| cae_base_patch16_224          | linprobe | ImageNet2012 | 90     | A100*N4C32 | 19713   | 68.32         | 68.32    | [download](https://plsc.bj.bcebos.com/models/cae/v2.5/cae_base_patch16_224_linprobed.pd) | [download](https://plsc.bj.bcebos.com/models/cae/v2.5/cae_base_patch16_224_linprobed.log) |

## Citations

```
@article{chen2022context,
  title={Context autoencoder for self-supervised representation learning},
  author={Chen, Xiaokang and Ding, Mingyu and Wang, Xiaodi and Xin, Ying and Mo, Shentong and Wang, Yunhao and Han, Shumin and Luo, Ping and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2202.03026},
  year={2022}
}
```
