FLAGS_cudnn_exhaustive_search=0
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --devices=$CUDA_VISIBLE_DEVICES ../../../../tools/train.py \
       -c ../../../../configs/moco/moco_clas_r50.yaml \
       -o epochs=50 \
       --pretrain ./pretrained/ssl/moco_v2_r50_backbone.pd