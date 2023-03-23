# Vision Transformer

PaddlePaddle reimplementation of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy\*†, Lucas Beyer\*, Alexander Kolesnikov\*, Dirk
Weissenborn\*, Xiaohua Zhai\*, Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby\*†.

(\*) equal technical contribution, (†) equal advising.

![Figure 1 from paper](https://github.com/google-research/vision_transformer/raw/main/vit_figure.png)

Overview of the model: we split an image into fixed-size patches, linearly embed
each of them, add position embeddings, and feed the resulting sequence of
vectors to a standard Transformer encoder. In order to perform classification,
we use the standard approach of adding an extra learnable "classification token"
to the sequence.

## Requirements
To enjoy some new features, PaddlePaddle 2.4 is required. For more installation tutorials
refer to [installation.md](../../../tutorials/get_started/installation.md)


## How to Train

```bash
# Note: If running on multiple nodes,
# set the following environment variables
# and then need to run the script on each node.
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/ViT_base_patch16_224_in1k_1n8c_dp_fp16o2.yaml
```

## How to Finetune

```bash
# [Optional] Download checkpoint
mkdir -p pretrained/vit/ViT_base_patch16_224/
wget -O ./pretrained/vit/ViT_base_patch16_224/imagenet2012-ViT-B_16-224.pdparams https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-224.pdparams

```


```bash
# Note: If running on multiple nodes,
# set the following environment variables
# and then need to run the script on each node.
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    passl-train \
    -c ./configs/ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2.yaml \
    -o Global.pretrained_model=./pretrained/vit/ViT_base_patch16_224/imagenet2012-ViT-B_16-224
```

## Other Configurations
We provide more directly runnable configurations, see [ViT Configurations](./configs/).


## Models

| Model        | Phase    | Dataset      | Configs                                                      | GPUs       | Img/sec | Top1 Acc | Official | Pre-trained checkpoint                                       | Fine-tuned checkpoint                                        | Log                                                          |
| ------------ | -------- | ------------ | ------------------------------------------------------------ | ---------- | ------- | -------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ViT-B_16_224 | pretrain | ImageNet2012 | [config](./configs/ViT_base_patch16_224_in1k_1n8c_dp_fp16o2.yaml) | A100*N1C8  | 3583    | 0.75196  | 0.7479   | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-224.pdparams) | -                                                            | [log](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-224.log) |
| ViT-B_16_384 | finetune | ImageNet2012 | [config](./configs/ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2.yaml) | A100*N1C8  | 719     | 0.77972  | 0.7791   | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-224.pdparams) | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-384.pdparams) | [log](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-384.log) |
| ViT-L_16_224 | pretrain | ImageNet21K  | [config](./configs/ViT_large_patch16_224_in21k_4n32c_dp_fp16o2.yaml) | A100*N4C32 | 5256    | -        | -        | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet21k-ViT-L_16-224.pdparams) | -                                                            | [log](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet21k-ViT-L_16-224.log) |
| ViT-L_16_384 | finetune | ImageNet2012 | [config](./configs/ViT_large_patch16_384_in1k_ft_4n32c_dp_fp16o2.yaml) | A100*N4C32 | 934     | 0.85030  | 0.8505   | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet21k-ViT-L_16-224.pdparams) | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet21k%2Bimagenet2012-ViT-L_16-384.pdparams) | [log](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet21k%2Bimagenet2012-ViT-L_16-384.log) |

## ImageNet21K data preparation

ImageNet21K is not "cleaned" but basically the (exact) same image may appear under multiple folders (labels)
(see [About ImageNet-21K train and eval label](https://github.com/google-research/vision_transformer/issues/237#issuecomment-1259631151)).
ViT official paper and repository also do not give "cleaned" `<image, label>` training label files.

According to various information and conjectures (thanks @lucasb-eyer), we got the accuracy given by ViT official repository.
If you want to pre-train ViT-Large on ImageNet 21K from scratch, you can process the data according to the following steps:

**Since ImageNet21K does not have an officially divided verification set, we use all the images as the training set.
We construct the dummy verification set not for parameter adjustment and evaluation, but for the convenience of
observing whether the training is ok.**

(1) Calculate the md5 value of each image

```
# 21841 classes
ImageNet21K/
└── images
    ├── n00004475/
    ├── n02087122/
    ├── ...
    └── n12492682/
```

```bash
find /data/ImageNet21K/images/ -type f -print0 | xargs --null md5sum > md5sum.txt
```

(2) Reassign multi-label based on md5 value
```python
from collections import defaultdict

lines = []
with open('md5sum.txt', 'r') as f:
    for line in f:
        # 35c1efae521b76e423cdd07a00d963c9  /data/ImageNet21K/images/n00004475/n00004475_54295.JPEG
        line = line.replace('/data/ImageNet21K/', '')
        lines.append(line)

ret = defaultdict(list)
classes = set()
for line in lines:
    line = line.strip()
    md5, path = line.split()
    ret[md5].append(path)
    classes.add(path.split('/')[-2])

classes = sorted(entry for entry in classes)
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

out = []
for key in ret:
    paths = ret[key]
    path = paths[0]
    labels = []
    for p in paths:
        class_to_idx[p.split('/')[-2]]
        labels.append(class_to_idx[p.split('/')[-2]])
    labels = [l for l in set(labels)]
    labels.sort()
    out.append((path, labels))

out.sort(key=lambda x: x[1][0])

fp = open('image_all_list.txt', 'w')
for path, labels in out:
    labels = [str(l) for l in labels]
    label = ','.join(labels)
    fp.write(f'{path} {label}\n')
```

(3) [Optinal] Choose a **dummy** validation set
```python
import os
from collections import defaultdict

val_list = []

id_to_images = defaultdict(list)

with open('image_all_list.txt', 'r') as f:
    for line in f:
        path, label = line.strip().split()
        label = label.split(',')
        if len(label) == 1:
            id_to_images[label[0]].append(path)

with open('image_dummy_val_list.txt', 'w') as f:
    for idx in id_to_images:
        for path in id_to_images[idx][:20]:
            f.write(f'{path} {idx}\n')
```


## Citations

```bibtex
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
