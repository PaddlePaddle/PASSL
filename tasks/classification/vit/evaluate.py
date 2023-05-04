# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image
import paddle

import paddle.distributed as dist
from passl.models import vision_transformer, vision_transformer_hybrid
from passl.data import preprocess as transforms
from passl.data import dataset as datasets
from passl.utils.misc import AverageMeter

from passl.distributed import distributed_env as dist_env

@paddle.no_grad()
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = (
        pred == target.reshape([1, -1]).expand_as(pred)).astype(paddle.float32)
    return [
        correct[:min(k, maxk)].reshape([-1]).sum(0) * 100. / batch_size
        for k in topk
    ]


mp_degree = 2
pp_degree = 1
sharding_degree = 1
if mp_degree > 1:
    dist_env.init_dist_env(seed=42, mp_degree=mp_degree, pp_degree=pp_degree, sharding_degree=sharding_degree)

    from paddle.distributed.fleet.meta_parallel import TensorParallel
    model = vision_transformer_hybrid.ViT_hybrid_base_patch16_224()
    model.load_pretrained('models/imagenet2012-ViT-B_16-224')

    model = TensorParallel(model, dist_env.get_hcg(), strategy=None)

else:
    model = vision_transformer.ViT_base_patch16_224()
    model.load_pretrained('models/imagenet2012-ViT-B_16-224')

transform_val = transforms.Compose([
    transforms.Resize(
        size=256, interpolation="bicubic", backend="pil"),  # 3 is bicubic
    transforms.CenterCrop(size=224),
    transforms.NormalizeImage(
        scale=1.0 / 255.0,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        order='hwc'),
    transforms.ToCHWImage()
])

data_path = 'dataset/ILSVRC2012'
dataset_val = datasets.ImageFolder(
    os.path.join(data_path, 'val'), transform=transform_val)

sampler_val = paddle.io.BatchSampler(
    dataset=dataset_val,
    batch_size=128,
    shuffle=False,
    drop_last=False)
data_loader_val = paddle.io.DataLoader(
    dataset_val,
    batch_sampler=sampler_val,
    num_workers=8,
    use_shared_memory=True, )

acc_top1_metric = AverageMeter('top1', '7.5f')
acc_top5_metric = AverageMeter('top5', '7.5f')
loss_metric = AverageMeter('loss', '7.5f')

model.eval()

criterion = paddle.nn.CrossEntropyLoss()
with paddle.no_grad():
    for images, target in data_loader_val:
        # compute output
        with paddle.amp.auto_cast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        loss_metric.update(loss.item())
        acc_top1_metric.update(acc1.item(), n=batch_size)
        acc_top5_metric.update(acc5.item(), n=batch_size)

print(
    '* Acc@1 {top1:.3f} Acc@5 {top5:.3f} loss {losses:.3f}'
    .format(
        top1=acc_top1_metric.avg,
        top5=acc_top5_metric.avg,
        losses=loss_metric.avg))
