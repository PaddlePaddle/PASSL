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

import math
import sys
import time
from typing import Iterable

import paddle
import paddle.nn as nn
import paddle.amp as amp
import passl.optimizer as optim

import util.misc as misc
import paddle.nn.functional as F


def loss_selector(loss_type, pred, target):
    if loss_type == 'mse':
        return F.mse_loss(pred, target, reduction="mean")
    elif loss_type == 'kld':
        return F.kl_div(
            F.log_softmax(
                pred, axis=-1),
            F.softmax(
                target, axis=-1),
            reduction='mean')


def train_one_epoch(model: nn.Layer,
                    d_vae: nn.Layer,
                    data_loader: Iterable,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    loss_scaler,
                    max_norm: float=0,
                    log_writer=None,
                    start_steps=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', misc.SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for step, (batch, _) in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)):

        global_iter_step = step + len(data_loader) * epoch
        if args.max_train_step is not None and global_iter_step >= args.max_train_step:
            print(
                f'step({global_iter_step}) >= max_train_step({args.max_train_step}), training stops early. This function is only used for debugging.'
            )
            exit(0)

        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        optimizer.lr_step(it)

        samples, images, bool_masked_pos = batch

        with paddle.no_grad():
            if args.target_mode == 'clusterID':
                input_ids = d_vae.get_codebook_indices(images).flatten(1)
            elif args.target_mode == 'random':
                input_ids = images
            elif args.target_mode == 'rgb':
                bsz = samples.shape[0]

                if args.denorm:
                    mean = paddle.to_tensor(misc.IMAGENET_DEFAULT_MEAN)[
                        None, :, None, None]
                    std = paddle.to_tensor(misc.IMAGENET_DEFAULT_STD)[
                        None, :, None, None]
                    target = samples * std + mean  # in [0, 1]
                else:
                    target = samples

                target = \
                    target.reshape([bsz, 3, args.window_size[0], args.patch_size[0], args.window_size[1], args.patch_size[1]])

                num_patches = args.window_size[0] * args.window_size[1]
                if args.normalized_pixel == 'none':
                    target = target.transpose([0, 2, 4, 3, 5, 1]).reshape(
                        [bsz, num_patches, -1])
                elif args.normalized_pixel == 'channel':  # norm the pixels in a patch along a certain channel
                    # Bx3x14x16x14x16 --> Bx(14*14)x(16*16*3)
                    target = target.transpose([0, 2, 4, 3, 5, 1]).reshape(
                        [bsz, num_patches, -1, 3])
                    mean = target.mean(axis=-2, keepdim=True)
                    std = target.var(axis=-2, unbiased=True,
                                     keepdim=True).sqrt()
                    target = (target - mean) / (std + 1e-6)
                    target = target.transpose(0, 1, 2, 5, 3, 4).reshape(
                        [bsz, num_patches, -1])  # channel x Pw x Ph
                elif args.normalized_pixel == 'patch':  # norm the pixels in a patch
                    target = target.transpose([0, 2, 4, 3, 5, 1]).reshape(
                        [bsz, num_patches, -1])
                    mean = target.mean(axis=-1, keepdim=True)
                    var = target.var(axis=-1, keepdim=True)
                    target = (target - mean) / (var + 1.e-6)**.5
                elif args.normalized_pixel == 'layernorm':  # norm the pixels with F.layer_norm
                    target = target.transpose([0, 2, 4, 3, 5, 1]).reshape(
                        [bsz, num_patches, -1])
                    target = F.layer_norm(
                        target, target.shape[-1:], epsilon=1e-6)
                else:
                    exit(0)
                # split_into_patches = split_into_patches.transpose([0, 2, 4, 3, 5, 1])
                # split_into_patches = split_into_patches.flatten(3).flatten(1, 2)
                input_ids = target
            else:
                raise NotImplementedError()

            bool_masked_pos = bool_masked_pos.flatten(1).astype(paddle.bool)
            labels = input_ids[bool_masked_pos]

        # all
        amp_black_list = {}
        with amp.auto_cast(
                enable=True, custom_black_list=amp_black_list, level='O1'):
            # dual path cae
            outputs, latent, latent_target = model(
                samples,
                bool_masked_pos=bool_masked_pos,
                return_all_tokens=False)

            if args.target_mode == 'clusterID' or args.target_mode == 'random':
                loss_main = nn.CrossEntropyLoss()(
                    input=outputs.astype(paddle.float32), label=labels)
                loss_aux = args.dual_loss_weight * loss_selector(
                    args.dual_loss_type,
                    latent.astype(paddle.float32),
                    latent_target.detach().astype(paddle.float32))
                loss = loss_main + loss_aux
            elif args.target_mode == 'rgb':
                loss_main = F.mse_loss(
                    outputs.astype(paddle.float32),
                    labels.astype(paddle.float32),
                    reduction="mean")
                loss_aux = args.dual_loss_weight * loss_selector(
                    args.dual_loss_type,
                    latent.astype(paddle.float32),
                    latent_target.detach().astype(paddle.float32))
                loss = loss_main + loss_aux
            else:
                exit(0)

        loss_value = loss.item()
        loss_main_value = loss_main.item()
        loss_aux_value = loss_aux.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.clear_grad()
        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=max_norm, parameters=model.parameters())
        loss_scale_value = loss_scaler.state_dict().get("scale").item()

        paddle.device.cuda.synchronize()

        if args.target_mode == 'clusterID':
            mlm_acc = (outputs.argmax(-1) == labels
                       ).astype(paddle.float32).mean().item()

            metric_logger.update(mlm_acc=mlm_acc)
            if log_writer is not None:
                log_writer.add_scalar('mlm_acc', mlm_acc, step=it)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_main=loss_main_value)
        metric_logger.update(loss_aux=loss_aux_value)
        metric_logger.update(loss_scale=loss_scale_value)

        metric_logger.update(lr=optimizer.get_lr())
        metric_logger.update(weight_decay=args.weight_decay)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.add_scalar('loss', loss_value, step=it)
            log_writer.add_scalar('loss_main', loss_main_value, step=it)
            log_writer.add_scalar('loss_aux', loss_aux_value, step=it)
            log_writer.add_scalar('loss_scale', loss_scale_value, step=it)
            log_writer.add_scalar('lr', optimizer.get_lr(), step=it)
            log_writer.add_scalar('weight_decay', args.weight_decay, step=it)
            log_writer.add_scalar('grad_norm', grad_norm, step=it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(now_time, "Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
