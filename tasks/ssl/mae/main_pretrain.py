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

# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import paddle
import passl.optimizer
from visualdl import LogWriter as SummaryWriter
from passl.data import preprocess as transforms

import util.misc as misc
from passl.data import dataset as datasets
import util.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from passl.models import mae as models_mae
from passl.models import convmae as models_convmae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus'
    )
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument(
        '--accum_iter',
        default=1,
        type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)'
    )

    # Model parameters
    parser.add_argument(
        '--model',
        default='mae_vit_large_patch16',
        type=str,
        metavar='MODEL',
        help='Name of model to train')

    parser.add_argument(
        '--input_size', default=224, type=int, help='images input size')

    parser.add_argument(
        '--mask_ratio',
        default=0.75,
        type=float,
        help='Masking ratio (percentage of removed patches).')

    parser.add_argument(
        '--norm_pix_loss',
        action='store_true',
        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='weight decay (default: 0.05)')

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        metavar='LR',
        help='learning rate (absolute lr)')
    parser.add_argument(
        '--blr',
        type=float,
        default=1e-3,
        metavar='LR',
        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=0.,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=40,
        metavar='N',
        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/datasets01/imagenet_full_size/061417/',
        type=str,
        help='dataset path')

    parser.add_argument(
        '--output_dir',
        default='./output_dir',
        help='path where to save, empty for no saving')
    parser.add_argument(
        '--log_dir',
        default='./output_dir',
        help='path where to tensorboard log')
    parser.add_argument(
        '--device', default='gpu', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument(
        '--max_train_step',
        default=None,
        type=int,
        help='only used for debugging')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = paddle.set_device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    paddle.seed(seed)
    np.random.seed(seed)

    RELATED_FLAGS_SETTING = {}
    RELATED_FLAGS_SETTING['FLAGS_cudnn_deterministic'] = 1
    paddle.fluid.set_flags(RELATED_FLAGS_SETTING)

    # simple augmentation
    # transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation="bicubic"),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            args.input_size, scale=(0.2, 1.0),
            interpolation="bicubic"),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.NormalizeImage(
            scale=1.0 / 255.0,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order='hwc'),
        transforms.ToCHWImage()
    ])
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = paddle.io.DistributedBatchSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
            batch_size=args.batch_size,
            drop_last=True)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = paddle.io.BatchSampler(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = paddle.io.DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=args.num_workers,
        use_shared_memory=args.pin_mem, )

    # define the model
    if 'convmae' in args.model:
        model = models_convmae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss)
    else:
        model = models_mae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = paddle.DataParallel(model)
        model_without_ddp = model._layers

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp,
                                                  args.weight_decay)
    optimizer = passl.optimizer.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.batch_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args)
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch)

        log_stats = {
            **
            {f'train_{k}': v
             for k, v in train_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                    os.path.join(args.output_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
