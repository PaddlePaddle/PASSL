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

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from passl.data import preprocess as transforms
from passl.data import dataset as datasets
from visualdl import LogWriter as SummaryWriter

import passl

import builder_moco
import vit_moco

model_names = [
    'moco_vit_small', 'moco_vit_base', 'moco_vit_conv_small',
    'moco_vit_conv_base'
]

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument(
    '-a',
    '--arch',
    metavar='ARCH',
    default='resnet50',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet50)')
parser.add_argument(
    '-j',
    '--workers',
    default=8,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 8)')
parser.add_argument(
    '--epochs',
    default=100,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=4096,
    type=int,
    metavar='N',
    help='mini-batch size (default: 4096), this is the total '
    'batch size of all GPUs on all nodes when '
    'using Data Parallel or Distributed Data Parallel')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.6,
    type=float,
    metavar='LR',
    help='initial (base) learning rate',
    dest='lr')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--wd',
    '--weight-decay',
    default=1e-6,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-6)',
    dest='weight_decay')
parser.add_argument(
    '-p',
    '--print-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--world-size',
    default=-1,
    type=int,
    help='number of nodes for distributed training')
parser.add_argument(
    '--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument(
    '--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')

# moco specific configs:
parser.add_argument(
    '--moco-dim',
    default=256,
    type=int,
    help='feature dimension (default: 256)')
parser.add_argument(
    '--moco-mlp-dim',
    default=4096,
    type=int,
    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument(
    '--moco-m',
    default=0.99,
    type=float,
    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument(
    '--moco-m-cos',
    action='store_true',
    help='gradually increase moco momentum to 1 with a '
    'half-cycle cosine schedule')
parser.add_argument(
    '--moco-t',
    default=1.0,
    type=float,
    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument(
    '--stop-grad-conv1',
    action='store_true',
    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument(
    '--optimizer',
    default='lars',
    type=str,
    choices=['lars', 'adamw'],
    help='optimizer used (default: lars)')
parser.add_argument(
    '--warmup-epochs',
    default=10,
    type=int,
    metavar='N',
    help='number of warmup epochs')
parser.add_argument(
    '--crop-min',
    default=0.08,
    type=float,
    help='minimum scale for random cropping (default: 0.08)')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        RELATED_FLAGS_SETTING = {}
        RELATED_FLAGS_SETTING['FLAGS_cudnn_deterministic'] = 1
        paddle.fluid.set_flags(RELATED_FLAGS_SETTING)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    device = paddle.set_device("gpu")
    dist.init_parallel_env()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.distributed = args.world_size > 1

    # suppress printing if not first GPU on each node
    if args.rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    # create model
    print("=> creating model '{}'".format(args.arch))

    model = builder_moco.MoCo_ViT(
        partial(
            vit_moco.__dict__[args.arch],
            stop_grad_conv1=args.stop_grad_conv1),
        args.moco_dim,
        args.moco_mlp_dim,
        args.moco_t)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if args.distributed:
        # apply SyncBN
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        args.batch_size = int(args.batch_size / args.world_size)
        model = paddle.DataParallel(model)

    print(model)  # print model after SyncBatchNorm

    if args.optimizer == 'lars':
        optimizer = passl.optimizer.MomentumLARS(
            model.parameters(),
            args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = passl.optimizer.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay)

    scaler = paddle.amp.GradScaler(
        init_loss_scaling=2.**16, incr_every_n_steps=2000)

    summary_writer = SummaryWriter() if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = paddle.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.set_state_dict(checkpoint['state_dict'])
            optimizer.set_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(
            224, scale=(args.crop_min, 1.)),
        transforms.RandomApply(
            [
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ],
            p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.SimCLRGaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(
            224, scale=(args.crop_min, 1.)),
        transforms.RandomApply(
            [
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ],
            p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.SimCLRGaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply(
            [transforms.BYOLSolarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.TwoViewsTransform(
            transforms.Compose(augmentation1),
            transforms.Compose(augmentation2)))

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True)

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.workers,
        use_shared_memory=True, )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.batch_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch,
              args)

        if args.rank == 0 and epoch % 10 == 0 or epoch == args.epochs - 1:  # only the first GPU saves checkpoint
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                },
                is_best=False,
                filename='checkpoint_%04d.pd' % epoch)

    if args.rank == 0:
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        images[0] = images[0].cuda()
        images[1] = images[1].cuda()

        # compute output
        with paddle.amp.auto_cast():
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].shape[0])
        if args.rank == 0:
            summary_writer.add_scalar("loss",
                                      loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.clear_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pd'):
    paddle.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pd')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (
            1. + math.cos(math.pi * (epoch - args.warmup_epochs) /
                          (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (
        1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
