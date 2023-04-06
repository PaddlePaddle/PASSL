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

import paddle
import paddle.nn as nn
import paddle.distributed as dist

import passl
from passl.utils import utils

from visualdl import LogWriter as SummaryWriter



def parse_args():
    parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')

    # Common params
    parser.add_argument(
        '--train_mode',
        help='The three training model choose from ["pretrain", "linear_probe", "finetune"].',
        type=str,
        default='pretrain')
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--device',
        help='Set the device place for training model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the model snapshot.',
        type=str,
        default='./output')
    parser.add_argument(
        '-j',
        '--num_workers',
        help='Number of workers for data loader. Bigger num_workers can speed up data processing.',
        type=int,
        default=8)
    parser.add_argument(
        '--use_vdl',
        help='Whether to record the data to VisualDL in training.',
        action='store_true')
    parser.add_argument(
        '--use_ema',
        help='Whether to ema the model in training.',
        action='store_true')

    # Runntime params
    parser.add_argument(
        '--resume_model',
        help='The path of the model to resume training.',
        type=str)
    parser.add_argument(
        '--epochs', 
        default=100,
        help='Epochs in training.',
        type=int)
    parser.add_argument(
        '-b',
        '--batch_size',
        default=4096,
        help='Mini batch size of one gpu or cpu. ', 
        type=int)
    parser.add_argument(
        '--lr',
        '--learning_rate',
        default=0.6,
        help='Learning rate.', 
        type=float)
    parser.add_argument(
        '--warmup_epochs',
        default=10,
        help="num of epochs for linear warmup.", 
        type=int)

    parser.add_argument(
        '--save_interval',
        help='How many epochs to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--log_iters',
        help='Display logging information at every `log_iters`.',
        default=10,
        type=int)

    # Other params
    parser.add_argument(
        '--seed',
        help='Set the random seed in training.',
        default=None,
        type=int)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal."
    )
    parser.add_argument(
        "--amp_level",
        default="O1",
        type=str,
        choices=["O1", "O2"],
        help="Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input \
                data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators \
                parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel \
                and batchnorm. Default is O1(amp).")

    parser.add_argument(
        '--opts', help='Update the key-value pairs of all options.', nargs='+')

    return parser.parse_args()

################################################
## Code need to be modified to compact Trainer #
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
################################################
def main():
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


    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if args.distributed:
        # apply SyncBN
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        args.batch_size = int(args.batch_size / args.world_size)
        model = paddle.DataParallel(model)

    print(model)  # print model after SyncBatchNorm

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
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    learning_rates = utils.AverageMeter('LR', ':.4e')
    losses = utils.AverageMeter('Loss', ':.4e')
    progress = utils.ProgressMeter(
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
################################################
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
################################################



# code starts here
class MOCOV3PretrainTrainer([passl.core.PasslTrainer]):
    def __init__(self) -> None:
        """
        The init part has three important components:
        self.builder: contains all the component build from config: loss, models,backbones, dataset, optimier, transforms.
        self.args: The cli args, mainly includes the training configs.
        self.cfg: The config get from the yaml, update serveral configs from args as well, for print to the log. 
        """
        super().__init__(args)

    
    def train(self):
        """
        Train the model
        """


if __name__ == '__main__':
    args = parse_args()
    if args.train_mode == "pretrain":
        trainer = MOCOV3PretrainTrainer(args)
    elif args.train_mode == "finetune":
        pass
        # trainer = MOCOV3FinetuneTrainer(args)
    elif args.train_mode == "linear_probe":
        pass
        # trainer = MOCOV3LinearTrainer(args)
    trainer.train()
