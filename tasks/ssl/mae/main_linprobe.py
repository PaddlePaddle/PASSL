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
from passl.data import dataset as datasets

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from passl.nn.init import trunc_normal_

from passl.models import mae as models_mae
from passl.models import convmae as models_convmae

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser(
        'MAE linear probing for image classification', add_help=False)
    parser.add_argument(
        '--batch_size',
        default=512,
        type=int,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus'
    )
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument(
        '--accum_iter',
        default=1,
        type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)'
    )

    # Model parameters
    parser.add_argument(
        '--model',
        default='vit_large_patch16',
        type=str,
        metavar='MODEL',
        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        metavar='LR',
        help='learning rate (absolute lr)')
    parser.add_argument(
        '--blr',
        type=float,
        default=0.1,
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
        default=10,
        metavar='N',
        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument(
        '--cls_token',
        action='store_false',
        dest='global_pool',
        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/datasets01/imagenet_full_size/061417/',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--nb_classes',
        default=1000,
        type=int,
        help='number of the classification types')

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
    parser.add_argument(
        '--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument(
        '--dist_eval',
        action='store_true',
        default=False,
        help='Enabling distributed evaluation (recommended during training for faster monitor'
    )
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

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
        transforms.MAERandCropImage(
            224, interpolation="bicubic", backend="pil"),
        transforms.RandomHorizontalFlip(), transforms.NormalizeImage(
            scale=1.0 / 255.0,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order='hwc'), transforms.ToCHWImage()
    ])

    transform_val = transforms.Compose([
        transforms.Resize(
            size=256, interpolation="bicubic", backend="pil"),  # 3 is bicubic
        transforms.CenterCrop(size=224),
        transforms.NormalizeImage(
            scale=1.0 / 255.0,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order='hwc'),
        transforms.ToCHWImage()
    ])

    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(
        os.path.join(args.data_path, 'val'), transform=transform_val)
    print(dataset_train)
    print(dataset_val)

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
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = paddle.io.DistributedBatchSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
                batch_size=args.batch_size,
                drop_last=False)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = paddle.io.BatchSampler(
                dataset=dataset_val,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False)
    else:
        sampler_train = paddle.io.BatchSampler(
            dataset=dataset_val,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True)
        sampler_val = paddle.io.BatchSampler(
            dataset=dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = paddle.io.DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=args.num_workers,
        use_shared_memory=args.pin_mem, )

    data_loader_val = paddle.io.DataLoader(
        dataset_val,
        batch_sampler=sampler_val,
        num_workers=args.num_workers,
        use_shared_memory=args.pin_mem, )

    if 'convvit' in args.model:
        model = models_convmae.__dict__[args.model](
            num_classes=args.nb_classes,
            global_pool=args.global_pool, )
    else:
        model = models_mae.__dict__[args.model](
            num_classes=args.nb_classes,
            global_pool=args.global_pool, )

    if args.finetune and not args.eval:
        checkpoint = paddle.load(args.finetune)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and list(checkpoint_model[
                    k].shape) != list(state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        if 'convvit' in args.model:
            pass
        else:
            interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        model.set_state_dict(checkpoint_model)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = paddle.nn.Sequential(
        paddle.nn.BatchNorm1D(
            model.head.weight.shape[0],
            weight_attr=False,
            bias_attr=False,
            epsilon=1e-6),
        model.head)

    # freeze all first but the head
    # Note(GuoxiaWang): Although weight_attr and bias_attr are set to False
    # but weight, bias, _mean, _variance will still be created as param,
    # so we only set p.stop_gradient = False in lastest classifer layer
    for _, p in model.named_parameters():
        p.stop_gradient = True

    for _, p in model.head[1].named_parameters():
        p.stop_gradient = False

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if not p.stop_gradient).item()

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

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

    optimizer = passl.optimizer.MomentumLARS(
        model_without_ddp.head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = paddle.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.batch_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args)
        if args.output_dir:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {
            **
            {f'train_{k}': v
             for k, v in train_stats.items()},
            **
            {f'test_{k}': v
             for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
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
