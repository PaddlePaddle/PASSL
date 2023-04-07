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
import datetime
import numpy as np
import os
import sys

import time
import json
import os
import random
import shutil

import paddle

import paddle.nn as nn

from pathlib import Path
from visualdl import LogWriter as SummaryWriter

import passl.optimizer as optim
from passl.scheduler import TimmCosine
from passl.data import preprocess as transforms
from passl.data import dataset as datasets

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import dall_e
from util.masking_generator import MaskingGenerator, RandomMaskingGenerator

from passl.models import cae as models_cae
from engine_pretrain import train_one_epoch


def get_args():
    parser = argparse.ArgumentParser('pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--save_ckpt_toBPFS_freq', default=50, type=int)
    parser.add_argument("--discrete_vae_weight_path", type=str)
    parser.add_argument("--discrete_vae_type", type=str, default="dall-e")
    parser.add_argument(
        '--amp', action='store_true', default=False, help='if or not use amp')
    # Model parameters
    parser.add_argument(
        '--model',
        default='deit_base_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true', default=False)
    parser.add_argument(
        '--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.add_argument('--abs_pos_emb', action='store_true', default=False)
    parser.add_argument('--sincos_pos_emb', action='store_true', default=False)
    parser.add_argument(
        '--layer_scale_init_value',
        default=0.1,
        type=float,
        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument(
        '--num_mask_patches',
        default=75,
        type=int,
        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument(
        '--input_size',
        default=224,
        type=int,
        help='images input size for backbone')
    parser.add_argument(
        '--second_input_size',
        default=112,
        type=int,
        help='images input size for discrete vae')

    parser.add_argument(
        '--drop_path',
        type=float,
        default=0,
        metavar='PCT',
        help='Drop path rate (default: 0)')

    # Optimizer parameters
    parser.add_argument(
        '--opt',
        default='adamw',
        type=str,
        metavar='OPTIMIZER',
        help='Optimizer (default: "adamw"')
    parser.add_argument(
        '--opt_eps',
        default=1e-8,
        type=float,
        metavar='EPSILON',
        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: 0.9, 0.98, use opt default)')
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='SGD momentum (default: 0.9)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='weight decay (default: 0.05)')
    parser.add_argument(
        '--weight_decay_end',
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD.
        (Set the same value with args.weight_decay to keep weight decay no change)"""
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        metavar='LR',
        help='learning rate (default: 5e-4)')
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )
    parser.add_argument(
        '--second_interpolation',
        type=str,
        default='lanczos',
        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")'
    )

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/datasets01/imagenet_full_size/061417/',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument(
        '--output_dir',
        default='',
        help='path where to save, empty for no saving')
    parser.add_argument(
        '--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument(
        '--device', default='gpu', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument(
        '--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training')

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument(
        '--enable_multi_print',
        action='store_true',
        default=False,
        help='allow each gpu prints something')
    parser.add_argument(
        '--regressor_depth',
        default=4,
        type=int,
        help='depth of self-attention block for decoder')
    parser.add_argument(
        '--num_decoder_self_attention',
        default=0,
        type=int,
        help='number of self-attention in decoder')

    parser.add_argument(
        '--decoder_embed_dim',
        default=768,
        type=int,
        help='dimensionaltiy of embeddings for decoder')
    parser.add_argument(
        '--decoder_num_heads',
        default=12,
        type=int,
        help='Number of heads for decoder')
    parser.add_argument(
        '--decoder_num_classes',
        default=8192,
        type=int,
        help='Number of classes for decoder')
    parser.add_argument(
        '--decoder_layer_scale_init_value',
        default=0.1,
        type=float,
        help='decoder layer scale init value')

    parser.add_argument(
        '--mask_generator',
        default='block',
        type=str,
        help='choice = [block, random]')
    parser.add_argument(
        '--ratio_mask_patches',
        default=None,
        type=float,
        help="mask ratio. only use when 'mask_generator' is random")

    # color jitter, default is False
    parser.add_argument(
        '--color_jitter',
        type=float,
        default=0,
        metavar='PCT',
        help='Color jitter factor (default: 0)')
    parser.add_argument(
        '--exp_name',
        default='',
        type=str,
        help='name of exp. it is helpful when save the checkpoint')

    parser.add_argument(
        '--target_mode',
        default='clusterID',
        type=str,
        help='target, [clusterID, rgb, random]')
    parser.add_argument(
        '--target_path',
        default='/home/vis/bpfsrw5/cxk/dalle-weights/random_vector_768x8192.pth',
        type=str,
        help='path to load target vectors')

    parser.add_argument(
        '--normalized_pixel',
        default='layernorm',
        type=str,
        help='how to generate the regression target, [layernorm, none, channel, patch]'
    )
    parser.add_argument(
        '--denorm',
        action='store_true',
        default=False,
        help='if true, the RGB target will be denorm first')

    parser.add_argument(
        '--rescale_init',
        action='store_true',
        default=False,
        help='if true, the fix_init_weight() func will be activated')
    # dual path CAE
    parser.add_argument(
        '--dual_loss_weight',
        type=float,
        default=1,
        help='loss weight for the dual path loss')
    parser.add_argument(
        '--dual_loss_type', type=str, default='mse', help='[mse, kld]')
    parser.add_argument(
        '--dual_path_ema',
        type=float,
        default=0,
        help='ema weight for the dual path network')

    # crop size
    parser.add_argument(
        '--crop_min_size', type=float, default=0.08, help='min size of crop')
    parser.add_argument(
        '--crop_max_size', type=float, default=1.0, help='max size of crop')

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument(
        '--max_train_step',
        default=None,
        type=int,
        help='only used for debugging')

    return parser.parse_args()


class DataAugmentationForCAE(object):
    def __init__(self, args):

        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = misc.ADE_DEFAULT_MEAN if not imagenet_default_mean_and_std else misc.IMAGENET_DEFAULT_MEAN
        std = misc.ADE_DEFAULT_STD if not imagenet_default_mean_and_std else misc.IMAGENET_DEFAULT_STD

        common_transform = []
        if args.color_jitter > 0:
            common_transform.append(
                transforms.ColorJitter(args.color_jitter, args.color_jitter,
                                       args.color_jitter))
        common_transform.append(transforms.RandomHorizontalFlip())
        common_transform.append(
            transforms.RandomResizedCropWithTwoImages(
                args.input_size,
                second_size=args.second_input_size,
                interpolation=args.train_interpolation,
                second_interpolation=args.second_interpolation,
                scale=(args.crop_min_size, args.crop_max_size)))
        self.common_transform = transforms.Compose(common_transform)

        self.patch_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(
                mean=mean, std=std)])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                dall_e.utils.map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD, ),
            ])
        else:
            raise NotImplementedError()

        if args.mask_generator == 'block':
            self.masked_position_generator = MaskingGenerator(
                args.window_size,
                num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block, )
        elif args.mask_generator == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size,
                ratio_masking_patches=args.ratio_mask_patches)

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)

        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForCAE,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(
            self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator)
        repr += ")"
        return repr


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

    if args.target_mode == 'rgb':
        assert args.discrete_vae_type == "to_tensor"

    print(f"Creating model: {args.model}")
    model = models_cae.__dict__[args.model](
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        args=args, )

    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = model.encoder.patch_embed.grid_size
    args.patch_size = patch_size

    transform_train = DataAugmentationForCAE(args)
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    # prepare discrete vae
    d_vae = dall_e.create_d_vae(
        weight_path=args.discrete_vae_weight_path,
        d_vae_type=args.discrete_vae_type,
        image_size=args.second_input_size)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    num_training_steps_per_epoch = len(
        dataset_train) // args.batch_size // num_tasks
    sampler_train = paddle.io.DistributedBatchSampler(
        dataset_train,
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
        places=device, )

    n_parameters = sum(p.numel().item() for p in model.parameters()
                       if not p.stop_gradient)

    model_without_ddp = model
    if args.distributed:
        model = paddle.DataParallel(model)
        model_without_ddp = model._layers

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * misc.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" %
          (total_batch_size * num_training_steps_per_epoch))

    # following timm: set wd as 0 for bias and norm layers
    decay = []
    no_decay = []
    skip_list = {}
    if hasattr(model_without_ddp, 'no_weight_decay'):
        skip_list = model_without_ddp.no_weight_decay()
    for name, param in model_without_ddp.named_parameters():
        if 'teacher' in name:
            continue
        if param.stop_gradient:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(
                ".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': args.weight_decay
    }]

    lr_scheduler = TimmCosine(
        args.lr,
        num_training_steps_per_epoch,
        args.epochs,
        decay_unit='step',
        eta_min=args.min_lr,
        warmup_epoch=args.warmup_epochs,
        warmup_prefix=True)

    optimizer = optim.AdamW(param_groups, lr=lr_scheduler, betas=(0.9, 0.999))

    loss_scaler = NativeScaler()

    assert args.weight_decay_end is None

    misc.auto_load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.batch_sampler.set_epoch(epoch)

        #if log_writer is not None:
        #    log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model,
            d_vae,
            data_loader_train,
            optimizer,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            args=args, )
        if args.output_dir:
            if (epoch + 1
                ) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    exp_name=args.exp_name)

        log_stats = {
            **
            {f'train_{k}': v
             for k, v in train_stats.items()},
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
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
