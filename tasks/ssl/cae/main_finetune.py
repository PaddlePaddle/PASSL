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
import time
import paddle
import paddle.nn as nn
import json
import os
import shutil
import sys

from pathlib import Path
from visualdl import LogWriter as SummaryWriter

import passl.optimizer as optim
from passl.data import dataset as datasets
from passl.data.utils.batch_collate_fn import collate_fn
from passl.data import preprocess as transforms
from passl.nn import init
from passl.models.utils.pos_embed import interpolate_pos_embed

from util import misc
import util.lr_decay as lrd
from util.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from engine_finetune import train_one_epoch, evaluate
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from scipy import interpolate
from passl.models import cae as models_cae


def get_args():
    parser = argparse.ArgumentParser(
        'CAE fine-tuning and evaluation script for image classification',
        add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument(
        '--accum_iter',
        default=1,
        type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)'
    )
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        default='deit_base_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument(
        '--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--sin_pos_emb', action='store_true')
    parser.set_defaults(sin_pos_emb=True)
    parser.add_argument(
        '--disable_sin_pos_emb', action='store_false', dest='sin_pos_emb')

    parser.add_argument(
        '--layer_scale_init_value',
        default=0.1,
        type=float,
        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument(
        '--input_size', default=224, type=int, help='images input size')

    parser.add_argument(
        '--drop',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Dropout rate (default: 0.)')
    parser.add_argument(
        '--attn_drop_rate',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Attention dropout rate (default: 0.)')
    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.1,
        metavar='PCT',
        help='Drop path rate (default: 0.1)')

    parser.add_argument(
        '--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument(
        '--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument(
        '--model_ema_force_cpu', action='store_true', default=False, help='')

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
        help='Optimizer Betas (default: None, use opt default)')
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
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        metavar='LR',
        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
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
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0'
    )

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter',
        type=float,
        default=0.4,
        metavar='PCT',
        help='Color jitter factor (default: 0.4)')
    parser.add_argument(
        '--aa',
        type=str,
        default='',
        metavar='NAME',
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'
    ),
    parser.add_argument(
        '--smoothing',
        type=float,
        default=0.1,
        help='Label smoothing (default: 0.1)')
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument(
        '--reprob',
        type=float,
        default=0.25,
        metavar='PCT',
        help='Random erase prob (default: 0.25)')
    parser.add_argument(
        '--remode',
        type=str,
        default='pixel',
        help='Random erase mode (default: "pixel")')
    parser.add_argument(
        '--recount',
        type=int,
        default=1,
        help='Random erase count (default: 1)')
    parser.add_argument(
        '--resplit',
        action='store_true',
        default=False,
        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument(
        '--mixup',
        type=float,
        default=0,
        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument(
        '--cutmix',
        type=float,
        default=0,
        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument(
        '--cutmix_minmax',
        type=float,
        nargs='+',
        default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup_prob',
        type=float,
        default=1.0,
        help='Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup_switch_prob',
        type=float,
        default=0.5,
        help='Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup_mode',
        type=str,
        default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', help='finetune from checkpoint')
    parser.add_argument(
        '--model_key', default='model|module|state_dict', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument(
        '--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument(
        '--disable_weight_decay_on_rel_pos_bias',
        action='store_true',
        default=False)

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/datasets01/imagenet_full_size/061417/',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--eval_data_path',
        default=None,
        type=str,
        help='dataset path for evaluation')
    parser.add_argument(
        '--nb_classes',
        default=0,
        type=int,
        help='number of the classification types')
    parser.add_argument(
        '--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument(
        '--data_set',
        default='IMNET',
        choices=['CIFAR', 'IMNET', 'IMNET100', 'image_folder'],
        type=str,
        help='ImageNet dataset path')
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

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument(
        '--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument(
        '--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument(
        '--dist_eval',
        action='store_true',
        default=False,
        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
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

    parser.add_argument(
        '--enable_deepspeed', action='store_true', default=False)
    parser.add_argument(
        '--enable_linear_eval', action='store_true', default=False)
    parser.add_argument(
        '--enable_multi_print',
        action='store_true',
        default=False,
        help='allow each gpu prints something')

    parser.add_argument(
        '--exp_name',
        default='',
        type=str,
        help='name of exp. it is helpful when save the checkpoint')

    parser.add_argument(
        '--linear_type',
        default='standard',
        type=str,
        help='[standard, attentive, attentive_no_parameter]')
    parser.add_argument('--linear_depth', default=1, type=int, help=' ')

    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument(
        '--max_train_step',
        default=None,
        type=int,
        help='only used for debugging')

    return parser.parse_args()


def main(args):

    if not args.enable_linear_eval:
        args.aa = 'rand-m9-mstd0.5-inc1'

    misc.init_distributed_mode(args)

    print(args)

    device = paddle.set_device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    paddle.seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    RELATED_FLAGS_SETTING = {}
    RELATED_FLAGS_SETTING['FLAGS_cudnn_deterministic'] = 1
    paddle.fluid.set_flags(RELATED_FLAGS_SETTING)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            args.input_size, interpolation="bicubic"),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.TimmAutoAugment(
            config_str=args.aa,
            interpolation="bicubic",
            img_size=args.input_size,
            mean=[0.485, 0.456, 0.406]),
        transforms.NormalizeImage(
            scale=1.0 / 255.0,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order='hwc'),
        transforms.RandomErasing(
            EPSILON=args.reprob,
            sl=0.02,
            sh=1.0 / 3.0,
            r1=0.3,
            attempt=10,
            use_log_aspect=True,
            mode=args.remode),
        transforms.ToCHWImage()
    ])

    batch_transform_ops = {}
    batch_transform_ops['Mixup'] = {
        "alpha": args.mixup,
        "prob": args.mixup_switch_prob,
        "epsilon": args.smoothing,
        "num_classes": args.nb_classes
    }
    batch_transform_ops['Cutmix'] = {
        "alpha": args.cutmix,
        "prob": args.mixup_switch_prob,
        "epsilon": args.smoothing,
        "num_classes": args.nb_classes
    }
    mixup_fn = transforms.TransformOpSampler(**batch_transform_ops)

    def mixup_collate_fn(batch):
        batch = mixup_fn(batch)
        batch = collate_fn(batch)
        return batch

    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'), transform=transform_train)

    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        transform_val = transforms.Compose([
            transforms.Resize(
                size=256, interpolation="bicubic",
                backend="pil"),  # 3 is bicubic
            transforms.CenterCrop(size=224),
            transforms.NormalizeImage(
                scale=1.0 / 255.0,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                order='hwc'),
            transforms.ToCHWImage()
        ])
        dataset_val = datasets.ImageFolder(
            os.path.join(args.data_path, 'val'), transform=transform_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        num_training_steps_per_epoch = len(
            dataset_train) // args.batch_size // num_tasks
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
        use_shared_memory=args.pin_mem,
        collate_fn=mixup_collate_fn, )

    if dataset_val is not None:
        data_loader_val = paddle.io.DataLoader(
            dataset_val,
            batch_sampler=sampler_val,
            num_workers=args.num_workers,
            use_shared_memory=args.pin_mem, )
    else:
        data_loader_val = None

    model = models_cae.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        lin_probe=args.enable_linear_eval,
        args=args, )

    if args.enable_linear_eval:
        # freeze all layers but the last fc
        linear_keyword = 'head'
        head_norm = 'fc_norm'
        requires_grad = []
        for name, param in model.named_parameters():
            if name not in [
                    '%s.weight' % linear_keyword, '%s.bias' % linear_keyword
            ] and head_norm not in name:
                param.stop_gradient = True
            else:
                requires_grad.append(name)
        print(f'require grad parameter: ', requires_grad)
        # init the fc layer
        init.normal_(getattr(model, linear_keyword).weight)
        init.zeros_(getattr(model, linear_keyword).bias)

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = model.patch_embed.grid_size
    args.patch_size = patch_size

    if args.finetune:
        checkpoint = paddle.load(args.finetune)

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        all_keys = list(checkpoint_model.keys())
        print("##########origin keys:", len(all_keys), all_keys)
        # NOTE: remove all decoder keys
        all_keys = [key for key in all_keys if key.startswith('encoder.')]
        print("all keys:", all_keys)
        for key in all_keys:
            new_key = key.replace('encoder.', '')
            # print("new_key:", new_key)
            checkpoint_model[new_key] = checkpoint_model[key]
            checkpoint_model.pop(key)

        for key in list(checkpoint_model.keys()):
            if key.startswith('regressor_and_decoder.'):
                # print("key:", key)
                checkpoint_model.pop(key)
            if key.startswith('teacher_network.'):
                # print("key:", key)
                checkpoint_model.pop(key)

        # NOTE: replace norm with fc_norm
        for key in list(checkpoint_model.keys()):
            # print("new key:", key)
            if key.startswith('norm.'):
                new_key = key.replace('norm.', 'fc_norm.')
                checkpoint_model[new_key] = checkpoint_model[key]
                checkpoint_model.pop(key)

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[
                    k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print(
                "Expand the shared relative position embedding to each transformer block. "
            )
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model[
                "rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table"
                                 % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        all_keys = list(checkpoint_model.keys())

        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key and args.rel_pos_bias:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = model.state_dict()[key].size()
                dst_patch_shape = model.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (
                    dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens)**0.5)
                dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" %
                          (key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q**(i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size,
                                                    src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            paddle.Tensor(f(dx, dy)).contiguous().view(-1, 1)
                            .to(rel_pos_bias.device))

                    rel_pos_bias = paddle.concat(all_rel_pos_bias, axis=-1)

                    new_rel_pos_bias = paddle.concat(
                        (rel_pos_bias, extra_tokens), axis=0)
                    checkpoint_model[key] = new_rel_pos_bias

        print("##############new keys:",
              len(checkpoint_model), checkpoint_model.keys())
        #print("##############model:", model)

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model and args.abs_pos_emb:
            interpolate_pos_embed(model, checkpoint_model)

        model.set_state_dict(checkpoint_model)

    if args.distributed:
        model = paddle.DataParallel(model)
        model_without_ddp = model._layers

    # define scaler for AMP training
    loss_scaler = NativeScaler()

    n_parameters = sum(p.numel() for p in model.parameters()
                       if not p.stop_gradient).item()

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.accum_iter * misc.get_world_size(
    )
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Accumulate grad iterations: %d" % args.accum_iter)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" %
          num_training_steps_per_epoch)

    no_weight_decay_list = {}
    if hasattr(model_without_ddp, 'no_weight_decay'):
        no_weight_decay_list = model_without_ddp.no_weight_decay()

    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(len(model_without_ddp.blocks)):
            no_weight_decay_list.add(
                "blocks.%d.attn.relative_position_bias_table" % i)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=no_weight_decay_list,
        layer_decay=args.layer_decay)

    optimizer = optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = paddle.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.auto_load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model)
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
        # if log_writer is not None:
        #     log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            loss_scaler,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            args=args)

        if args.output_dir and args.save_ckpt:
            if (epoch + 1
                ) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch)
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model)
            print(
                f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
            )
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    misc.save_model(
                        args=args,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best")

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'],
                                      epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'],
                                      epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'],
                                      epoch)

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
        else:
            log_stats = {
                **
                {f'train_{k}': v
                 for k, v in train_stats.items()},
                # **{f'test_{k}': v for k, v in test_stats.items()},
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
