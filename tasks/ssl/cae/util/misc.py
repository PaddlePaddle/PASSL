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

import builtins
import datetime
import os
import time
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

import paddle
import paddle.distributed as dist

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
ADE_DEFAULT_MEAN = (0.48897026, 0.46548377, 0.42939525)
ADE_DEFAULT_STD = (0.22846712, 0.22941928, 0.24038891)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = paddle.to_tensor([self.count, self.total], dtype="float64")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = paddle.to_tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = paddle.to_tensor(list(self.deque), dtype=paddle.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
            'time: {time}', 'data: {data}'
        ]
        if paddle.fluid.core.is_compiled_with_cuda():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if paddle.fluid.core.is_compiled_with_cuda():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=paddle.device.cuda.max_memory_allocated() /
                            MB))
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not paddle.fluid.core.is_compiled_with_cuda():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        paddle.save(*args, **kwargs)


def init_distributed_mode(args):
    dist.init_parallel_env()

    args.distributed = True

    device = paddle.set_device(args.device)
    if dist.get_world_size() > 1:
        paddle.distributed.barrier()
    setup_for_distributed(dist.get_rank() == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = paddle.amp.GradScaler(
            init_loss_scaling=2.**16,
            incr_every_n_steps=2000,
            decr_every_n_nan_or_inf=1, )  # same as pytorch

    def __call__(self,
                 loss,
                 optimizer,
                 clip_grad=None,
                 parameters=None,
                 update_grad=True):
        self._scaler.scale(loss).backward()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


@paddle.no_grad()
def clip_grad_norm_(parameters,
                    max_norm: float,
                    norm_type: float=2.0,
                    error_if_nonfinite: bool=False):
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return paddle.to_tensor([0.])

    total_norm = paddle.norm(
        paddle.stack([paddle.norm(p.grad, norm_type) for p in parameters]),
        norm_type)
    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(),
                                                total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().scale_(clip_coef_clamped)
    return total_norm


@paddle.no_grad()
def get_grad_norm_(parameters, norm_type: float=2.0):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return paddle.to_tensor([0.])
    total_norm = paddle.norm(
        paddle.stack([paddle.norm(p.grad, norm_type) for p in parameters]),
        norm_type)
    return total_norm


def save_model(args,
               epoch,
               model_without_ddp,
               optimizer,
               loss_scaler,
               exp_name=None):
    output_dir = args.output_dir
    epoch_name = str(epoch)
    if loss_scaler is not None:
        if exp_name is not None:
            checkpoint_paths = [
                output_dir + '/' + '{}_checkpoint-{}.pd'.format(exp_name,
                                                                epoch_name)
            ]
        else:
            checkpoint_paths = [
                output_dir + '/' + 'checkpoint-%s.pd' % epoch_name
            ]
        for checkpoint_path in checkpoint_paths:
            to_save_state_dict = model_without_ddp.state_dict()
            # all_keys = list(state_dict.keys())

            for key in list(to_save_state_dict.keys()):
                if key.startswith('teacher_network.'):
                    to_save_state_dict.pop(key)

            to_save = {
                'model': to_save_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        raise NotImplementedError


def auto_load_model(args,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    model_ema=None):
    output_dir = Path(args.output_dir)

    # amp
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(
            os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir,
                                       'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        checkpoint = paddle.load(args.resume)

        # handle ema model
        need_state_dict = model_without_ddp.state_dict()
        need_ema = False
        for key in need_state_dict.keys():
            if 'teacher_network' in key:
                need_ema = True
                break

        checkpoint_model = checkpoint['model']

        if need_ema:
            all_keys = list(checkpoint_model.keys())
            all_keys = [key for key in all_keys if key.startswith('encoder.')]
            for key in all_keys:
                new_key = key.replace('encoder.', 'teacher_network.')
                checkpoint_model[new_key] = checkpoint_model[key]

        model_without_ddp.set_state_dict(checkpoint_model)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.set_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            # if hasattr(args, 'model_ema') and args.model_ema:
            #     _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = paddle.to_tensor(x)
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


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


def save_np(filename, obj):
    basename, ext = os.path.splitext(filename)
    filename = basename + f'_{dist.get_rank()}' + ext
    with open(filename, 'wb') as f:
        np.save(f, obj)


def load_np(filename):
    basename, ext = os.path.splitext(filename)
    filename = basename + f'_{dist.get_rank()}' + ext
    with open(filename, 'rb') as f:
        obj = np.load(f, allow_pickle=True)
    if obj.size == 1:
        return obj.item()
    else:
        return obj
