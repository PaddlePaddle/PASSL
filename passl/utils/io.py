# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import paddle
import time
import glob

from passl.utils import logger

__all__ = ['load_checkpoint', 'save_checkpoint']


def _mkdir_if_not_exist(path):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def _remove_if_exist(path):
    try:
        if os.path.exists(path):
            # logger.info("Remove checkpoint {}.".format(path))
            os.remove(path)
    except OSError as e:
        pass


def load_checkpoint(checkpoint_path, net, optimizer, loss_scaler):
    """
    load model from checkpoint
    """

    rank = paddle.distributed.get_rank()

    assert isinstance(
        checkpoint_path,
        str), "checkpoint type is not available. Please use `string`."

    # load model
    net.load_pretrained(checkpoint_path, rank=rank, finetune=False)

    # load optimizer state
    opt_path = checkpoint_path + '.pdopt'
    assert os.path.exists(opt_path), \
        "Optimizer checkpoint path {} does not exists.".format(opt_path)
    opt_dict = paddle.load(opt_path)

    scaler_dict = opt_dict.pop('scaler_state', {})

    dist_opt_path = checkpoint_path + "_rank{}.pdopt".format(rank)
    if os.path.exists(dist_opt_path):
        dist_opt_dict = paddle.load(dist_opt_path)
        opt_dict['state'].update(dist_opt_dict['state'])
        opt_dict['param_groups'].extend(dist_opt_dict['param_groups'])

        # clear
        dist_opt_dict['state'].clear()
        dist_opt_dict['param_groups'].clear()

    optimizer.set_state_dict(opt_dict)

    # load loss scaler
    if len(scaler_dict) > 0 and loss_scaler is not None:
        loss_scaler.load_state_dict(scaler_dict)

    # load metric state
    metric_path = checkpoint_path + '.pdstates'
    metric_dict = None
    if os.path.exists(metric_path):
        metric_dict = paddle.load(metric_path)

    logger.info("Finish load checkpoint from {}".format(checkpoint_path))
    return metric_dict


def _optimizer_state_dict_split(state_dict):
    dist_state_dict = {'state': {}, 'param_groups': []}
    for group in state_dict['param_groups']:
        if group.get('is_distributed', False):
            dist_state_dict['param_groups'].append(group)
            state_dict['param_groups'].remove(group)
            for name in group['params']:
                if name in state_dict['state']:
                    dist_state_dict['state'][name] = state_dict['state'].pop(
                        name)
    return state_dict, dist_state_dict


def save_checkpoint(net,
                    optimizer,
                    loss_scaler,
                    metric_info,
                    model_path,
                    model_name="",
                    prefix='passl',
                    max_num_checkpoint=3):
    """
    save model to the target path
    """

    local_rank = paddle.distributed.ParallelEnv().dev_id
    rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()

    metric_info.update({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S',
                                   time.localtime(time.time()))
    })
    model_dir = os.path.join(model_path, model_name)
    _mkdir_if_not_exist(model_dir)
    model_prefix = os.path.join(model_dir, prefix)
    net.save(model_prefix, local_rank, rank)

    opt_state_dict = optimizer.state_dict()

    opt_state_dict, dist_opt_state_dict = _optimizer_state_dict_split(
        opt_state_dict)

    if local_rank == 0:
        if loss_scaler is not None:
            opt_state_dict['scaler_state'] = loss_scaler.state_dict()
        paddle.save(opt_state_dict, model_prefix + ".pdopt")
        paddle.save(metric_info, model_prefix + ".pdstates")
    if len(dist_opt_state_dict['state']) > 0:
        paddle.save(dist_opt_state_dict,
                    model_prefix + "_rank{}.pdopt".format(rank))

    logger.info("Already save {} model in {}".format(prefix, model_dir))

    keep_prefixs = ['best', 'latest']

    if local_rank == 0:
        if all(p not in prefix
               for p in keep_prefixs) and max_num_checkpoint >= 0:
            pdstates_list = glob.glob(os.path.join(model_dir, '*.pdstates'))

            timestamp_to_path = {}
            for path in pdstates_list:
                if any(p in path for p in keep_prefixs):
                    continue
                metric_dict = paddle.load(path)
                timestamp_to_path[metric_dict['timestamp']] = path[:-9]

            # sort by ascend
            timestamps = list(timestamp_to_path.keys())
            timestamps.sort()

            if max_num_checkpoint > 0:
                to_remove = timestamps[:-max_num_checkpoint]
            else:
                to_remove = timestamps
            for timestamp in to_remove:
                model_prefix = timestamp_to_path[timestamp]
                for ext in ['.pdparams', '.pdopt', '.pdstates']:
                    path = model_prefix + ext
                    _remove_if_exist(path)

                    if ext in ['.pdparams', '.pdopt']:
                        for rank_id in range(world_size):
                            path = model_prefix + "_rank{}".format(
                                rank_id) + ext
                            _remove_if_exist(path)


def export(config, net, path):
    assert config["export_type"] in ['paddle', 'onnx']
    input_shape = [None if e == 'None' else e for e in config["input_shape"]]

    if config["export_type"] == 'onnx':
        paddle.onnx.export(
            net,
            path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=input_shape, dtype='float32')
            ])
    else:
        paddle.jit.save(
            net,
            path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=input_shape, dtype='float32')
            ])
    logger.info("Export model to '{}'.".format(path))


def load_ema_checkpoint(checkpoint_path, ema):
    """
    load ema state from checkpoint
    """
    # load ema state
    ema_path = checkpoint_path + '.pdema'
    assert os.path.exists(ema_path), \
        "EMA checkpoint path {} does not exists.".format(ema_path)
    ema_dict = paddle.load(ema_path)
    ema.set_state_dict(ema_dict)

    # load ema metric state
    ema_metric_path = checkpoint_path + '.pdemastates'
    ema_metric_dict = None
    if os.path.exists(ema_metric_path):
        ema_metric_dict = paddle.load(ema_metric_path)

    logger.info("Finish load ema state from {}".format(checkpoint_path))
    return ema_metric_dict


def save_ema_checkpoint(model,
                        ema,
                        model_path,
                        metric_info=None,
                        model_name="",
                        prefix='passl',
                        max_num_checkpoint=3):
    """
    save ema state to the target path
    """
    local_rank = paddle.distributed.ParallelEnv().dev_id
    rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()

    if metric_info is None:
        metric_info = {}

    metric_info.update({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S',
                                   time.localtime(time.time()))
    })
    model_dir = os.path.join(model_path, model_name)
    _mkdir_if_not_exist(model_dir)
    model_prefix = os.path.join(model_dir, prefix)

    model.save(model_prefix, local_rank, rank)

    ema_state_dict = ema.state_dict()

    local_rank = paddle.distributed.ParallelEnv().dev_id
    if local_rank == 0:
        paddle.save(ema_state_dict, model_prefix + ".pdema")
        paddle.save(metric_info, model_prefix + ".pdemastates")

    logger.info("Already save {} ema state in {}".format(prefix, model_dir))

    keep_prefixs = ['best', 'latest']

    if local_rank == 0:
        if all(p not in prefix
               for p in keep_prefixs) and max_num_checkpoint >= 0:
            pdstates_list = glob.glob(os.path.join(model_dir, '*.pdemastates'))

            timestamp_to_path = {}
            for path in pdstates_list:
                if any(p in path for p in keep_prefixs):
                    continue
                metric_dict = paddle.load(path)
                timestamp_to_path[metric_dict['timestamp']] = path[:-9]

            # sort by ascend
            timestamps = list(timestamp_to_path.keys())
            timestamps.sort()

            if max_num_checkpoint > 0:
                to_remove = timestamps[:-max_num_checkpoint]
            else:
                to_remove = timestamps
            for timestamp in to_remove:
                model_prefix = timestamp_to_path[timestamp]
                for ext in ['.pdema', '.pdemastates']:
                    path = model_prefix + ext
                    _remove_if_exist(path)
