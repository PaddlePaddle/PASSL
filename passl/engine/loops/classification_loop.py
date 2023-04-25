# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import sys
import time
import collections
import platform
import paddle
from passl.core import grad_sync, param_sync
from passl.utils import io

from passl.utils.misc import AverageMeter
from passl.utils import profiler
from passl.utils import logger
from .loop import _Loop, TrainingEpochLoop


import os
import logging
import time
from datetime import timedelta
import pandas as pd


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = "%s-%i" % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


def init_logger(name):
    logger = create_logger(
        os.path.join("{}.log".format(name)), rank=0
    )
    logger.info("============ Initialized logger ============")
    logger.info("")
    return logger


def log_model(model, logger):
    model1 = model.res_model
    for name, param in model1.named_parameters():
        logger.info(name)
        logger.info(param.abs().mean())

        
class ClassificationTrainingEpochLoop(TrainingEpochLoop):

    def __init__(self, trainer, epochs, max_train_step=None, val_loop=None):
        super().__init__(trainer, epochs, max_train_step=max_train_step, val_loop=val_loop)

    def forward_backward(self, batch):
        # Gradient Merge(GuoxiaWang): Accumulate gradient over multiple
        # steps to save on memory.

        self.batch_size = batch[0].shape[0]
        assert self.batch_size % self.trainer.accum_steps == 0, f'Bad accum_steps {self.trainer.accum_steps} for batch size {self.batch_size}. This may be caused by two reasons: 1) the batch size setting is unreasonable and cannot be divisible, 2) drop_last in the sampler configuration is not set to True.'
        step_size = self.batch_size // self.trainer.accum_steps

        final_loss_dict = collections.defaultdict(float)
        final_out = []

        for idx in range(self.trainer.accum_steps):
            data = batch[0][idx * step_size:(idx + 1) * step_size]
            label = batch[1][idx * step_size:(idx + 1) * step_size]
            
            ####### test            #######
            # label = paddle.to_tensor([133, 141, 371, 254,  89, 244,  33,  64, 542,  93, 262, 674, 898, 796, 785, 727, 228, 792, 853, 639, 410, 357, 545, 473, 637, 400, 863, 386, 689, 359, 476, 960]).cast('int32')
            # import numpy as np
            # np.random.seed(42)
            # a = np.random.rand(32, 3, 224, 224)
            # data = paddle.to_tensor(a).astype('float32')
            
            # do cast if using fp16 otherwise do nothing
            with paddle.amp.auto_cast(
                    enable=self.trainer.fp16,
                    custom_white_list=self.trainer.fp16_custom_white_list,
                    custom_black_list=self.trainer.fp16_custom_black_list,
                    level=self.trainer.fp16_level):

                out = self.trainer.model(data)
                final_out.append(out)
                            
            loss_dict = self.trainer.train_loss_func(out, label)
            # import pdb; pdb.set_trace()

            ####### test            #######
            # logger1 = init_logger('before')
            # log_model(self.trainer.model, logger1)

            for key in loss_dict:
                loss_dict[key] = loss_dict[key] / self.trainer.accum_steps

                with paddle.no_grad():
                    final_loss_dict[key] += loss_dict[key]

            # loss scaling if using fp16 otherwise do nothing
            scaled = self.trainer.scaler.scale(loss_dict["loss"])
            scaled.backward()
            
            ####### test            #######
#             grad_sync(self.trainer.optimizer.param_groups)

#             # do unscale and step if using fp16 and not found nan/inf
#             # otherwise do nothing
#             self.trainer.scaler.step(self.trainer.optimizer)
#             # do update loss scaling if using fp16
#             # otherwise do nothing
#             self.trainer.scaler.update()
            
            # logger2 = init_logger('after')
            # log_model(self.trainer.model, logger2)
            

        out = paddle.concat(final_out, axis=0)
        return out, final_loss_dict

    def train_one_step(self, batch):

        # do forward and backward
        out, loss_dict = self.forward_backward(batch)

        grad_sync(self.trainer.optimizer.param_groups)

        # do unscale and step if using fp16 and not found nan/inf
        # otherwise do nothing
        self.trainer.scaler.step(self.trainer.optimizer)
        # do update loss scaling if using fp16
        # otherwise do nothing
        self.trainer.scaler.update()
        # clear gradients
        self.trainer.optimizer.clear_grad()
        
        if self.trainer.lr_decay_unit == 'step': # default is step
            self.trainer.optimizer.lr_step(self.global_step)

        return out, loss_dict


class ClassificationEvaluationLoop(_Loop):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.best_model_metric = None
        self.best_model_to_save = False
        self.latest_model_metric = None

        self.best_ema_model_metric = None
        self.best_ema_model_to_save = False
        self.latest_ema_model_metric = None

    def reset_state(self):
        self.best_model_to_save = False
        self.best_ema_model_to_save = False

    def update_best_model_metric_info(self):
        assert isinstance(self.latest_model_metric, dict)
        if 'metric' in self.latest_model_metric and (not self.best_model_metric or self.latest_model_metric['metric'] > self.best_model_metric['metric']):
            self.best_model_metric = deepcopy(self.latest_model_metric)
            self.best_model_to_save = True

    def update_best_ema_model_metric_info(self):
        assert isinstance(self.latest_ema_model_metric, dict)
        if 'metric' in self.latest_ema_model_metric and (not self.best_ema_model_metric or self.latest_ema_model_metric['metric'] > self.best_ema_model_metric['metric']):
            self.best_ema_model_metric = deepcopy(self.latest_ema_model_metric)
            self.best_ema_model_to_save = True

    def run(self):
        assert self.trainer.mode in ["train", "eval"]
        assert self.trainer.validating == True
        self.reset_state()

        self.latest_model_metric = self.eval_one_dataset(self.trainer.eval_dataloader)
        self.update_best_model_metric_info()

        if self.trainer.enabled_ema and self.trainer.ema_eval and self.trainer.cur_epoch_id > self.trainer.ema_eval_start_epoch:
            self.trainer.ema.apply_shadow()
            self.latest_ema_model_metric = self.eval_one_dataset(self.trainer.eval_dataloader)
            self.trainer.ema.restore()

            self.update_best_ema_model_metric_info()

        self.validating = False

    @paddle.no_grad()
    def eval_one_dataset(self, eval_dataloader):
        self.trainer.model.eval()

        output_info = dict()
        metric_key = None
        tic = time.time()
        accum_samples = 0
        total_samples = len(eval_dataloader.dataset) if not self.trainer.use_dali else eval_dataloader.size
        self.total_batch_idx = len(eval_dataloader) - 1 if platform.system(
        ) == "Windows" else len(eval_dataloader)

        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= self.total_batch_idx:
                break
            if batch_idx == 5:
                for key in self.time_info:
                    self.time_info[key].reset()
            if self.trainer.use_dali:
                batch = [
                    paddle.to_tensor(batch[0]['data']),
                    paddle.to_tensor(batch[0]['label'])
                ]
            self.time_info["reader_cost"].update(time.time() - tic)
            batch_size = batch[0].shape[0]

            # do cast if using fp16 otherwise do nothing
            with paddle.amp.auto_cast(
                    enable=self.trainer.fp16,
                    custom_white_list=self.trainer.fp16_custom_white_list,
                    custom_black_list=self.trainer.fp16_custom_black_list,
                    level=self.trainer.fp16_level):
                
                ####### test            #######
                # label = paddle.to_tensor([133, 141, 371, 254,  89, 244,  33,  64, 542,  93, 262, 674, 898, 796, 785, 727, 228, 792, 853, 639, 410, 357, 545, 473, 637, 400, 863, 386, 689, 359, 476, 960, 133, 141, 371, 254,  89, 244,  33,  64, 542,  93, 262, 674, 898, 796, 785, 727, 228, 792, 853, 639, 410, 357, 545, 473, 637, 400, 863, 386, 689, 359, 476, 960]).cast('int32')
                # import numpy as np
                # np.random.seed(42)
                # a = np.random.rand(32, 3, 224, 224)
                # data = paddle.to_tensor(a).astype('float32')
                
                # import pdb; pdb.set_trace()
                # out = self.trainer.model(data)
                out = self.trainer.model(batch[0])
                # calc loss
                if self.trainer.eval_loss_func is not None:
                    # loss_dict = self.trainer.eval_loss_func(out, target)
                    loss_dict = self.trainer.eval_loss_func(out, batch[1])
                    for key in loss_dict:
                        if key not in output_info:
                            output_info[key] = AverageMeter(key, '7.5f')
                        output_info[key].update(float(loss_dict[key]), batch_size)

            # just for DistributedBatchSampler issue: repeat sampling
            current_samples = batch_size * paddle.distributed.get_world_size()
            accum_samples += current_samples

            # calc metric
            if self.trainer.eval_metric_func is not None:
                if paddle.distributed.get_world_size() > 1:
                    label_list = []
                    paddle.distributed.all_gather(label_list, batch[1])
                    labels = paddle.concat(label_list, 0)

                    if isinstance(out, dict):
                        out = out["logits"]
                    if isinstance(out, list):
                        pred = []
                        for x in out:
                            pred_list = []
                            paddle.distributed.all_gather(pred_list, x)
                            pred_x = paddle.concat(pred_list, 0)
                            pred.append(pred_x)
                    else:
                        pred_list = []
                        paddle.distributed.all_gather(pred_list, out)
                        pred = paddle.concat(pred_list, 0)

                    if accum_samples > total_samples and not self.trainer.use_dali:
                        pred = pred[:total_samples + current_samples -
                                    accum_samples]
                        labels = labels[:total_samples + current_samples -
                                        accum_samples]
                        current_samples = total_samples + current_samples - accum_samples
                    metric_dict = self.trainer.eval_metric_func(pred, labels)
                else:
                    metric_dict = self.trainer.eval_metric_func(out, batch[1])
                for key in metric_dict:
                    if key not in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')

                    output_info[key].update(metric_dict[key], current_samples)

            self.time_info["batch_cost"].update(time.time() - tic)

            if batch_idx % self.trainer.print_batch_step == 0:
                time_msg = "s, ".join([
                    "{}: {:.5f}".format(key, self.time_info[key].avg)
                    for key in self.time_info
                ])

                ips_msg = "ips: {:.5f} images/sec".format(
                    batch_size / self.time_info["batch_cost"].avg)

                metric_msg = ", ".join([
                    "{}: {:.5f}".format(key, output_info[key].val)
                    for key in output_info
                ])
                logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                    self.trainer.cur_epoch_id, batch_idx,
                    len(eval_dataloader), metric_msg, time_msg, ips_msg))

            tic = time.time()

        if self.trainer.use_dali:
            self.trainer.eval_dataloader.reset()

        # do average
        for key in output_info:
            if isinstance(output_info[key], AverageMeter):
                output_info[key] = output_info[key].avg

        metric_msg = logger.dict_format(output_info)
        logger.info("[Eval][Epoch {}][Avg]{}".format(self.trainer.cur_epoch_id, metric_msg))

        if self.trainer.eval_metric_func is None:
            return None

        return output_info
