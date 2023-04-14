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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import platform
import datetime
import paddle
from passl.utils import logger
from passl.utils.misc import SmoothedValue

class _Loop:
    """Basic Loops interface."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer

        self.output_info = dict()
        self.time_info = {
            "batch_cost": SmoothedValue(window_size=self.trainer.print_batch_step),
            "reader_cost": SmoothedValue(window_size=self.trainer.print_batch_step),
        }
        if self.trainer.enabled_ema:
            if "ema_cost" not in self.time_info:
                self.time_info['ema_cost'] = SmoothedValue(window_size=self.trainer.print_batch_step)

    def __call__(self):
        self.run()

    def run(self):
        raise NotImplementedError

    def _should_check_val(self):
        if hasattr(self, 'val_loop') and self.val_loop is None:
            return False

        epoch_mode_flag = self.trainer.config["Global"]["eval_unit"] == 'epoch' and self.trainer.config[
                "Global"]["eval_during_train"] and self.cur_epoch_id % self.trainer.config[
                    "Global"]["eval_interval"] == 0 and getattr(self, 'val_loop', None) is not None

        step_mode_flag = self.trainer.config["Global"]["eval_unit"] == 'step' and self.trainer.config[
                "Global"]["eval_during_train"] and self.global_step % self.trainer.config[
                    "Global"]["eval_interval"] == 0 and getattr(self, 'val_loop', None) is not None

        return epoch_mode_flag or step_mode_flag

    def update_metric(self, out, batch):
        if out is None:
            return

        # calc metric
        if self.trainer.train_metric_func is not None:
            metric_dict = self.trainer.train_metric_func(out, batch[-1])
            for key in metric_dict:
                if key not in self.output_info:
                    self.output_info[key] = SmoothedValue(
                        window_size=self.trainer.print_batch_step)
                self.output_info[key].update(metric_dict[key], self.batch_size)


    def update_loss(self, loss_dict):
        # update_output_info
        for key in loss_dict:
            if key not in self.output_info:
                self.output_info[key] = SmoothedValue(
                    window_size=self.trainer.print_batch_step)
            self.output_info[key].update(loss_dict[key].item(), self.batch_size)

    def log_info(self):

        lr_msg = "lr: {:.6f}".format(self.trainer.optimizer.get_lr())

        metric_msg = ", ".join([
            "{}: {:.5f}".format(key, self.output_info[key].avg)
            for key in self.output_info
        ])

        time_msg = ", ".join([
            "{}: {:.5f}".format(key, self.time_info[key].avg)
            for key in self.time_info
        ])

        total_batch_size = self.batch_size * self.trainer.config["Global"]["world_size"]
        ips_msg = "ips: {:.5f} images/sec".format(
            total_batch_size / self.time_info["batch_cost"].avg)
        eta_sec = ((self.epochs - self.cur_epoch_id + 1) * self.total_batch_idx - self.cur_batch_idx) * self.time_info["batch_cost"].avg
        eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))

        if paddle.is_compiled_with_cuda():
            GB = 1024.0 * 1024.0 * 1024.0
            max_memory_allocated = paddle.device.cuda.max_memory_allocated() / GB
            mem_msg = "max mem: {:.2f} GB".format(max_memory_allocated)
            logger.info("[Train][Epoch {}/{}][Iter: {}/{}] {}, {}, {}, {}, {}, {}".format(
                self.cur_epoch_id, self.epochs, self.cur_batch_idx, self.total_batch_idx,
                lr_msg, metric_msg, time_msg, ips_msg, mem_msg, eta_msg))
        else:
            logger.info("[Train][Epoch {}/{}][Iter: {}/{}] {}, {}, {}, {}, {}".format(
                self.cur_epoch_id, self.epochs, self.cur_batch_idx, self.total_batch_idx,
                lr_msg, metric_msg, time_msg, ips_msg, eta_msg))

        logger.scaler(
            name="lr",
            value=self.trainer.optimizer.get_lr(),
            step=self.global_step,
            writer=self.trainer.vdl_writer)
        for key in self.output_info:
            logger.scaler(
                name="train_{}".format(key),
                value=self.output_info[key].avg,
                step=self.global_step,
                writer=self.trainer.vdl_writer)


class TrainingEpochLoop(_Loop):

    def __init__(self, trainer, epochs, max_train_step=None, val_loop=None):
        super().__init__(trainer)
        self.start_eopch = 0
        self.epochs = epochs
        self.global_step = 0
        self.max_train_step = max_train_step
        self.val_loop = val_loop

    @property
    def max_steps(self) -> int:
        return self.epochs * (len(self.trainer.train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(self.trainer.train_dataloader))

    def run(self):
        assert self.trainer.mode == "train"
        assert self.trainer.training == True

        self.total_batch_idx = len(self.trainer.train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(self.trainer.train_dataloader)
        for epoch_id in range(self.start_eopch + 1, self.epochs + 1):
            self.cur_epoch_id = epoch_id

            if hasattr(self.trainer.train_dataloader.batch_sampler, "set_epoch"):
                self.trainer.train_dataloader.batch_sampler.set_epoch(epoch_id)

            # for one epoch train
            self.train_one_epoch()

            if self.trainer.lr_decay_unit == 'epoch':
                self.trainer.optimizer.lr_step(self.cur_epoch_id)

            if self.trainer.use_dali:
                self.train_dataloader.reset()

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, self.output_info[key].global_avg)
                for key in self.output_info
            ])
            logger.info("[Train][Epoch {}/{}][Avg]{}".format(
                epoch_id, self.epochs, metric_msg))
            self.output_info.clear()

            if self._should_check_val():
                self.trainer.validating = True
                self.val_loop.run()
                self.trainer.training = True

        # end of training
        self.trainer.training = False


    def train_one_epoch(self):
        self.trainer.model.train()

        tic = time.time()

        for batch_idx, batch in enumerate(self.trainer.train_dataloader):
            self.cur_batch_idx = batch_idx

            if self.max_train_step is not None and self.global_step >= self.max_train_step:
                logger.info(
                    f'global_step({self.global_step}) >= max_train_step({self.max_train_step}), training stops early.'
                )
                exit(0)

            if batch_idx >= self.total_batch_idx:
                break

            if batch_idx == 5:
                for key in self.time_info:
                    self.time_info[key].reset()

            self.time_info["reader_cost"].update(time.time() - tic)
            if self.trainer.use_dali:
                batch = [
                    paddle.to_tensor(batch[0]['data']),
                    paddle.to_tensor(batch[0]['label'])
                ]

            self.global_step += 1

            # do forward and backward
            out, loss_dict = self.train_one_step(batch)

            self.time_info["batch_cost"].update(time.time() - tic)

            # below code just for logging
            # update metric_for_logger
            self.update_metric(out, batch)
            # update_loss_for_logger
            self.update_loss(loss_dict)

            if batch_idx % self.trainer.print_batch_step == 0:
                self.log_info()

            tic = time.time()
            # ema update
            if self.trainer.enabled_ema:
                self.trainer.ema.update()
                self.time_info["ema_cost"].update(time.time() - tic)

            tic = time.time()


    def train_one_step(self, batch):
        raise NotImplementedError
