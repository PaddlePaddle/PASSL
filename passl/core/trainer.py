# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from copy import deepcopy
import os
import random

import paddle
import numpy as np

from passl.core import Config, PasslBuilder
from passl.utils import utils


class PassleTrainer():
    def __init__(self, args) -> None:
        # load configs
        self.args = args
        self.cfg = Config(self.args.config,
                    learning_rate=self.args.learning_rate,
                    epochs=self.args.epochs,
                    batch_size=self.args.batch_size,
                    opts=self.args.opts)
        
        # build registered componets
        self.builder = PasslBuilder(self.cfg)

        self.init_env()

        self.model_conversion()

        self.component_init()
    
    def init_env(self) -> None:
        utils.show_env_info()
        utils.show_cfg_info(self.cfg)
        utils.set_seed(self.args.seed)
        utils.set_device(self.args.device)
        utils.set_cv2_num_threads(self.args.num_workers)

        self.init_save_dir()
        if self.args.use_vdl:
            self.init_visualdl()
    
    def init_save_dir(self):
        if not os.path.isdir(self.args.save_dir):
            if os.path.exists(self.args.save_dir):
                os.remove(self.args.save_dir)
            os.makedirs(self.args.save_dir, exist_ok=True)
    
    def init_visualdl(self) -> None:
        from visualdl import LogWriter
        log_writer = LogWriter(self.args.save_dir)

    def model_conversion(self):
        self.model_syncbn_convert()
        if self.args.ema:
            self.ema_model = self.build_ema()
        self.resume_model()

    def model_syncbn_convert(self) -> None:
        self.model = utils.convert_sync_batchnorm(self.builder.model, self.args.device)

    def build_ema(self):
        ema_model = deepcopy(self.model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.stop_gradient = True
        return ema_model
    
    def resume_model(self):
        self.start_epoch = 0
        if self.args.resume_model is not None:
            self.start_epoch = utils.resume(self.builder.model, self.builder.optimizer, self.args.resume_model)

    def distributed_conversion(self):
        if paddle.distributed.ParallelEnv().nranks > 1:
            paddle.distributed.fleet.init(is_collective=True)
            self.builder.optimizer.optimizer = paddle.distributed.fleet.distributed_optimizer(
                self.builder.optimizer)  # The return is Fleet object
            self.builder.model = paddle.distributed.fleet.distributed_model(self.builder.model)

    def component_init(self):
        self.batch_sampler = paddle.io.DistributedBatchSampler(
            self.builder.train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.loader = paddle.io.DataLoader(
            self.builder.train_dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.args.num_workers,
            return_list=True,
            worker_init_fn=utils.worker_init_fn, )

    def train(self) -> None:
        print('You have build a PassleTrainer, please inherit this class and implement the train method for each model')
        


if __name__ == '__main__':
    trainer = PassleTrainer()
    trainer.train()
