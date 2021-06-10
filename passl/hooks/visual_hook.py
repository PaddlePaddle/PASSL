# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import paddle
from .hook import Hook
from .builder import HOOKS
from visualdl import LogWriter
from visualdl.server import app
import paddle.distributed as dist
import os

@HOOKS.register()
class VisualHook(Hook):
    def __init__(self, priority=1):
        self.priority = priority
        
    def run_begin(self, trainer):
        rank = dist.get_rank()
        if rank != 0:
            return
        logdir = os.path.join(trainer.output_dir, 'visual_dl')
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.writer = LogWriter(logdir=logdir)
        # app.run(logdir=logdir, port=8040, host="0.0.0.0")

    def train_epoch_end(self, trainer):
        rank = dist.get_rank()
        if rank != 0:
            return
        outputs = trainer.outputs
        for k in outputs.keys():
            v = trainer.logs[k].avg
            self.writer.add_scalar(tag='train/{}'.format(k), step=trainer.current_epoch, value=v)
        with paddle.no_grad():
            if dist.get_world_size() > 1:
                for name, param in trainer.model._layers.named_parameters():
                    if 'bn' not in name:
                        self.writer.add_histogram(name, param.numpy(), trainer.current_epoch)
            else:
                for name, param in trainer.model.named_parameters():
                    if 'bn' not in name:
                        self.writer.add_histogram(name, param.numpy(), trainer.current_epoch)
    
    def run_end(self, trainer):
        rank = dist.get_rank()
        if rank != 0:
            return
        self.writer.close()
