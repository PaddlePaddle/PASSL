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

from tqdm import tqdm
from collections import OrderedDict
import paddle
import paddle.distributed as dist
from .hook import Hook
from .builder import HOOKS
from ..utils.logger import get_logger
from ..utils.misc import AverageMeter


@HOOKS.register()
class EvaluateHook(Hook):
    def __init__(self, init_eval=False, eval_kargs=None, priority=1):
        if eval_kargs is None:
            self.eval_kargs = {}
        else:
            self.eval_kargs = eval_kargs

        self.init_eval = init_eval
        self.priority = priority

    def run_begin(self, trainer):
        if self.init_eval:
            trainer.val(**self.eval_kargs)

    def train_epoch_end(self, trainer):
        trainer.val(**self.eval_kargs)