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
    def __init__(self, init_eval=False, eval_kargs=None):
        if eval_kargs is None:
            self.eval_kargs = {}
        else:
            self.eval_kargs = eval_kargs

        self.init_eval = init_eval

    def run_begin(self, trainer):
        if self.init_eval:
            self._evaluate(trainer)

    def train_epoch_end(self, trainer):
        self._evaluate(trainer)

    def _evaluate(self, trainer):
        if not hasattr(trainer, 'val_dataloader'):
            from ..datasets.builder import build_dataloader
            trainer.val_dataloader = build_dataloader(
                trainer.cfg.dataloader.val)
        logger = get_logger()

        logger.info(
            'start evaluate on epoch {} ..'.format(trainer.current_epoch + 1))
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            dataloader = tqdm(trainer.val_dataloader)
        else:
            dataloader = trainer.val_dataloader

        model = trainer.model
        total_samples = len(trainer.val_dataloader.dataset)
        logger.info('total samples {}'.format(total_samples))
        accum_samples = 0

        trainer.model.eval()
        outs = OrderedDict()

        for data in dataloader:
            if isinstance(data, paddle.Tensor):
                batch_size = data.shape[0]
            elif isinstance(data, (list, tuple)):
                batch_size = data[0].shape[0]
            else:
                raise TypeError('unknown type of data')

            labels = data[-1]
            pred = model(*data, mode='test')

            current_samples = batch_size * world_size
            accum_samples += current_samples

            # for k, v in outputs.items():
            if world_size > 1:
                pred_list = []
                dist.all_gather(pred_list, pred)
                pred = paddle.concat(pred_list, 0)
                label_list = []
                dist.all_gather(label_list, labels)
                labels = paddle.concat(label_list, 0)
                if accum_samples > total_samples:
                    logger.info('total samples {} {} {}'.format(
                        total_samples, accum_samples,
                        total_samples + current_samples - accum_samples))
                    pred = pred[:total_samples + current_samples -
                                accum_samples, ...]
                    labels = labels[:total_samples + current_samples -
                                    accum_samples, ...]
                    current_samples = total_samples + current_samples - accum_samples

            res = trainer.val_dataloader.dataset.evaluate(
                pred, labels, **self.eval_kargs)

            for k, v in res.items():
                if k not in outs:
                    outs[k] = AverageMeter(k, ':6.3f')
                outs[k].update(float(v), current_samples)

        log_str = f'Validate Epoch [{trainer.current_epoch + 1}] '
        log_items = []
        for name, val in outs.items():
            if isinstance(val, AverageMeter):
                val = str(val)
            log_items.append(val)
        log_str += ', '.join(log_items)
        logger.info(log_str)

        trainer.model.train()
