# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

from ..hooks import build_hook, Hook
from ..utils.misc import AverageMeter
from ..datasets.builder import build_dataloader
from ..modeling.architectures import build_model
from ..solver import build_lr_scheduler, build_lr_scheduler_simclr, build_optimizer


def set_hyrbid_parallel_seed(basic_seed,
                             dp_rank,
                             mp_rank,
                             pp_rank,
                             device="cuda"):
    if not basic_seed:
        return
    assert device != "cpu"
    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + dp_rank
    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)


class IterLoader:
    def __init__(self, dataloader, epoch=0):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class Trainer:
    r"""
    # trainer calling logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               build_lr_scheduler                         ||    lr_scheduler
    #                     |                                    ||
    #               build_optimizer                            ||    optimizers
    #                     |                                    ||
    #               build_train_hooks                          ||    train hooks
    #                     |                                    ||
    #               build_custom_hooks                         ||    custom hooks
    #                     |                                    ||
    #                 train loop                               ||    train loop
    #                     |                                    ||
    #      hook(print log, checkpoint, evaluate, ajust lr)     ||    call hook
    #                     |                                    ||
    #                    end                                   \/
    """

    def __init__(self, cfg):
        # base config
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.output_dir = cfg.output_dir

        dp_rank = dist.get_rank()
        self.log_interval = cfg.log_config.interval

        # set seed
        seed = cfg.get('seed', False)
        if seed:
            seed += dp_rank
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # set device
        assert cfg['device'] in ['cpu', 'gpu', 'xpu', 'npu']
        self.device = paddle.set_device(cfg['device'])
        self.logger.info('train with paddle {} on {} device'.format(
            paddle.__version__, self.device))

        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.inner_iter = 0
        self.batch_id = 0
        self.global_steps = 0
        use_byol_iters = cfg.get('use_byol_iters', False)
        self.use_byol_iters = use_byol_iters
        use_simclr_iters = cfg.get('use_simclr_iters', False)
        self.use_simclr_iters = use_simclr_iters
        self.epochs = cfg.get('epochs', None)
        self.timestamp = cfg.timestamp
        self.logs = OrderedDict()
        # Ensure that the vdl log file can be closed normally

        # build model
        self.model = build_model(cfg.model)

        n_parameters = sum(p.numel() for p in self.model.parameters()
                           if not p.stop_gradient).item()

        i = int(math.log(n_parameters, 10) // 3)
        size_unit = ['', 'K', 'M', 'B', 'T', 'Q']
        self.logger.info("Number of Parameters is {:.2f}{}.".format(
            n_parameters / math.pow(1000, i), size_unit[i]))

        # build train dataloader
        self.train_dataloader, self.mixup_fn = build_dataloader(
            cfg.dataloader.train, self.device)
        self.iters_per_epoch = len(self.train_dataloader)

        # use byol iters
        if self.use_byol_iters:
            self.global_batch_size = cfg.global_batch_size
            self.byol_total_iters = self.epochs * cfg.total_images // self.global_batch_size

        if self.use_byol_iters:
            self.lr_scheduler = build_lr_scheduler(cfg.lr_scheduler,
                                                   self.byol_total_iters)
        elif self.use_simclr_iters:
            self.batch_size = cfg.dataloader.train.sampler.batch_size
            self.global_batch_size = cfg.global_batch_size
            self.epochs = cfg.epochs
            self.lr_scheduler = build_lr_scheduler_simclr(
                cfg.lr_scheduler, self.iters_per_epoch, self.batch_size * 8,
                cfg.epochs, self.current_iter)
        else:
            self.lr_scheduler = build_lr_scheduler(cfg.lr_scheduler,
                                                   self.iters_per_epoch)

        self.optimizer = build_optimizer(cfg.optimizer, self.lr_scheduler,
                                         [self.model])

        # distributed settings
        if dist.get_world_size() > 1:
            strategy = fleet.DistributedStrategy()
            ## Hybrid Parallel Training
            strategy.hybrid_configs = cfg.pop(
                'hybrid') if 'hybrid' in cfg else {}
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()
            mp_rank = hcg.get_model_parallel_rank()
            pp_rank = hcg.get_stage_id()
            dp_rank = hcg.get_data_parallel_rank()
            set_hyrbid_parallel_seed(
                seed, 0, mp_rank, pp_rank, device=self.device)

        # amp training
        self.use_amp = cfg.get('use_amp',
                               False)  #if 'use_amp' in cfg else False
        if self.use_amp:
            amp_cfg = cfg.pop('AMP')
            self.auto_cast = amp_cfg.pop('auto_cast')
            scale_loss = amp_cfg.pop('scale_loss')
            self.scaler = paddle.amp.GradScaler(init_loss_scaling=scale_loss)
            amp_cfg['models'] = self.model
            self.model = paddle.amp.decorate(**amp_cfg)  # decorate for level O2

        # ZeRO
        self.sharding_strategies = cfg.get('sharding', False)
        if self.sharding_strategies:
            from paddle.distributed.fleet.meta_parallel.sharding.sharding_utils import ShardingScaler
            from paddle.distributed.fleet.meta_parallel.sharding.sharding_stage2 import ShardingStage2
            from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import ShardingOptimizerStage2
            self.sharding_stage = self.sharding_strategies['sharding_stage']
            accumulate_grad = self.sharding_strategies['accumulate_grad']
            offload = self.sharding_strategies['offload']
            if self.sharding_stage == 2:
                self.optimizer = ShardingOptimizerStage2(
                    params=self.model.parameters(),
                    optim=self.optimizer,
                    offload=offload)
                self.model = ShardingStage2(
                    self.model,
                    self.optimizer,
                    accumulate_grads=accumulate_grad)
                self.scaler = ShardingScaler(self.scaler)
            else:
                raise NotImplementedError()
        # data parallel
        elif dist.get_world_size() > 1:
            self.model = fleet.distributed_model(self.model)

        # build hooks
        self.hooks = []

        self.add_train_hooks()
        self.add_custom_hooks()
        self.hooks = sorted(self.hooks, key=lambda x: x.priority)

        if self.epochs:
            self.total_iters = self.epochs * self.iters_per_epoch
            self.by_epoch = True
        else:
            self.by_epoch = False
            self.total_iters = cfg.total_iters

    def add_train_hooks(self):
        optim_cfg = self.cfg.get('optimizer_config', None)
        if optim_cfg is not None:
            self.add_hook(build_hook(optim_cfg))
        else:
            self.add_hook(build_hook({'name': 'OptimizerHook'}))

        timer_cfg = self.cfg.get('timer_config', None)
        if timer_cfg is not None:
            self.add_hook(build_hook(timer_cfg))
        else:
            self.add_hook(build_hook({'name': 'IterTimerHook'}))
        ckpt_cfg = self.cfg.get('checkpoint', None)
        if ckpt_cfg is not None:
            self.add_hook(build_hook(ckpt_cfg))
        else:
            self.add_hook(build_hook({'name': 'CheckpointHook'}))

        log_cfg = self.cfg.get('log_config', None)
        if log_cfg is not None:
            self.add_hook(build_hook(log_cfg))
        else:
            self.add_hook(build_hook({'name': 'LogHook'}))

        lr_cfg = self.cfg.get('lr_config', None)
        if lr_cfg is not None:
            self.add_hook(build_hook(lr_cfg))
        else:
            self.add_hook(build_hook({'name': 'LRSchedulerHook'}))

    def add_custom_hooks(self):
        custom_cfgs = self.cfg.get('custom_config', None)
        if custom_cfgs is None:
            return

        for custom_cfg in custom_cfgs:
            cfg_ = custom_cfg.copy()
            insert_index = cfg_.pop('insert_index', None)
            self.add_hook(build_hook(cfg_), insert_index)

    def add_hook(self, hook, insert_index=None):
        assert isinstance(hook, Hook)

        if insert_index is None:
            self.hooks.append(hook)
        elif isinstance(insert_index, int):
            self.hooks.insert(insert_index, hook)

    def call_hook(self, fn_name):
        for hook in self.hooks:
            getattr(hook, fn_name)(self)

    def train(self):
        self.mode = 'train'
        self.model.train()
        iter_loader = IterLoader(self.train_dataloader, self.current_epoch)
        self.call_hook('run_begin')

        while self.current_iter < (self.total_iters):
            if self.current_iter % self.iters_per_epoch == 0:
                self.call_hook('train_epoch_begin')
            self.inner_iter = self.current_iter % self.iters_per_epoch
            self.current_iter += 1
            self.current_epoch = iter_loader.epoch

            data = next(iter_loader)

            self.call_hook('train_iter_begin')

            if self.use_amp:
                with paddle.amp.auto_cast(**self.auto_cast):
                    if self.use_byol_iters:
                        self.outputs = self.model(
                            *data,
                            total_iters=self.byol_total_iters,
                            current_iter=self.current_iter,
                            mixup_fn=self.mixup_fn)
                    else:
                        self.outputs = self.model(
                            *data,
                            total_iters=self.total_iters,
                            current_iter=self.current_iter,
                            mixup_fn=self.mixup_fn)
            else:
                if self.use_byol_iters:
                    self.outputs = self.model(
                        *data,
                        total_iters=self.byol_total_iters,
                        current_iter=self.current_iter,
                        mixup_fn=self.mixup_fn)
                else:
                    self.outputs = self.model(
                        *data,
                        total_iters=self.total_iters,
                        current_iter=self.current_iter,
                        mixup_fn=self.mixup_fn)
            self.call_hook('train_iter_end')

            if self.current_iter % self.iters_per_epoch == 0:
                self.call_hook('train_epoch_end')
                self.current_epoch += 1

        self.call_hook('run_end')

    def val(self, **kargs):
        if not hasattr(self, 'val_dataloader'):
            self.val_dataloader, mixup_fn = build_dataloader(
                self.cfg.dataloader.val, self.device)

        self.logger.info('start evaluate on epoch {} ..'.format(
            self.current_epoch + 1))
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        model = self.model
        total_samples = len(self.val_dataloader.dataset)
        self.logger.info('Evaluate total samples {}'.format(total_samples))

        if rank == 0:
            dataloader = tqdm(self.val_dataloader)
        else:
            dataloader = self.val_dataloader

        accum_samples = 0

        self.model.eval()
        outs = OrderedDict()

        for data in dataloader:
            if isinstance(data, paddle.Tensor):
                batch_size = data.shape[0]
            elif isinstance(data, (list, tuple)):
                batch_size = data[0].shape[0]
            else:
                raise TypeError('unknown type of data')

            labels = data[-1]
            if self.use_amp:
                with paddle.amp.auto_cast(**self.auto_cast):
                    pred = model(*data, mode='test')
            else:
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
                    self.logger.info('total samples {} {} {}'.format(
                        total_samples, accum_samples, total_samples +
                        current_samples - accum_samples))
                    pred = pred[:total_samples + current_samples -
                                accum_samples]
                    labels = labels[:total_samples + current_samples -
                                    accum_samples]
                    current_samples = total_samples + current_samples - accum_samples

            res = self.val_dataloader.dataset.evaluate(pred, labels, **kargs)

            for k, v in res.items():
                if k not in outs:
                    outs[k] = AverageMeter(k, ':6.3f')
                outs[k].update(float(v), current_samples)

        log_str = f'Validate Epoch [{self.current_epoch + 1}] '
        log_items = []
        for name, val in outs.items():
            if isinstance(val, AverageMeter):
                string = '{} ({' + outs[k].fmt + '})'
                val = string.format(val.name, val.avg)

            log_items.append(val)
        log_str += ', '.join(log_items)
        self.logger.info(log_str)

        self.model.train()

    def resume(self, checkpoint_path):
        checkpoint = paddle.load(checkpoint_path)
        if checkpoint.get('epoch', None) is not None:
            self.start_epoch = checkpoint['epoch']
            self.current_epoch = checkpoint['epoch']
            self.current_iter = (self.start_epoch - 1) * self.iters_per_epoch

        self.model.set_state_dict(checkpoint['state_dict'])
        self.optimizer.set_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.set_state_dict(checkpoint['lr_scheduler'])

        self.logger.info('Resume training from {} success!'.format(
            checkpoint_path))

    def load(self, weight_path, export=False):
        state_dict = paddle.load(weight_path)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        if export:
            state_dict_ = dict()
            for k, v in state_dict.items():
                state_dict_['model.backbone.' + k] = v
            state_dict = state_dict_
        self.model.set_state_dict(state_dict)

    def export(self, ckpt):
        self.model.eval()
        self.model = ExportModel(self.cfg, self.model)
        self.load(ckpt, export=True)

        save_path = os.path.join(self.cfg.save_inference_dir, "inference")
        if self.model.quanter:
            model.quanter.save_quantized_model(
                model.base_model,
                save_path,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + self.cfg["image_shape"], dtype='float32')
                ])
        else:
            model = paddle.jit.to_static(
                self.model,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + self.cfg["image_shape"], dtype='float32')
                ])
            paddle.jit.save(model, save_path)


class ExportModel(nn.Layer):
    """
    ExportModel: add softmax onto the model
    """

    def __init__(self, cfg, model):
        super().__init__()
        self.model = model
        self.pruner = None
        self.quanter = None
        if cfg.get("infer_add_softmax", True):
            self.out_act = nn.Softmax(axis=-1)
        else:
            self.out_act = None

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()

    def forward(self, x):
        x = self.model(x, mode='infer')
        if isinstance(x, list):
            x = x[0]
        if self.out_act is not None:
            x = self.out_act(x)
        return x
