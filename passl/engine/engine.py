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

import os
import math
import errno
import random
import platform
import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from visualdl import LogWriter

from passl.utils.misc import SmoothedValue
from passl.utils import logger
from passl.utils.config import print_config
from passl.data import build_dataloader
from passl.models import build_model
from passl.loss import build_loss
from passl.metric import build_metrics
from passl.scheduler import build_lr_scheduler
from passl.optimizer import build_optimizer
from passl.utils import io
from passl.core import recompute_warp, GradScaler, param_sync
from passl.models.utils import EMA
from passl.utils.infohub import runtime_info_hub
from . import loops


class Engine(object):
    def __init__(self, config, mode="train"):
        assert mode in ["train", "eval", "export"]
        self.mode = mode
        self.config = config
        self.task_type = self.config["Global"].get("task_type",
                                                   "classification")
        self.use_dali = self.config['Global'].get("use_dali", False)
        self.print_batch_step = self.config['Global'].get('print_batch_step',
                                                          10)
        self.save_interval = self.config["Global"].get("save_interval", 1)
        self.accum_steps = self.config["Global"].get("accum_steps", 1)

        self.enabled_ema = True if "EMA" in self.config else False

        assert isinstance(self.accum_steps, int) and self.accum_steps > 0, \
            "accum_steps must be int dtype and greater than 0"

        # global iter counter
        self.max_train_step = self.config["Global"].get("max_train_step", None)
        assert self.max_train_step is None or (
            isinstance(self.max_train_step, int) and self.max_train_step > 0
        ), "max_train_step must be int dtype and greater than 0"

        # init distribution env
        self.config["Global"]["distributed"] = dist.get_world_size() != 1
        self.config["Global"]["rank"] = dist.get_rank()
        self.config["Global"]["world_size"] = dist.get_world_size()
        if self.config["Global"]["distributed"]:
            dist.init_parallel_env()

        # set seed
        seed = self.config["Global"].get("seed", False)
        if seed:
            assert isinstance(seed, int), "The 'seed' must be a integer!"
            seed += self.config["Global"]["rank"]
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            def worker_init_fn(worker_id):
                """ set seed in subproces for dataloader when num_workers > 0"""
                np.random.seed(seed + worker_id)
                random.seed(seed + worker_id)

        RELATED_FLAGS_SETTING = {}
        RELATED_FLAGS_SETTING['FLAGS_cudnn_exhaustive_search'] = 1
        RELATED_FLAGS_SETTING['FLAGS_cudnn_batchnorm_spatial_persistent'] = 1
        RELATED_FLAGS_SETTING['FLAGS_max_inplace_grad_add'] = 8

        related_flags_setting = self.config["Global"].get(
            "flags", RELATED_FLAGS_SETTING)
        RELATED_FLAGS_SETTING.update(related_flags_setting)
        paddle.set_flags(RELATED_FLAGS_SETTING)

        # init logger
        log_file = os.path.join(self.output_dir, self.config["Model"]["name"],
                                f"{mode}.log")
        logger.init_logger(log_file=log_file)
        print_config(config)

        # for visualdl
        self.vdl_writer = None
        if self.config['Global']['use_visualdl'] and mode == "train":
            vdl_writer_path = os.path.join(self.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                # may be more than one processes trying
                # to create the directory
                try:
                    os.makedirs(vdl_writer_path)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
                    pass
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)

        # set device
        assert self.config["Global"]["device"] in ["cpu", "gpu", "xpu", "npu"]
        self.device = paddle.set_device(self.config["Global"]["device"])
        logger.info('train with paddle {}, commit id {} and device {}'.format(
            paddle.__version__, paddle.__git_commit__[:8], self.device))

        class_num = config["Model"].get("class_num", None)
        self.config["DataLoader"].update({"class_num": class_num})
        # build dataloader
        if self.mode == 'train':
            self.train_dataloader = build_dataloader(
                self.config["DataLoader"], "Train", self.device, self.use_dali,
                worker_init_fn)
        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            self.eval_dataloader = build_dataloader(
                self.config["DataLoader"], "Eval", self.device, self.use_dali,
                worker_init_fn)

        # build loss
        self.train_loss_func = None
        self.eval_loss_func = None
        if 'Loss' in self.config:
            if self.mode == "train":
                loss_info = self.config["Loss"]["Train"]
                self.train_loss_func = build_loss(loss_info)
            if self.mode == "eval" or (self.mode == "train" and
                                       self.config["Global"]["eval_during_train"]):
                loss_config = self.config.get("Loss", None)
                if loss_config is not None:
                    loss_config = loss_config.get("Eval")
                    if loss_config is not None:
                        self.eval_loss_func = build_loss(loss_config)
                    else:
                        self.eval_loss_func = None
                else:
                    self.eval_loss_func = None

        # build metric
        self.train_metric_func = None
        if self.mode == 'train':
            metric_config = self.config.get("Metric")
            if metric_config is not None:
                metric_config = metric_config.get("Train")
                if metric_config is not None:
                    self.train_metric_func = build_metrics(metric_config)

        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            metric_config = self.config.get("Metric")
            if metric_config is not None:
                metric_config = metric_config.get("Eval")
                if metric_config is not None:
                    self.eval_metric_func = build_metrics(metric_config)
        else:
            self.eval_metric_func = None

        # FP16 training
        self.fp16 = True if "FP16" in self.config else False
        config_fp16 = self.config.get('FP16', {})
        assert config_fp16 is not None
        self.fp16_level = config_fp16.get("level", 'O2')
        assert self.fp16_level in ['O0', 'O1', 'O2']
        if self.fp16 and self.fp16_level == 'O0':
            self.fp16 = False
        self.fp16_custom_white_list = config_fp16.get("fp16_custom_white_list",
                                                      None)
        self.fp16_custom_black_list = config_fp16.get("fp16_custom_black_list",
                                                      None)

        if self.fp16 and self.fp16_level == 'O2':
            default_dtype = paddle.framework.get_default_dtype()
            paddle.set_default_dtype("float16")

        # build model
        self.model = build_model(self.config["Model"])

        n_parameters = sum(p.numel() for p in self.model.parameters()
                           if not p.stop_gradient).item()
        i = int(math.log(n_parameters, 10) // 3)
        size_unit = ['', 'K', 'M', 'B', 'T', 'Q']
        logger.info("Number of Parameters is {:.2f}{}.".format(
            n_parameters / math.pow(1000, i), size_unit[i]))

        # build grad scaler
        config_gradscaler = config_fp16.get('GradScaler', {})
        self.scaler = GradScaler(
            enable=self.fp16,
            **config_gradscaler, )

        if self.fp16 and self.fp16_level == 'O2':
            paddle.set_default_dtype(default_dtype)

        # build optimizer and lr scheduler
        assert self.config.get("Optimizer", None) is not None, "Optimizer must be defined in config."
        self.lr_decay_unit = self.config["Optimizer"].pop('lr_decay_unit', None)
        if self.lr_decay_unit is None:
            self.lr_decay_unit = 'step'
            logger.warning('lr_decay_unit is not set in optimizer config, set to step by default!')
        if self.mode == 'train':
            config_lr_scheduler = self.config.get('LRScheduler', None)
            self.lr_scheduler = None
            if config_lr_scheduler is not None:
                self.lr_decay_unit = config_lr_scheduler.get('decay_unit',
                                                             'step')
                self.lr_scheduler = build_lr_scheduler(
                    config_lr_scheduler, self.config["Global"]["epochs"],
                    len(self.train_dataloader))

            self.optimizer = build_optimizer(self.config["Optimizer"], self.lr_scheduler, self.model,
                                             self.config["Global"]["epochs"], len(self.train_dataloader),
                                             self.lr_decay_unit)

        # load pretrained model
        if self.config["Global"]["pretrained_model"] is not None:
            assert isinstance(
                self.config["Global"]["pretrained_model"], str
            ), "pretrained_model type is not available. Please use `string`."
            self.model.load_pretrained(
                self.config["Global"]["pretrained_model"],
                self.config["Global"]["rank"],
                self.config["Global"].get("finetune", False))

        # for distributed
        if self.config["Global"]["distributed"]:
            # config DistributedStrategy
            assert self.config.get("DistributedStrategy", None) is not None

            if self.config["DistributedStrategy"].get("recompute",
                                                      None) is not None:
                # insert recompute warp according relues
                recompute_warp(
                    self.model,
                    **self.config["DistributedStrategy"]['recompute'])
            if self.config["DistributedStrategy"].get("data_sharding", False):
                assert 'data_parallel' not in self.config[
                    "DistributedStrategy"], "data_parallel cannot be set when using data_sharding"
                # from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import ShardingOptimizerStage2
                # from paddle.distributed.fleet.meta_parallel.sharding.sharding_stage2 import ShardingStage2
                # from paddle.distributed.fleet.meta_parallel.sharding.sharding_utils import ShardingScaler

                # # Note(GuoxiaWang): Only support global data parallel now!
                # # First, we need to split optimizer
                # self.optimizer = ShardingOptimizerStage2(
                #     params=self.model.parameters(), optim=self.optimizer)

                # # Second, warpper the origin model to have gradient sharding function
                # self.model = ShardingStage2(
                #     self.model,
                #     self.optimizer,
                #     accumulate_grads=self.accum_steps > 1,
                #     device=self.config["Global"]["device"], )
                # self.scaler = ShardingScaler(self.scaler)
                assert False, "Do not support data_sharding now!"
            else:
                # we always use pure data parallel default
                assert 'data_parallel' in self.config["DistributedStrategy"] and \
                    self.config["DistributedStrategy"]["data_parallel"] == True, \
                    "If you want to use data parallel you should set data_parallel=True"

                param_sync(self.model)
                self.data_parallel_recompute = self.config[
                    "DistributedStrategy"].get("recompute", None) is not None

        if self.enabled_ema and self.mode == 'train':
            ema_cfg = self.config.get("EMA", {})
            self.ema_eval = ema_cfg.pop('ema_eval', False)
            self.ema_eval_start_epoch = ema_cfg.pop('eval_start_epoch', 0)
            if self.ema_eval:
                logger.info(
                    f'You have enable ema evaluation and start from {self.ema_eval_start_epoch} epoch, and it will save the best ema state.'
                )
            else:
                logger.info(
                    f'You have enable ema, and also can set ema_eval=True and eval_start_epoch to enable ema evaluation.'
                )
            self.ema = EMA(self.optimizer._param_groups, **ema_cfg)
            self.ema.register()

        # build train_loop and eval_loop
        validate_loop_name = self.config['Global'].get("validate_loop", None)
        self.validate_loop = None
        if validate_loop_name is not None:
            self.validate_loop = eval('loops.{}'.format(validate_loop_name))(self)

        train_loop_name = self.config['Global'].get("train_loop")
        self.train_loop = eval('loops.{}'.format(train_loop_name))(
            self,
            epochs=self.config["Global"]["epochs"],
            max_train_step=self.max_train_step,
            val_loop=self.validate_loop,
        )

        if self.mode == 'train':
            self.init_runtime_info_hub()

    @property
    def cur_epoch_id(self):
        return self.train_loop.cur_epoch_id

    @property
    def global_step(self):
        return self.train_loop.global_step

    @property
    def epochs(self):
        return self.train_loop.epochs

    @property
    def model_name(self):
        return self.config["Model"]["name"]

    @property
    def max_num_checkpoint(self):
        return self.config["Global"]["max_num_latest_checkpoint"]

    @property
    def output_dir(self):
        return self.config['Global']['output_dir']

    @property
    def checkpoint(self):
        return self.config["Global"]["checkpoint"]

    def init_runtime_info_hub(self):
        runtime_info_hub.epochs = self.train_loop.epochs
        runtime_info_hub.max_steps = self.train_loop.max_steps
        runtime_info_hub.total_iterations = self.train_loop.global_step

    def train(self):
        assert self.mode == "train"
        self.training = True
        self.model.train()
        self.train_loop.run()

        if self.vdl_writer is not None:
            self.vdl_writer.close()

    @paddle.no_grad()
    def eval(self):
        assert self.mode in ["train", "eval"]
        self.model.eval()
        self.validating = True
        eval_result = self.validate_loop.run()
        self.model.train()
        return eval_result

    @paddle.no_grad()
    def export(self):
        assert self.mode in ["export"]
        assert self.config["Export"] is not None
        assert self.config["Global"]["pretrained_model"] is not None
        self.model.eval()

        path = os.path.join(self.output_dir, self.config["Model"]["name"])
        io.export(self.config["Export"], self.model, path)
