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

import time
import platform
import paddle

from passl.utils.misc import AverageMeter
from passl.utils import logger


def default_eval(engine, epoch_id=0):
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }

    metric_key = None
    tic = time.time()
    accum_samples = 0
    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)

    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        #batch[0] = paddle.to_tensor(batch[0]).astype("float32")

        # do cast if using fp16 otherwise do nothing
        with paddle.amp.auto_cast(
                enable=engine.fp16,
                custom_white_list=engine.fp16_custom_white_list,
                custom_black_list=engine.fp16_custom_black_list,
                level=engine.fp16_level):
            out = engine.model(batch[0])
            # calc loss
            if engine.eval_loss_func is not None:
                loss_dict = engine.eval_loss_func(out, batch[1])
                for key in loss_dict:
                    if key not in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')
                    output_info[key].update(float(loss_dict[key]), batch_size)

        # just for DistributedBatchSampler issue: repeat sampling
        current_samples = batch_size * paddle.distributed.get_world_size()
        accum_samples += current_samples

        # calc metric
        if engine.eval_metric_func is not None:
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

                if accum_samples > total_samples and not engine.use_dali:
                    pred = pred[:total_samples + current_samples -
                                accum_samples]
                    labels = labels[:total_samples + current_samples -
                                    accum_samples]
                    current_samples = total_samples + current_samples - accum_samples
                metric_dict = engine.eval_metric_func(pred, labels)
            else:
                metric_dict = engine.eval_metric_func(out, batch[1])
            for key in metric_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')

                output_info[key].update(metric_dict[key], current_samples)

        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % engine.print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].val)
                for key in output_info
            ])
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()

    if engine.use_dali:
        engine.eval_dataloader.reset()

    # do average
    for key in output_info:
        if isinstance(output_info[key], AverageMeter):
            output_info[key] = output_info[key].avg

    metric_msg = logger.dict_format(output_info)
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    if engine.eval_metric_func is None:
        return None

    output_info['epoch'] = epoch_id
    output_info['global_step'] = engine.global_step
    return output_info
