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

import copy

import paddle

from passl.utils import logger
from passl.data import dataset
from passl.data import sampler
from passl.data import utils


def build_dataloader(config, mode, device, use_dali=False,
                     worker_init_fn=None, hybrid_parallel=False):
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Dataset mode should be Train, Eval, Test"

    # build dataset
    class_num = config.get("class_num", None)
    config_dataset = config[mode]['dataset']
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop('name')
    config_transform = config_dataset.pop('transform', None)
    if config_transform is not None:
        config_dataset.transform = utils.create_preprocess_operators(
            config_transform)
    # backward compatibility
    config_transform_ops = config_dataset.pop('transform_ops', None)
    if config_transform_ops is not None:
        config_dataset.transform_ops = utils.create_preprocess_operators(
            config_transform_ops)

    config_batch_transform = config_dataset.pop('batch_transform', None)
    batch_transform = utils.create_preprocess_operators(config_batch_transform)
    # backward compatibility
    config_batch_transform_ops = config_dataset.pop('batch_transform_ops',
                                                    None)
    if config_batch_transform_ops is not None:
        batch_transform = utils.create_preprocess_operators(
            config_batch_transform_ops)
    dataset = eval("dataset.{}".format(dataset_name))(**config_dataset)
    logger.debug("build dataset({}) success...".format(dataset))

    # build sampler
    config_sampler = config[mode]['sampler']
    config_sampler = copy.deepcopy(config_sampler)
    sampler_name = config_sampler.pop("name")

    if hybrid_parallel:
        from paddle.distributed import fleet
        hcg = fleet.get_hybrid_communicate_group()
        data_ranks = hcg.get_data_sharding_parallel_world_size()
        data_rank = hcg.get_data_sharding_parallel_world_rank()
        print(data_ranks, data_rank)
        config_sampler.update({'num_replicas': data_ranks, 'rank': data_rank})

    batch_sampler = eval("sampler.{}".format(sampler_name))(dataset,
                                                            **config_sampler)
    logger.debug("build batch_sampler({}) success...".format(batch_sampler))

    # build dataloader
    config_loader = config[mode]['loader']
    num_workers = config_loader["num_workers"]
    use_shared_memory = config_loader["use_shared_memory"]
    collate_fn_name = config_loader.get('collate_fn', 'default_collate_fn')
    collate_fn = getattr(utils, collate_fn_name)(
        batch_transform=batch_transform)

    data_loader = paddle.io.DataLoader(
        dataset=dataset,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)

    logger.debug("build data_loader({}) success...".format(data_loader))
    return data_loader
