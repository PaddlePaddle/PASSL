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

import os
import random
import copy
import types
import numpy as np
from itertools import product

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

_seed = None
_dp_seed = None
_hcg = None
_dp_sharding_comm_group = None
_mp_ring_comm_group = None


def set_seed(seed):
    """
    calculate and set local seed and global seed
    """

    # NOTE(shenliang03): For parameter init seed:
    # seed: dp/mp_undistributed_paramter/sharding is same; others is different
    # For compute seed(dropout):
    # global seed: only mp group is same.
    # local seed: all groups are different

    if dist.get_world_size() > 1:
        # obtain rank message of hybrid parallel
        hcg = get_hcg()

        mp_rank = hcg.get_model_parallel_rank()
        mp_size = hcg.get_model_parallel_world_size()

        pp_rank = hcg.get_stage_id()
        pp_size = hcg.get_pipe_parallel_world_size()

        dp_rank = hcg.get_data_parallel_rank()
        dp_size = hcg.get_data_parallel_world_size()

        sharding_rank = hcg.get_sharding_parallel_rank()
        sharding_size = hcg.get_sharding_parallel_world_size()

    else:
        mp_rank, mp_size = 0, 1
        pp_rank, pp_size = 0, 1
        dp_rank, dp_size = 0, 1
        sharding_rank, sharding_size = 0, 1

    seed_offset = seed + 1024 + paddle.distributed.get_world_size()
    global_seed = seed_offset + \
                  pp_rank * (mp_size) + \
                  dp_rank * (mp_size * pp_size) + \
                  sharding_rank * (mp_size * pp_size * dp_size)

    seed_offset += paddle.distributed.get_world_size()
    local_seed = seed_offset + \
                 mp_rank + \
                 pp_rank * (mp_size) + \
                 dp_rank * (mp_size * pp_size) + \
                 sharding_rank * (mp_size * pp_size * dp_size)

    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)

    paddle.seed(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)

    global _seed
    global _dp_seed
    _seed = seed
    _dp_seed = global_seed


def set_hcg(hcg):
    """
    set hcg
    """
    global _hcg
    _hcg = hcg


def get_hcg():
    """
    get hcg
    """
    global _hcg
    return _hcg


def get_seed():
    """
    get seed
    """
    global _seed
    return _seed


def get_dp_seed():
    """
    get dp seed
    """
    global _dp_seed
    return _dp_seed


def get_local_rank():
    """
    get local rank
    """
    return int(os.getenv("PADDLE_RANK_IN_NODE", 0))


def get_data_sharding_parallel_world_size():
    """
    get data sharding parallel world size
    """
    global _dp_sharding_comm_group
    return _dp_sharding_comm_group.nranks

def get_data_sharding_parallel_world_rank():
    """
    get daisp sharding parallel world rank
    """
    global _dp_sharding_comm_group
    return _dp_sharding_comm_group.rank

def get_data_sharding_parallel_group():
    """
    get data sharding parallel group
    """
    global _dp_sharding_comm_group
    return _dp_sharding_comm_group

def get_data_parallel_world_size():
    """
    get data parallel world size
    """
    hcg = get_hcg()
    return hcg.get_data_parallel_world_size()

def get_data_parallel_world_rank():
    """
    get dataparallel world rank
    """
    hcg = get_hcg()
    return hcg.get_data_parallel_rank()

def get_data_parallel_group():
    """
    get data parallel group
    """
    hcg = get_hcg()
    return hcg.get_data_parallel_group()

def get_sharding_parallel_world_size():
    """
    get sharding parallel world size
    """
    hcg = get_hcg()
    return hcg.get_sharding_parallel_world_size()

def get_sharding_parallel_world_rank():
    """
    get sharding parallel world rank
    """
    hcg = get_hcg()
    return hcg.get_sharding_parallel_rank()

def get_sharding_parallel_group():
    """
    get shaking  parallel group
    """
    hcg = get_hcg()
    return hcg.get_sharding_parallel_group()

def get_model_parallel_world_size():
    """
    get model parell  world size
    """
    if dist.get_world_size() == 1:
        return 1

    hcg = get_hcg()
    mp_size = hcg.get_model_parallel_world_size()

    return mp_size

def get_model_parallel_world_rank():
    """
    get model  parell  world rank
    """
    if dist.get_world_size() == 1:
        return 0

    hcg = get_hcg()
    mp_rank = hcg.get_model_parallel_rank()

    return mp_rank

def get_model_parallel_group():
    """
    get model  parell  group
    """
    hcg = get_hcg()
    return hcg.get_model_parallel_group()

def get_model_parallel_ring_group():
    global _mp_ring_comm_group
    return _mp_ring_comm_group


def init_dp_sharding_parallel_group():
    """
    Init the distributed data parallel sharding parallel group
    """
    hcg = get_hcg()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    degrees = [hcg._dp_degree, hcg._pp_degree, hcg._sharding_degree, hcg._mp_degree]
    group_arr = np.arange(0, world_size).reshape(degrees)

    transpose_axes = [1, 3, 0, 2]
    degree = hcg._dp_degree * hcg._sharding_degree
    arr = group_arr.transpose(transpose_axes).reshape((-1, degree))

    global _dp_sharding_comm_group
    for i in range(world_size // degree):
        ranks = arr[i].tolist()
        group = dist.new_group(ranks)
        if rank in ranks:
            _dp_sharding_comm_group = group

    # register attr and method to hcg instance
    setattr(hcg, '_dp_sharding_comm_group', _dp_sharding_comm_group)

    def get_data_sharding_parallel_group(self):
        return self._dp_sharding_comm_group

    def get_data_sharding_parallel_world_rank(self):
        return self._dp_sharding_comm_group.rank

    def get_data_sharding_parallel_world_size(self):
        return self._dp_sharding_comm_group.nranks

    hcg.get_data_sharding_parallel_group = types.MethodType(get_data_sharding_parallel_group, hcg)
    hcg.get_data_sharding_parallel_world_rank = types.MethodType(get_data_sharding_parallel_world_rank, hcg)
    hcg.get_data_sharding_parallel_world_size = types.MethodType(get_data_sharding_parallel_world_size, hcg)

def init_model_parallel_ring_group():
    global _mp_ring_comm_group
    _mp_ring_comm_group = {}

    hcg = fleet.get_hybrid_communicate_group()
    mp_group = hcg.get_model_parallel_group()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    degrees = [hcg._dp_degree, hcg._pp_degree, hcg._sharding_degree, hcg._mp_degree]
    group_arr = np.arange(0, world_size).reshape(degrees)
    degree = hcg._mp_degree
    arr = group_arr.reshape((-1, degree))

    for i in range(world_size // degree):
        ranks = arr[i].tolist()
        ranks = ranks + [ranks[0]]
        for idx in range(len(ranks)-1):
            p2p_ranks = [ranks[idx], ranks[idx+1]]
            group = dist.new_group(p2p_ranks)
            if p2p_ranks[0] in mp_group.ranks or p2p_ranks[1] in mp_group.ranks:
                src_rank = mp_group.get_group_rank(p2p_ranks[0])
                dst_rank = mp_group.get_group_rank(p2p_ranks[1])
                _mp_ring_comm_group[f'mp_{src_rank}to{dst_rank}'] = group

    # register attr and method to hcg instance
    setattr(hcg, '_mp_ring_comm_group', _mp_ring_comm_group)

    def get_model_parallel_ring_group(self):
        return self._mp_ring_comm_group

    hcg.get_model_parallel_ring_group = types.MethodType(get_model_parallel_ring_group, hcg)


def init_dist_env(seed, mp_degree=1, pp_degree=1, sharding_degree=1):
    """
    init distributed env
    """
    strategy = fleet.DistributedStrategy()
    other_degree = mp_degree * pp_degree * sharding_degree
    assert dist.get_world_size() % other_degree == 0
    dp_degree = dist.get_world_size() // other_degree
    strategy.hybrid_configs = {
        "dp_degree": dp_degree,
        "mp_degree": mp_degree,
        "pp_degree": pp_degree,
        "sharding_degree": sharding_degree
    }
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

    # init Fleet env
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    set_hcg(hcg)

    # merge DP and Sharding to a common communication group so that you can communicate data dim
    init_dp_sharding_parallel_group()
    init_model_parallel_ring_group()

    # set seed
    set_seed(seed)
