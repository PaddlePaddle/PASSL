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

import os
import paddle
from copy import deepcopy
from ..backbones.discrete_vae import Dalle_VAE, DiscreteVAE, load_model, Encoder, Decoder
from ...utils.registry import Registry, build_from_config

MODELS = Registry("MODEL")


def build_model(cfg):
    return build_from_config(cfg, MODELS)


def create_d_vae(cfg):
    cfg = deepcopy(cfg)
    name = cfg.pop('name')
    if name == "dall-e":
        return get_dalle_vae(cfg)
    elif name == "customized":
        return get_d_vae(cfg)
    else:
        raise NotImplementedError()


def get_dalle_vae(cfg):
    cfg = deepcopy(cfg)
    image_size = cfg.pop('image_size')
    weight_path = cfg.pop('weight_path')
    with paddle.no_grad():
        vae = Dalle_VAE(image_size)
        vae.encoder = load_model('encoder', model_dir=weight_path)
        vae.decoder = load_model('decoder', model_dir=weight_path)
        return vae


def get_d_vae(cfg):
    cfg = deepcopy(cfg)
    image_size = cfg.pop('image_size')
    weight_path = cfg.pop('weight_path')
    NUM_TOKENS = 8192
    NUM_LAYERS = 3
    EMB_DIM = 512
    HID_DIM = 256

    state_dict = paddle.load(os.path.join(weight_path, "pytorch_model.bin"),
                             map_location="cpu")["weights"]

    model = DiscreteVAE(
        image_size=image_size,
        num_layers=NUM_LAYERS,
        num_tokens=NUM_TOKENS,
        codebook_dim=EMB_DIM,
        hidden_dim=HID_DIM,
    )

    model.load_state_dict(state_dict)
    return model
