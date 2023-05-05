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

# Ref: https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

import copy
from collections import defaultdict
from functools import partial

import paddle
import paddle.nn as nn

from passl.models.base_model import Model
from passl.nn import init

from .resnet import ResNet, BottleneckBlock

__all__ = [
    'simsiam_resnet50_pretrain',
    'simsiam_resnet50_linearprobe',
    'SimSiamPretain',
    'SimSiamLinearProbe',
]

class SimSiamPretain(Model):
    """
    Build a SimSiam Pretrain model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiamPretain, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(class_num=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias_attr=False),
                                        nn.BatchNorm1D(prev_dim),
                                        nn.ReLU(), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias_attr=False),
                                        nn.BatchNorm1D(prev_dim),
                                        nn.ReLU(), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1D(dim, weight_attr=False, bias_attr=False)) # output layer
        self.encoder.fc[6].bias.stop_gradient = True # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias_attr=False),
                                        nn.BatchNorm1D(pred_dim),
                                        nn.ReLU(), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.criterion = nn.CosineSimilarity(axis=1)

    def forward(self, inputs):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        assert isinstance(inputs, list)
        x1 = inputs[0]
        x2 = inputs[1]

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5

        return loss

    def load_pretrained(self, path, rank=0, finetune=False):
        if not os.path.exists(path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        state_dict = self.state_dict()
        param_state_dict = paddle.load(path + ".pdparams")

        # for FP16 saving pretrained weight
        for key, value in param_state_dict.items():
            if key in param_state_dict and key in state_dict and param_state_dict[
                    key].dtype != state_dict[key].dtype:
                param_state_dict[key] = param_state_dict[key].astype(
                    state_dict[key].dtype)

        self.set_dict(param_state_dict)

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")

        # rename pre-trained keys
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('encoder') and not k.startswith('encoder.fc'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        paddle.save(state_dict, path + "_encoder.pdparams")


class SimSiamLinearProbe(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.stop_gradient = True

        # optimize only the linear classifier
        parameters = list(
            filter(lambda p: not p.stop_gradient, self.parameters()))
        assert len(parameters) == 2  # weight, bias

        init.normal_(self.fc.weight, mean=0.0, std=0.01)
        init.zeros_(self.fc.bias)

        self.apply(self._freeze_norm)

    def _freeze_norm(self, layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True

def simsiam_resnet50_pretrain(**kwargs):
    encoder = partial(ResNet, block=BottleneckBlock, depth=50)
    model = SimSiamPretain(
        base_encoder=encoder,
        dim=2048,
        pred_dim=512,
        **kwargs)

    # Apply SyncBN
    if paddle.distributed.get_world_size() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def simsiam_resnet50_linearprobe(**kwargs):
    model = SimSiamLinearProbe(block=BottleneckBlock, depth=50, **kwargs)
    return model
