#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn as nn
from .resnetcifar import ResNet
from .resnetcifar import BasicBlock, BottleneckBlock


from .builder import BACKBONES
from ...modules import init, freeze
from ...utils.logger import get_logger


@BACKBONES.register()
class ResNetsimclr(ResNet):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            resnet50 = ResNet(BottleneckBlock, 50)

            resnet18 = ResNet(BasicBlock, 18)

    """
    def __init__(self,
                 depth,
                 num_classes=0,
                 with_pool=True,
                 zero_init_residual=False,
                 frozen_stages=-1,
                 pretrained=None):

        block = BasicBlock if depth in [18, 34] else BottleneckBlock

        super(ResNetsimclr, self).__init__(block, depth, num_classes, with_pool)
        self.zero_init_residual = zero_init_residual
        self.frozen_stages = frozen_stages
        # self.init_parameters()

        if pretrained is not None:
            state_dict = paddle.load(pretrained)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            self.set_state_dict(state_dict)
            logger = get_logger()
            logger.info(
                'Load pretrained backbone weight from {} success!'.format(
                    pretrained))

   
        if self.frozen_stages >= 0:
            freeze.freeze_batchnorm_statictis(self.bn1)
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.trainable = False

        for i in range(1, self.frozen_stages + 1):
            # print('*?')
            m = getattr(self, 'layer{}'.format(i))
            freeze.freeze_batchnorm_statictis(m)
            for param in m.parameters():
                param.trainable = False

        if self.frozen_stages >= 0:
            logger = get_logger()
            logger.info(
                'Frozen layer before stage {}'.format(self.frozen_stages + 1))

