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

import paddle
import paddle.nn as nn

from .builder import HEADS
import paddle.nn.functional as F
import paddle.fluid.layers as layers
LARGE_NUM = 1e9

@HEADS.register()
class SimCLRContrastiveHead(nn.Layer):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.5, return_accuracy=True, multi_rank=False):
        super(SimCLRContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.return_accuracy = return_accuracy
        self.multi_rank = multi_rank

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        hidden1, hidden2 = pos, neg
        batch_size = pos.shape[0]



        # Gather hidden1/hidden2 across replicas and create local labels.
        if self.multi_rank is True:
            hidden1_large = self.add_allgather(hidden1, "hidden1"+str(self.co2))
            hidden2_large = self.add_allgather(hidden2, "hidden2"+str(self.co2))
            hidden1_large = paddle.reshape(hidden1_large,
                                           [-1, hidden1_large.shape[-1]])
            hidden2_large = paddle.reshape(hidden2_large,
                                           [-1, hidden2_large.shape[-1]])
            enlarged_batch_size = paddle.shape(hidden1_large)[0]
            
            trainer_id = self.args.trainer_id
            labels_idx = paddle.arange(0, batch_size, 1,
                                      "int32") + trainer_id * batch_size
            labels = F.one_hot(
                paddle.reshape(labels_idx, [batch_size]),
                enlarged_batch_size * 2)
            masks = F.one_hot(
                paddle.reshape(labels_idx, [batch_size]),
                enlarged_batch_size)
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = F.one_hot(
                paddle.reshape(
                    paddle.arange(0, batch_size, 1, "int32"), [batch_size]),
                batch_size * 2)  
            masks = F.one_hot(
                paddle.reshape(
                    paddle.arange(0, batch_size, 1, "int32"), [batch_size]),
                batch_size)  
        
        logits_aa = paddle.matmul(
            hidden1, hidden1_large, transpose_y=True) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = paddle.matmul(
            hidden2, hidden2_large, transpose_y=True) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = paddle.matmul(
            hidden1, hidden2_large, transpose_y=True) / self.temperature
        logits_ba = paddle.matmul(
            hidden2, hidden1_large, transpose_y=True) / self.temperature
      
        loss_a = paddle.nn.functional.softmax_with_cross_entropy(
                paddle.concat([logits_ab, logits_aa], 1), labels, soft_label=True)
        loss_b = paddle.nn.functional.softmax_with_cross_entropy(
            paddle.concat([logits_ba, logits_bb], 1), labels, soft_label=True)
        contrast_loss = loss_a + loss_b

        logits_ab_co2 = logits_ab - masks * LARGE_NUM
        logits_ba_co2 = logits_ba - masks * LARGE_NUM
        logit_a = paddle.concat([logits_aa, logits_ab_co2], 1)
        logit_b = paddle.concat([logits_ba_co2, logits_bb], 1)
        log_a = paddle.nn.functional.log_softmax(logit_a)
        log_b = paddle.nn.functional.log_softmax(logit_b)
        a = paddle.nn.functional.softmax(logit_a)
        b = paddle.nn.functional.softmax(logit_b)
        kl_1 = paddle.nn.functional.kl_div(log_a, b, reduction='batchmean')
        kl_2 = paddle.nn.functional.kl_div(log_b, a, reduction='batchmean')
        co2_loss = 1 * (kl_1 + kl_2)

        total_contrast_loss = contrast_loss + 3 * co2_loss
        loss = layers.reduce_mean(total_contrast_loss)
        contrastive_label = paddle.unsqueeze(
            paddle.argmax(
                labels, axis=1), 1)
      
        acc1 = layers.accuracy(input=logits_ab, label=contrastive_label)
        outputs = dict()
        outputs['loss'] = loss
        outputs['acc1'] = acc1
    

        return outputs

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = paddle.cast(pred == target.reshape([1, -1]).expand_as(pred),
                              'float32')

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).sum(0, keepdim=True)
            res.append(correct_k * 100.0 / batch_size)
        
        return res


def add_allgather(self, hidden, name=""):
    block = self._train_program.global_block()
    hidden_large = block.create_var(
        name=name,
        shape=[self.args.trainer_num] + list(hidden.shape),
        persistable=False,
        dtype=core.VarDesc.VarType.FP32)
    op_len = len(list(enumerate(block.ops)))

    op_maker = core.op_proto_and_checker_maker
    self.op_role_key = op_maker.kOpRoleAttrName()
    block._insert_op(
        op_len,
        type='c_allgather',
        inputs={'X': hidden},
        outputs={'Out': hidden_large},
        attrs={
            'nranks': self.args.trainer_num,
            self.op_role_key: OpRole.Forward,
            "use_calc_stream": True
        })
    return hidden_large


