# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import sklearn
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


__all__ = ['TopkAcc', 'mAP']


class TopkAcc(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        if x.dtype == paddle.float16 or x.dtype == paddle.bfloat16:
            x = paddle.cast(x, 'float32')

        if len(label.shape) == 1:
            label = label.reshape([label.shape[0], -1])

        if label.dtype == paddle.int32:
            label = paddle.cast(label, 'int64')
        metric_dict = dict()
        for i, k in enumerate(self.topk):
            acc = paddle.metric.accuracy(x, label, k=k).item()
            metric_dict["top{}".format(k)] = acc
            if i == 0:
                metric_dict["metric"] = acc

        return metric_dict


class mAP(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, similarities_matrix, query_img_id, gallery_img_id,
                keep_mask):
        metric_dict = dict()

        choosen_indices = paddle.argsort(
            similarities_matrix, axis=1, descending=True)
        gallery_labels_transpose = paddle.transpose(gallery_img_id, [1, 0])
        gallery_labels_transpose = paddle.broadcast_to(
            gallery_labels_transpose,
            shape=[
                choosen_indices.shape[0], gallery_labels_transpose.shape[1]
            ])
        choosen_label = paddle.index_sample(gallery_labels_transpose,
                                            choosen_indices)
        equal_flag = paddle.equal(choosen_label, query_img_id)
        if keep_mask is not None:
            keep_mask = paddle.index_sample(
                keep_mask.astype('float32'), choosen_indices)
            equal_flag = paddle.logical_and(equal_flag,
                                            keep_mask.astype('bool'))
        equal_flag = paddle.cast(equal_flag, 'float32')

        num_rel = paddle.sum(equal_flag, axis=1)
        num_rel = paddle.greater_than(num_rel, paddle.to_tensor(0.))
        num_rel_index = paddle.nonzero(num_rel.astype("int"))
        num_rel_index = paddle.reshape(num_rel_index, [num_rel_index.shape[0]])
        equal_flag = paddle.index_select(equal_flag, num_rel_index, axis=0)

        acc_sum = paddle.cumsum(equal_flag, axis=1)
        div = paddle.arange(acc_sum.shape[1]).astype("float32") + 1
        precision = paddle.divide(acc_sum, div)

        #calc map
        precision_mask = paddle.multiply(equal_flag, precision)
        ap = paddle.sum(precision_mask, axis=1) / paddle.sum(equal_flag,
                                                             axis=1)
        metric_dict["mAP"] = paddle.mean(ap).item()
        metric_dict["metric"] = metric_dict["mAP"]

        return metric_dict
