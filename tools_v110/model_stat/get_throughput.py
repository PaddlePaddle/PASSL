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

import time

import paddle
from paddle.io import DataLoader, Dataset


class TestDataset(Dataset):
    def __init__(self, num_samples, image_size):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size

    def __getitem__(self, idx):
        image = paddle.randn([1, 3, self.image_size, self.image_size])
        return image

    def __len__(self):
        return self.num_samples


def throughput(model, image_size):

    test_loader = DataLoader(TestDataset(2000, image_size), batch_size=1)

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    model.eval()
    with paddle.no_grad():
        # inference
        print("Warmup iter 5 ...")

        for i, data in enumerate(test_loader()):

            start_time = time.perf_counter()
            model(data[0])
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed

                if i % 100 == 0:
                    infer_speed = (i + 1 - num_warmup) / pure_inf_time
                    print(
                        "[{: >4}/{:}]  throughput: {:.1f} img/s ".format(
                            i, len(test_loader), infer_speed
                        )
                    )

    print(f"Overall throughput: {infer_speed:.1f} img/s")
