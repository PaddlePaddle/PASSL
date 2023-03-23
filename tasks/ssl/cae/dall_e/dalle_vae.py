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
import paddle
from paddle import nn
import paddle.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder


def load_model(path: str) -> nn.Layer:
    state_dict = paddle.load(path)
    if 'encoder' in path.lower():
        model = Encoder()
    elif 'decoder' in path.lower():
        model = Decoder()
    else:
        raise ValueError
    model.set_state_dict(state_dict)
    return model


class BasicVAE(nn.Layer):
    def get_codebook_indices(self, images):
        raise NotImplementedError()

    def decode(self, img_seq):
        raise NotImplementedError()

    def get_codebook_probs(self, img_seq):
        raise NotImplementedError()

    def get_image_tokens_size(self):
        pass

    def get_image_size(self):
        pass


class Dalle_VAE(BasicVAE):
    def __init__(self, image_size):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.image_size = image_size

    def load_model(self, model_dir):
        self.encoder = load_model(os.path.join(model_dir, "encoder_weight.pd"))
        self.decoder = load_model(os.path.join(model_dir, "decoder_weight.pd"))

    def decode(self, img_seq):
        bsz = img_seq.shape[0]
        img_seq = img_seq.reshape(
            [bsz, self.image_size // 8, self.image_size // 8])
        z = F.one_hot(
            img_seq, num_classes=self.encoder.vocab_size).transpose(
                [0, 3, 1, 2]).astype(paddle.float32)
        return self.decoder(z).astype(paddle.float32)

    def get_codebook_indices(self, images):
        z_logits = self.encoder(images)
        return paddle.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images):
        z_logits = self.encoder(images)
        return nn.Softmax(axis=1)(z_logits)

    def forward(self, img_seq_prob, no_process=False):
        if no_process:
            return self.decoder(img_seq_prob.astype(paddle.float32)).astype(
                paddle.float32)
        else:
            bsz, seq_len, num_class = img_seq_prob.shape
            z = img_seq_prob.reshape([
                bsz, self.image_size // 8, self.image_size // 8,
                self.encoder.vocab_size
            ])
            return self.decoder(
                z.transpose([0, 3, 1, 2]).astype(paddle.float32)).astype(
                    paddle.float32)


def create_d_vae(weight_path, d_vae_type, image_size):
    if d_vae_type == "dall-e":
        return get_dalle_vae(weight_path, image_size)
    elif d_vae_type == "customized":
        return get_d_vae(weight_path, image_size)
    elif d_vae_type == "to_tensor":
        return None
    else:
        raise NotImplementedError()


def get_dalle_vae(weight_path, image_size):
    vae = Dalle_VAE(image_size)
    vae.load_model(model_dir=weight_path)
    return vae
