import paddle.nn as nn
import paddle.nn.functional as F

from .builder import HEADS


@HEADS.register()
class CLIPHead(nn.Layer):
    def __init__(self):
        super(CLIPHead, self).__init__()
        self.criterion= nn.CrossEntropyLoss()
    
    def forward(self, img_logits, text_logits, img_labels, text_labels):
        outputs = dict()
        img_loss = self.criterion(img_logits, img_labels)
        text_loss = self.criterion(text_logits, text_labels)
        loss = img_loss + text_loss
        outputs['img_loss'] = img_loss
        outputs['text_loss'] = text_loss
        outputs['loss'] = loss 
        return outputs
