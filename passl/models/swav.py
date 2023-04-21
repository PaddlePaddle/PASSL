import os

import paddle
import paddle.nn as nn

from passl.nn import init
from passl.models.resnet import resnet50
from passl.models.base_model import Model


__all__ = [
    'swav_resnet50_finetune',
    'swav_resnet50_linearprobe',
    'swav_resnet50_pretrain',
    'SwAV',
    'SwAVLinearProbe',
    'SwAVFinetune',
    'SwAVPretrain',
]

# def model and 
class SwAV(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.res_model = resnet50(**kwargs)
    
    def _load_model(self, path, model, tag):
        if os.path.isfile(path):
            para_state_dict = paddle.load(path)
            
            # resnet
            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                            .shape):
                    print(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict), tag))
        else:
            print("No pretrained weights found in {} => training with random weights".format(tag))

    def load_pretrained(self, path, rank=0, finetune=False):
        pass
#         if not os.path.exists(path + '.pdparams'):
#             raise ValueError("Model pretrain path {} does not "
#                              "exists.".format(path))

#         state_dict = self.state_dict()
#         param_state_dict = paddle.load(path + ".pdparams")

#         # for FP16 saving pretrained weight
#         for key, value in param_state_dict.items():
#             if key in param_state_dict and key in state_dict and param_state_dict[
#                     key].dtype != state_dict[key].dtype:
#                 param_state_dict[key] = param_state_dict[key].astype(
#                     state_dict[key].dtype)

#         if not finetune:
#             self.set_dict(param_state_dict)
#         else: # load model when finetune
#             for k in ['head0.weight', 'head0.bias', 'head.weight', 'head.bias']:
#                 if k in param_state_dict:
#                     logger.info(f"Removing key {k} from pretrained checkpoint")
#                     del param_state_dict[k]

#             self.set_dict(param_state_dict)

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")
        
        
class SwAVLinearProbe(SwAV):
    def __init__(self, class_num=1000, **kwargs):
        super().__init__(**kwargs)
        self.linear = RegLog(class_num)
        self.res_model.eval()
        
        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['linear.linear.weight', 'linear.linear.bias']:
                param.stop_gradient = True

        # optimize only the linear classifier
        parameters = list(
            filter(lambda p: not p.stop_gradient, self.parameters()))
        assert len(parameters) == 2  # weight, bias
        
        self.apply(self._freeze_norm)

    def _freeze_norm(self, layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True
    
    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model(path, self.res_model, 'backbone')
        self._load_model("linear.pdparams", self.linear, 'linear')

    def forward(self, inp):
#         import numpy as np
        # import pdb; pdb.set_trace()
        
#         np.random.seed(42)
#         a = np.random.rand(32, 3, 224, 224)
#         inp = paddle.to_tensor(a).astype('float32')
        
        with paddle.no_grad():
            output = self.res_model(inp)
        output = self.linear(output)
        
        return output

class SwAVFinetune(SwAV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model(path, self.res_model, 'backbone') 
    
    def forward(self, inp):
        return self.res_model(inp)

class SwAVPretrain(SwAV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, inp):
        return self.res_model(inp)

        
def swav_resnet50_linearprobe(**kwargs):
    model = SwAVLinearProbe(**kwargs)
    return model

def swav_resnet50_finetune(**kwargs):
    model = SwAVFinetune(**kwargs)
    return model

def swav_resnet50_pretrain(**kwargs):
    model = SwAVPretrain(**kwargs)
    return model       
            
# def normal_init(param, **kwargs):
#     initializer = nn.initializer.Normal(**kwargs)
#     initializer(param, param.block)

# def constant_init(param, **kwargs):
#     initializer = nn.initializer.Constant(**kwargs)
#     initializer(param, param.block)
        
        
class RegLog(paddle.nn.Layer):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels):
        super(RegLog, self).__init__()
        s = 2048
        self.av_pool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.linear = paddle.nn.Linear(in_features=s, out_features=num_labels)
        
        init.normal_(self.linear.weight, mean=0.0, std=0.01)
        init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.reshape((x.shape[0], -1))
        return self.linear(x)