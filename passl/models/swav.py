from collections import defaultdict
import copy
import os
from sys import flags

import paddle
import paddle.nn as nn

from passl.nn import init
from passl.scheduler import build_lr_scheduler, lr_scheduler
from passl.utils import logger
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

    def param_groups(self, config, tensor_fusion=True, epochs=None, trainset_length=None):
        """
        custom_cfg(dict|optional): [{'name': 'backbone', 'lr': 0.1, 'LRScheduler': {"lr":1.0}}, {'name': 'norm', 'weight_decay_mult': 0}]
        """

        self.custom_cfg = config.pop('custom_cfg', None)
        if self.custom_cfg is not None:
            assert isinstance(self.custom_cfg, list), "`custom_cfg` must be a list."
        assert self.custom_cfg['PasslDefault'].get('LRScheduler', None) is not None, 'LRScheduler is not set in group with name PasslDefault, please set them.'
            for item in self.custom_cfg:
                assert isinstance(
                    item, dict), "The item of `custom_cfg` must be a dict"
        
        param_group = self._collect_params(config, self.res_model, tensor_fusion, epochs, trainset_length)

        return param_group
    
    def _collect_params(self, config, model, tensor_fusion, epochs, trainset_length):
        # Collect different parameter groups
        if self.custom_cfg is None or len(self.custom_cfg) == 0:
            return [{'params': model.parameters(), 'tensor_fusion': tensor_fusion}]

        # split params
        self.weight_decay = config['weight_decay']
        params_dict = {item['name']: [] for item in self.custom_cfg}
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            for idx, item in enumerate(self.custom_cfg):
                if item['name'] in name and item['name']!='PasslDefault':
                    params_dict[item['name']].append(param)
                    break
            else:
                params_dict['PasslDefault'].append(param)

        res = []
        for item in self.custom_cfg:
            weight_decay_mult = item.get("weight_decay_mult", None)
            if item.get("LRScheduler", None) is not None:
                lr_scheduler = build_lr_scheduler(item['LRScheduler'], epochs, trainset_length, config['decay_unit'])
            else:
                Warning('The LRScheduler is not set for group with name {}, use default LRScheduler'.format(item['name']))
            # todo: initialize LRCallable here.
                lr_scheduler = build_lr_scheduler(self.custom_cfg['PasslDefault']['LRScheduler'], epochs, trainset_length, config['decay_unit'])
            param_dict = {'params': params_dict[item['name']], 'lr': lr_scheduler}    

            if self.weight_decay is not None and weight_decay_mult is not None:
                param_dict['weight_decay'] = self.weight_decay * weight_decay_mult
            param_dict['tensor_fusion'] = tensor_fusion
            res.append(param_dict)

        msg = 'Parameter groups for optimizer: \n'
        for idx, item in enumerate(self.custom_cfg):
            params_name = [p.name for p in params_dict[item['name']]]
            item = item.copy()
            item['params_name'] = params_name
            msg += 'Group {}: \n{} \n'.format(idx, item)
        logger.info(msg)

        return res
    
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

def swav_resnet50_pretrain(**kwargs): # todo
    flags = {}
    flags['FLAGS_cudnn_exhaustive_search'] = True
    flags['FLAGS_cudnn_deterministic'] = True
    paddle.set_flags(flags)
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