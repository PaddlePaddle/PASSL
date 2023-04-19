import os
import sys
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '../..'))
sys.path.append(os.path.join(LOCAL_PATH, '.'))
sys.path.append(os.path.join(LOCAL_PATH, '../'))

import paddle
import paddle.nn as nn

from passl.nn import init
from passl.models.resnet import resnet50
from passl.models.base_model import Model


# from resnet import resnet50
# from base_model import Model

__all__ = [
    # 'swav_resnet50',
    'swav_resnet50_linearprobe',
    # 'swav_resnet50_pretrain',
    'SwAV',
    'SwAVLinearProbe',
    # 'SwAVPretrain',
]

# def model and 
class SwAV(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.res_model = resnet50(**kwargs)
    
        
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
    def __init__(self, class_num=1000, linear_arch="resnet50", global_avg=True, use_bn=False, **kwargs):
        super().__init__(**kwargs)
        self.linear = RegLog(1000, "resnet50", global_avg=True, use_bn=False)
        self.res_model.eval()
    
    def load_pretrained(self, path, rank=0, finetune=False):
        # only load res_model
        if os.path.isfile(path):
            para_state_dict = paddle.load(path)
            
            # resnet
            model_state_dict = self.res_model.state_dict()
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
            self.res_model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict), "backbone"))
        else:
            print("No pretrained weights found => training with random weights")
        
    def forward(self, inp):
        # import numpy as np
        # np.random.seed(42)
        # a = np.random.rand(32, 3, 224, 224)
        # inp = paddle.to_tensor(a).astype('float32')
        
        with paddle.no_grad():
            output = self.res_model(inp)
        output = self.linear(output)
        
        return output

        
def swav_resnet50_linearprobe(**kwargs):
    model = SwAVLinearProbe(linear_arch="resnet50", 
                            global_avg=True, 
                            use_bn=False,
                            output_dim=0, 
                            eval_mode=True,
                            **kwargs)
    return model
        
            
# def normal_init(param, **kwargs):
#     initializer = nn.initializer.Normal(**kwargs)
#     initializer(param, param.block)

# def constant_init(param, **kwargs):
#     initializer = nn.initializer.Constant(**kwargs)
#     initializer(param, param.block)
        
class RegLog(paddle.nn.Layer):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch='resnet50', global_avg=False,
        use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == 'resnet50':
                s = 2048
            elif arch == 'resnet50w2':
                s = 4096
            elif arch == 'resnet50w4':
                s = 8192
            self.av_pool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))
        else:
            assert arch == 'resnet50'
            s = 8192
            self.av_pool = paddle.nn.AvgPool2D(6, stride=1)
            if use_bn:
                self.bn = paddle.nn.BatchNorm2D(num_features=2048, momentum
                    =1 - 0.1, epsilon=1e-05, weight_attr=None, bias_attr=
                    None, use_global_stats=True)
        
        self.linear = paddle.nn.Linear(in_features=s, out_features=num_labels)
        # normal_init(self.linear.weight, mean=0.0, std=0.01) # init is correct?
        # constant_init(self.linear.bias, value=0.0) # padiff
        init.normal_(self.linear.weight, mean=0.0, std=0.01)
        init.zeros_(self.linear.bias)


    def forward(self, x):
        x = self.av_pool(x)
        if self.bn is not None:
            x = self.bn(x)

        x = x.reshape((x.shape[0], -1))
        return self.linear(x)

if __name__ == "__main__":
    import torch
    
    class RegLog_torch(torch.nn.Module):
        """Creates logistic regression on top of frozen features"""

        def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
            super(RegLog_torch, self).__init__()
            self.bn = None
            if global_avg:
                if arch == "resnet50":
                    s = 2048
                elif arch == "resnet50w2":
                    s = 4096
                elif arch == "resnet50w4":
                    s = 8192
                self.av_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            else:
                assert arch == "resnet50"
                s = 8192
                self.av_pool = torch.nn.AvgPool2d(6, stride=1)
                if use_bn:
                    self.bn = torch.nn.BatchNorm2d(2048)
            self.linear = torch.nn.Linear(s, num_labels)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()

        def forward(self, inp2):
            # average pool the final feature map
            x = self.av_pool(inp2)

            # optional BN
            if self.bn is not None:
                x = self.bn(x)

            # flatten
            x = x.view(x.size(0), -1)

            # linear layer
            return self.linear(x)
    
    layer = RegLog(1000, 'resnet50', True, False)
    module = RegLog_torch(1000, 'resnet50', True, False)
    
    
    inp = paddle.rand((4, 2048, 7, 7)).numpy().astype("float32")
    inp = ({'x': paddle.to_tensor(inp)}, 
     {'inp2': torch.as_tensor(inp) })
    
    import padiff
    from padiff import auto_diff
    auto_diff(layer, module, inp, auto_weights=True, options={'atol': 1e-4, 'rtol':0, 'compare_mode': 'strict', 'single_step':False, 'diff_phase': 'both'})