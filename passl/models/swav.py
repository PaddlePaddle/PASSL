import paddle
import paddle.nn as nn

from passl.models.resnet import resnet50
from passl.models.base_model import Model


__all__ = [
    'swav_resnet50',
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
    def __init__(self, num_classes=1000, linear_arch="resnet50", global_avg=True, use_bn=False, **kwargs):
        super().__init__(**kwargs)
        self.linear = RegLog(1000, "resnet50", global_avg=True, use_bn=False)
        self.res_model.eval()
        self.criterion = nn.CrossEntropyLoss()
    
    def load_pretrained(self, path):
        # only load res_model
        model = path + ".pdparams"
        if os.path.isfile(path):
            state_dict = paddle.load(path)

            # remove prefixe "module."
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            for k, v in model.state_dict().items():
                if k not in list(state_dict):
                    logger.info('key "{}" could not be found in provided state dict'.format(k))
                elif state_dict[k].shape != v.shape:
                    logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                    state_dict[k] = v
            msg = self.res_model.set_dict(state_dict, strict=False)
            logger.info("Load pretrained model with msg: {}".format(msg))
        else:
            logger.info("No pretrained weights found => training with random weights")
        
    def forward()
        with paddle.no_grad():
            output = self.res_model(inp)
        output = reglog(output)
        
        return output

        
def swav_resnet50_linearprobe(**kwargs):
    model = SwAVLinearProbe(num_classes=1000, 
                            linear_arch="resnet50", 
                            global_avg=True, 
                            use_bn=False,
                            output_dim=0, 
                            eval_mode=True,
                            **kwargs)
    return model
        
            

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
        x = self.linear.weight.data
        paddle.assign(paddle.normal(mean=0.0, std=0.01, shape=x.shape).
            astype(x.dtype), x)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.av_pool(x)
        if self.bn is not None:
            x = self.bn(x)

        x = x.view((x.shape[0], -1))
        return self.linear(x)