import os
import numpy as np
from sys import flags
from collections import defaultdict

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
                    # conpact FP16 saving pretrained weight
                    if model_state_dict[k].dtype != para_state_dict[k].dtype:
                        para_state_dict[k] = para_state_dict[k].astype(model_state_dict[k].dtype)
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict), tag))
        else:
            print("No pretrained weights found in {} => training with random weights".format(tag))

    def load_pretrained(self, path, rank=0, finetune=False):
        pass

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")

    def _freeze_norm(self, layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True
        
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
    
    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model(path, self.res_model, 'backbone')

    def forward(self, inp):
        with paddle.no_grad():
            output = self.res_model(inp)
        output = self.linear(output)
        
        return output

class SwAVFinetune(SwAV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply(self._freeze_norm)
    
    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model(path, self.res_model, 'backbone') 

    def param_groups(self, config, tensor_fusion=True, epochs=None, trainset_length=None):
        """
        custom_cfg(dict|optional): [{'name': 'backbone', 'lr': 0.1, 'LRScheduler': {"lr":1.0}}, {'name': 'norm', 'weight_decay_mult': 0}]
        """

        self.custom_cfg = config.pop('custom_cfg', None)
        if self.custom_cfg is not None:
            assert isinstance(self.custom_cfg, list), "`custom_cfg` must be a list."
        
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
        params_dict = {item['name']: [] for item in self.custom_cfg} # key name and a PasslDefault
        params_dict['PasslDefault'] = []
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            for idx, item in enumerate(self.custom_cfg):
                if item['name'] in name:
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
            param_dict = {'params': params_dict[item['name']], 'lr': lr_scheduler}    

            if self.weight_decay is not None and weight_decay_mult is not None:
                param_dict['weight_decay'] = self.weight_decay * weight_decay_mult
            param_dict['tensor_fusion'] = tensor_fusion
            res.append(param_dict)
        else:
            res.append({'params': params_dict['PasslDefault'], 'tensor_fusion': tensor_fusion})

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
    def __init__(self, queue_length=0, crops_for_assign=(0, 1), nmb_crops=[2, 6], epsilon=0.05, freeze_prototypes_niters=5005, **kwargs):
        super().__init__(**kwargs)
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops
        self.temperature = 0.1
        self.epsilon = epsilon
        self.freeze_prototypes_niters = freeze_prototypes_niters

        # initialize queue
        self.queue = None
        # queue_path = os.path.join('.', "queue" + str(0) + ".pth")
        # if os.path.isfile(queue_path):
        #     self.queue = paddle.load(queue_path)["queue"]
        # # the queue needs to be divisible by the batch size
        # queue_length = queue_length
        # queue_length -= queue_length % (256)
        # if queue_length > 0 and epoch >= 15 and self.queue is None:
        #     self.queue = paddle.zeros([len(crops_for_assign),
        #             queue_length // 4, kwargs['output_dim']])
        # self.load_pretrained('swav_800ep_pretrain.pdparams') 
        self.apply(self._freeze_norm)
    
    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model('swav_800ep_pretrain.pdparams', self.res_model, 'backbone') 
        
    @paddle.no_grad()
    def distributed_sinkhorn(self, out, sinkhorn_iterations=3):
        Q = paddle.exp(x=out / self.epsilon).t()
        B = Q.shape[1] * 4
        K = Q.shape[0]
        sum_Q = paddle.sum(x=Q)
        paddle.distributed.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(sinkhorn_iterations):
            sum_of_rows = paddle.sum(x=Q, axis=1, keepdim=True)
            paddle.distributed.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            Q /= paddle.sum(x=Q, axis=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def forward(self, inp):
        # ####### test            #######
        # import numpy as np
        # np.random.seed(42)
        # a = np.random.rand(32, 3, 224, 224)
        # inp = paddle.to_tensor(a).astype('float32')
        bs = inp[0].shape[0]

        # normalize the prototypes
        with paddle.no_grad():
            w = self.res_model.prototypes.weight.clone()
            w = paddle.nn.functional.normalize(x=w, axis=0, p=2) # 1
            paddle.assign(w, self.res_model.prototypes.weight)
        embedding, output = self.res_model(inp)
        # print('output, embedding', embedding.mean(), output.mean(), inp.mean())
        # import pdb; pdb.set_trace()
        embedding = embedding.detach()

        # compute loss
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with paddle.no_grad():
                out = output[bs * crop_id:bs * (crop_id + 1)].detach()
                # print('bs, crop_id', bs, crop_id, self.nmb_crops)
                if self.queue is not None:
                    if use_the_queue or not paddle.all(x=self.queue[(i), (-1), :] == 0):
                        use_the_queue = True
                        out = paddle.concat(x=(paddle.mm(input=self.queue[i],
                            mat2=self.res_model.prototypes.weight.t()), out))
                    self.queue[(i), bs:] = self.queue[(i), :-bs].clone()
                    self.queue[(i), :bs] = embedding[crop_id * bs:(crop_id + 1) * bs]

                q = self.distributed_sinkhorn(out)[-bs:]
                # print('out.mean(), q.mean()', out.mean(), q.mean())
            
            subloss = 0
            # print(output.shape)
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v:bs * (v + 1)] / self.temperature
                subloss -= paddle.mean(x=paddle.sum(x=q * paddle.nn.
                    functional.log_softmax(x=x, axis=1), axis=1))
                # print('v, subloss', v, subloss)
                
            loss += subloss / (np.sum(self.nmb_crops) - 1)
            # print('i, loss', i, loss)
        # import pdb; pdb.set_trace()
        loss /= len(self.crops_for_assign)

        return loss
    
    def after_loss_backward(self, iteration):
        if iteration < self.freeze_prototypes_niters:
            for name, p in self.res_model.named_parameters():
                if 'prototypes' in name and p.grad is not None:
                    p.clear_grad()
        
def swav_resnet50_linearprobe(**kwargs):
    model = SwAVLinearProbe(**kwargs)
    return model

def swav_resnet50_finetune(**kwargs):
    model = SwAVFinetune(**kwargs)
    if paddle.distributed.get_world_size() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def swav_resnet50_pretrain(apex, **kwargs): # todo
    flags = {}
    flags['FLAGS_cudnn_exhaustive_search'] = True
    flags['FLAGS_cudnn_deterministic'] = False
    paddle.set_flags(flags)

    model = SwAVPretrain(**kwargs)

    if paddle.distributed.get_world_size() > 1:
        if not apex:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            # with apex syncbn speeds up computation than global syncbn
            process_group = apex.parallel.create_syncbn_process_group(8)
            model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    
    return model       
            
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