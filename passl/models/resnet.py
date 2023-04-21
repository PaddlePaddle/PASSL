import functools

import paddle
import paddle.nn as nn

from passl.models.base_model import Model

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=3, stride=stride, padding=dilation, groups=groups,
        dilation=dilation, bias_attr=False, )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = paddle.nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = paddle.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(paddle.nn.Layer):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = paddle.nn.BatchNorm2D
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = paddle.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def kaiming_normal_init(param, **kwargs):
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)

def constant_init(param, **kwargs):
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)
    
    
class ResNet(paddle.nn.Layer):
    def __init__(self, block, layers, zero_init_residual=False, groups=1,
        widen=1, width_per_group=64, replace_stride_with_dilation=None,
        norm_layer=None, normalize=False, output_dim=0, hidden_mlp=0,
        nmb_prototypes=0, eval_mode=False):
        
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = functools.partial(paddle.nn.BatchNorm2D, use_global_stats=True)
        self._norm_layer = norm_layer
        self.eval_mode = eval_mode
        self.padding = paddle.nn.Pad2D(padding=1, value=0.0)
        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        num_out_filters = width_per_group * widen
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=
            num_out_filters, kernel_size=7, stride=2, padding=2, bias_attr=
            False)
        self.bn1 = norm_layer(num_out_filters)
        self.relu = paddle.nn.ReLU()
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(block, num_out_filters, layers[1],
            stride=2, dilate=replace_stride_with_dilation[0])
        num_out_filters *= 2
        self.layer3 = self._make_layer(block, num_out_filters, layers[2],
            stride=2, dilate=replace_stride_with_dilation[1])
        num_out_filters *= 2
        self.layer4 = self._make_layer(block, num_out_filters, layers[3],
            stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.l2norm = normalize
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = paddle.nn.Linear(in_features=
                num_out_filters * block.expansion, out_features=output_dim)
        else:
            self.projection_head = paddle.nn.Sequential(paddle.nn.Linear(
                in_features=num_out_filters * block.expansion, out_features
                =hidden_mlp), paddle.nn.BatchNorm1D(num_features=hidden_mlp,
                momentum=1 - 0.1, epsilon=1e-05, weight_attr=None,
                bias_attr=None, use_global_stats=True), paddle.nn.ReLU(),
                paddle.nn.Linear(in_features=hidden_mlp, out_features=
                output_dim))
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = paddle.nn.Linear(in_features=output_dim,
                out_features=nmb_prototypes, bias_attr=False)
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2D):
                    kaiming_normal_init(sublayer.weight) # todo mode='fan_out',
                elif isinstance(sublayer, (nn.BatchNorm2D, nn.GroupNorm)):
                    constant_init(sublayer.weight, value=1.0)
                    constant_init(sublayer.bias, value=0.0)

        if zero_init_residual:
            for sublayer in self.sublayers():
                if isinstance(m, Bottleneck):
                    param_init.constant_init(sublayer.bn3.weight, value=0.0)
                elif isinstance(m, BasicBlock):
                    param_init.constant_init(sublayer.bn2.weight, value=0.0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = paddle.nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))
        return paddle.nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.eval_mode:
            return x
        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = paddle.nn.functional.normalize(x=x, axis=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = paddle.cumsum(x=paddle.unique_consecutive(x=paddle.
            to_tensor(data=[inp.shape[-1] for inp in inputs]),
            return_counts=True)[1], axis=0) # padiff
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(paddle.concat(x=inputs[start_idx:
                end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = paddle.concat(x=(output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class MultiPrototypes(paddle.nn.Layer):

    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module('prototypes' + str(i), paddle.nn.Linear(
                in_features=output_dim, out_features=k, bias_attr=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50w2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def resnet50w4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


def resnet50w5(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)
