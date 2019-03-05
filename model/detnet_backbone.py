from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from model.utils.config import cfg
#from model.fpn.fpn import _FPN

import torch
import torch.nn as nn
import math

__all__ = ['DetNet', 'detnet59']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckA, self).__init__()
        assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckB(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckB, self).__init__()
        assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.extra_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.extra_conv(x)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DetNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(DetNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_new_layer(256, layers[3])
        self.layer5 = self._make_new_layer(256, layers[4])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_new_layer(self, planes, blocks):
        downsample = None
        block_b = BottleneckB
        block_a = BottleneckA

        layers = []
        layers.append(block_b(self.inplanes, planes, stride=1, downsample=downsample))
        self.inplanes = planes * block_b.expansion
        for i in range(1, blocks):
            layers.append(block_a(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def load_pretrained_imagenet_weights(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if ('layer4' in name) or ('layer5' in name) or ('fc' in name):
            continue
        if (name in own_state):
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))


def detnet59(pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DetNet(Bottleneck, [3, 4, 6, 3, 3])
    if pretrained:
        print ("loading pretrained detnet")
        path = './weights/detnet59.pth'
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

    return model


'''
class detnet(_FPN):
    def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/detnet59.pth'
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.dout_base_model = 256

        _FPN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        detnet = detnet59()

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            detnet.load_state_dict({k: v for k, v in state_dict.items() if k in detnet.state_dict()})

        self.RCNN_layer0 = nn.Sequential(detnet.conv1, detnet.bn1, detnet.relu, detnet.maxpool)
        self.RCNN_layer1 = nn.Sequential(detnet.layer1)
        self.RCNN_layer2 = nn.Sequential(detnet.layer2)
        self.RCNN_layer3 = nn.Sequential(detnet.layer3)
        self.RCNN_layer4 = nn.Sequential(detnet.layer4)
        self.RCNN_layer5 = nn.Sequential(detnet.layer5)  # add one layer, for c6

        # Top layer
        self.RCNN_toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # reduce channel, for p6

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # for p3
        self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # for p2
        # self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # for c5
        self.RCNN_latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # for c4
        self.RCNN_latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # for c3
        self.RCNN_latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # for c2

        self.RCNN_top = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)

        # Fix blocks
        for p in self.RCNN_layer0[0].parameters(): p.requires_grad = False
        for p in self.RCNN_layer0[1].parameters(): p.requires_grad = False

        assert (0 <= cfg.DETNET.FIXED_BLOCKS < 4)
        if cfg.DETNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_layer3.parameters(): p.requires_grad = False
        if cfg.DETNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_layer2.parameters(): p.requires_grad = False
        if cfg.DETNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_layer1.parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.RCNN_layer0.apply(set_bn_fix)
        self.RCNN_layer1.apply(set_bn_fix)
        self.RCNN_layer2.apply(set_bn_fix)
        self.RCNN_layer3.apply(set_bn_fix)
        self.RCNN_layer4.apply(set_bn_fix)
        self.RCNN_layer5.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_layer0.eval()
            self.RCNN_layer1.eval()
            self.RCNN_layer2.train()
            self.RCNN_layer3.train()
            self.RCNN_layer4.train()
            self.RCNN_layer5.train()

            self.RCNN_smooth1.train()
            self.RCNN_smooth2.train()

            self.RCNN_latlayer1.train()
            self.RCNN_latlayer2.train()
            self.RCNN_latlayer3.train()
            self.RCNN_latlayer4.train()

            self.RCNN_toplayer.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_layer0.apply(set_bn_eval)
            self.RCNN_layer1.apply(set_bn_eval)
            self.RCNN_layer2.apply(set_bn_eval)
            self.RCNN_layer3.apply(set_bn_eval)
            self.RCNN_layer4.apply(set_bn_eval)
            self.RCNN_layer5.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        block5 = self.RCNN_top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7
'''
