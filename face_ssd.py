from __future__ import division, print_function

import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

from data import widerface_640
from layers import *
from model.detnet_backbone import *

#import pretrainedmodels

cfg = widerface_640

mo = cfg['max_out']
fpn = cfg['feature_pyramid_network']
fem = cfg['feature_enhance_module']
mio = cfg['max_in_out']
pa = cfg['progressive_anchor']
backbone = cfg['backbone']
bup = cfg['bottom_up_path']
refine = cfg['refinedet']

assert(not mo or not mio)

class FEM(nn.Module):
    def __init__(self, channel_size):
        super(FEM , self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d( self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d( self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d( 256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d( 256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d( 128, 128, kernel_size=3, dilation=1, stride=1, padding=1)
    def forward(self, x):
        x1_1 = F.relu(self.cpm1(x), inplace=True)
        x1_2 = F.relu(self.cpm2(x), inplace=True)
        x2_1 = F.relu(self.cpm3(x1_2), inplace=True)
        x2_2 = F.relu(self.cpm4(x1_2), inplace=True)
        x3_1 = F.relu(self.cpm5(x2_2), inplace=True)
        return torch.cat([x1_1, x2_1, x3_1] , 1)

class SSD(nn.Module):
    '''Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be 'test' or 'train'
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: 'multibox head' consists of loc and conf conv layers
    '''

    def __init__(self, phase, size, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        assert(num_classes == 2)
        self.cfg = cfg
        self.size = size

        # backbone
        if backbone == 'vgg':
            self.vgg = nn.ModuleList( vgg(cfg['base'],3) )
            self.extras = nn.ModuleList( add_extras(cfg['extras'], 1024) )
            self.L2Norm_3_3 = L2Norm(256, cfg['l2norm_scale'][0])
            self.L2Norm_4_3 = L2Norm(512, cfg['l2norm_scale'][1])
            self.L2Norm_5_3 = L2Norm(512, cfg['l2norm_scale'][2])
        elif backbone == 'detnet':
            detnet = detnet59(pretrained=True)
            self.layer1 = nn.Sequential(detnet.conv1, detnet.bn1, detnet.relu, detnet.maxpool, detnet.layer1)
            self.layer2 = nn.Sequential(detnet.layer2)
            self.layer3 = nn.Sequential(detnet.layer3)
            self.layer4 = nn.Sequential(detnet.layer4)
            self.layer5 = nn.Sequential(detnet.layer5)  # add one layer, for c6s
            self.layer6 = nn.Sequential(
               *[nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3,padding=1,stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)]
            )
        elif backbone in ['resnet50' , 'resnet101' , 'resnet152' , 'senet'] :
             print('loading pretrained resnet model')
             if backbone == 'resnet101':
                 resnet = torchvision.models.resnet101(pretrained=True)
             elif backbone == 'resnet50':
                 resnet = torchvision.models.resnet50(pretrained=True)
             elif backbone == 'resnet152':
                 resnet = torchvision.models.resnet152(pretrained=True)
             elif backbone == 'senet':
                 resnet = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained='imagenet')
                 #resnet = pretrainedmodels.__dict__['se_resnet101'](pretrained='imagenet')
             if backbone =='senet':
                 self.layer1 = nn.Sequential(resnet.layer0,resnet.layer1)
             else:
                 self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
             self.layer2 = nn.Sequential(resnet.layer2)
             self.layer3 = nn.Sequential(resnet.layer3)
             self.layer4 = nn.Sequential(resnet.layer4)
             self.layer5 = nn.Sequential(                                      
               *[nn.Conv2d(2048, 512, kernel_size=1),                         
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512,512, kernel_size=3,padding=1,stride=2),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True)]
             )
             self.layer6 = nn.Sequential(
               *[nn.Conv2d(512, 128, kernel_size=1,),
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, 256, kernel_size=3,padding=1,stride=2),
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True)]
             )
        if backbone == 'vgg':
            output_channels = [256, 512, 512, 1024, 512, 256 ]
        elif backbone == 'detnet':
            output_channels = [256, 512, 1024, 1024, 1024, 512]
        elif backbone in ['senet' , 'resnet50' , 'resnet101' , 'resnet152']:
            output_channels = [256, 512, 1024, 2048, 512, 256]

        if fpn:    
            fpn_in = output_channels

            #self.latlayer6 = nn.AdaptiveAvgPool2d((1,1))
            #self.latlayer5 = nn.Conv2d( fpn_in[5], fpn_in[4], kernel_size=1, stride=1, padding=0)
            #self.latlayer4 = nn.Conv2d( fpn_in[4], fpn_in[3], kernel_size=1, stride=1, padding=0)
            self.latlayer3 = nn.Conv2d( fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d( fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
            self.latlayer1 = nn.Conv2d( fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)

            #self.smooth6 = nn.Conv2d( fpn_in[5], fpn_in[5], kernel_size=1, stride=1, padding=0)
            #self.smooth5 = nn.Conv2d( fpn_in[4], fpn_in[4], kernel_size=1, stride=1, padding=0)
            #self.smooth4 = nn.Conv2d( fpn_in[3], fpn_in[3], kernel_size=1, stride=1, padding=0)
            self.smooth3 = nn.Conv2d( fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
            self.smooth2 = nn.Conv2d( fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
            self.smooth1 = nn.Conv2d( fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

            #self.fpn_layer = nn.Sequential(*[self.latlayer0, self.latlayer1, self.latlayer2, self.latlayer3, self.latlayer4, self.latlayer5])
        if bup:
            bup_in = output_channels
            #self.bup1 = nn.Conv2d(bup_in[0], bup_in[1], kernel_size=3, stride=2, padding=1)
            #self.bup2 = nn.Conv2d(bup_in[1], bup_in[2], kernel_size=3, stride=2, padding=1)
            self.bup3 = nn.Conv2d(bup_in[2], bup_in[3], kernel_size=3, stride=2, padding=1)
            self.bup4 = nn.Conv2d(bup_in[3], bup_in[4], kernel_size=3, stride=2, padding=1)
            self.bup5 = nn.Conv2d(bup_in[4], bup_in[5], kernel_size=3, stride=2, padding=1)
        if fem:
            cpm_in = output_channels
            #self.cpm3_3 = nn.Conv2d(cpm_in[0], 512, kernel_size=1)
            self.cpm3_3 = FEM(cpm_in[0])
            self.cpm4_3 = FEM(cpm_in[1])
            self.cpm5_3 = FEM(cpm_in[2])
            self.cpm7 = FEM(cpm_in[3])
            self.cpm6_2 = FEM(cpm_in[4])
            self.cpm7_2 = FEM(cpm_in[5])
            #self.cpm_layer = nn.Sequential( *[self.cpm3_3, self.cpm4_3, self.cpm5_3, self.cpm7, self.cpm6_2, self.cpm7_2] )
            
        if pa:
            head = pa_multibox(output_channels, cfg['mbox'], num_classes)  
        else:
            head = multibox(output_channels, cfg['mbox'], num_classes)  
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if refine:
            arm_head = arm_multibox(output_channels , cfg['mbox'], num_classes)
            self.arm_loc = nn.ModuleList(arm_head[0])
            self.arm_conf = nn.ModuleList(arm_head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, cfg['num_thresh'], cfg['conf_thresh'], cfg['nms_thresh'])
    
    def init_priors(self ,cfg , min_size=cfg['min_sizes'], max_size=cfg['max_sizes']):
        priorbox = PriorBox(cfg , min_size, max_size)
        prior = Variable( priorbox.forward() , volatile=True)
        return prior
        
    def forward(self, x):
        '''Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        '''
        image_size = [x.shape[2] , x.shape[3]]
        loc = list()
        conf = list()
        
        if backbone == 'vgg':
            for k in range(16):
                x  = self.vgg[k](x)
            conv3_3_x = x
            for k in range(16 , 23):
                x = self.vgg[k](x)
            conv4_3_x = x
            for k in range(23 , 30):
                x = self.vgg[k](x)
            conv5_3_x = x
            for k in range(30, len(self.vgg)):
                x = self.vgg[k](x)
            fc7_x = x
            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if k == 1:
                    conv6_2_x = x
                if k == 3 :
                    conv7_2_x = x

        elif backbone in ['senet','resnet50', 'detnet','resnet101','resnet152' , 'resnext']:
            conv3_3_x = self.layer1(x)
            conv4_3_x = self.layer2(conv3_3_x)
            conv5_3_x = self.layer3(conv4_3_x)
            fc7_x = self.layer4(conv5_3_x)
            conv6_2_x = self.layer5(fc7_x)
            conv7_2_x = self.layer6(conv6_2_x)

        if refine:   
            arm_loc = list()
            arm_conf = list()
            arm_sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]
            for (feat, l, c) in zip(arm_sources, self.arm_loc, self.arm_conf):
                arm_loc.append( l(feat).permute(0, 2, 3, 1).contiguous() )    
                arm_conf.append( c(feat).permute(0, 2, 3, 1).contiguous() )
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
              
        if fpn:
            #lfpn6 = self._upsample_product( self.latlayer6(conv7_2_x) , self.smooth6(conv7_2_x))
            #lfpn5 = self._upsample_product( self.latlayer5(lfpn6) , self.smooth5(conv6_2_x))
            #lfpn4 = self._upsample_product( self.latlayer4(lfpn5) , self.smooth4(fc7_x) )
            #lfpn3 = self._upsample_product( self.latlayer3(lfpn4) , self.smooth3(conv5_3_x) )

            lfpn3 = self._upsample_product( self.latlayer3(fc7_x) , self.smooth3(conv5_3_x) )
            lfpn2 = self._upsample_product( self.latlayer2(lfpn3) , self.smooth2(conv4_3_x) )
            lfpn1 = self._upsample_product( self.latlayer1(lfpn2) , self.smooth1(conv3_3_x) )

            #conv7_2_x = lfpn6
            #conv6_2_x = lfpn5
            #fc7_x     = lfpn4

            conv5_3_x = lfpn3
            conv4_3_x = lfpn2
            conv3_3_x = lfpn1

        if backbone == 'vgg':
            conv3_3_x = self.L2Norm_3_3(conv3_3_x)
            conv4_3_x = self.L2Norm_4_3(conv4_3_x)
            conv5_3_x = self.L2Norm_5_3(conv5_3_x)
 
        if bup:
            #conv4_3_x = F.relu(self.bup1(conv3_3_x))  * conv4_3_x 
            #conv5_3_x = F.relu(self.bup2(conv4_3_x))  * conv5_3_x 
            fc7_x     = F.relu(self.bup3(conv5_3_x))  * fc7_x 
            conv6_2_x = F.relu(self.bup4(fc7_x))      * conv6_2_x 
            conv7_2_x = F.relu( self.bup5(conv6_2_x)) * conv7_2_x 
        
        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]
        if fem:
           sources[0] = self.cpm3_3(sources[0])
           sources[1] = self.cpm4_3(sources[1])
           sources[2] = self.cpm5_3(sources[2])
           sources[3] = self.cpm7(sources[3])
           sources[4] = self.cpm6_2(sources[4])
           sources[5] = self.cpm7_2(sources[5])
        
        # apply multibox head to source layers
        featuremap_size = []
        for  (feat, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([ feat.shape[2], feat.shape[3]])
            loc.append(l(feat).permute(0, 2, 3, 1).contiguous())
            if mo:
                if len(conf)==0:
                    chunk = torch.chunk(c(feat) , 4 , 1)
                    bmax  = torch.max(torch.max(chunk[0], chunk[1]) , chunk[2])
                    cls1  = torch.cat([bmax,chunk[3]], dim=1)
                    conf.append( cls1.permute(0, 2, 3, 1).contiguous() )
                else:
                    conf.append(c(feat).permute(0, 2, 3, 1).contiguous())
            elif mio:
                len_conf = len(conf)
                if cfg['mbox'][0] ==1 :
                    cls = self.mio_module(c(feat),len_conf)
                else:
                    mmbox = torch.chunk(c(feat) , cfg['mbox'][0] , 1)
                    cls_0 = self.mio_module(mmbox[0], len_conf)
                    cls_1 = self.mio_module(mmbox[1], len_conf)
                    cls_2 = self.mio_module(mmbox[2], len_conf)
                    cls_3 = self.mio_module(mmbox[3], len_conf)
                    cls = torch.cat([cls_0, cls_1, cls_2, cls_3] , dim=1)
                conf.append(cls.permute(0, 2, 3, 1).contiguous())
            else:
                conf.append(c(feat).permute(0, 2, 3, 1).contiguous())
        if pa:
            mbox_num = cfg['mbox'][0]
            face_loc = torch.cat(  [o[:,:,:,:4*mbox_num].contiguous().view(o.size(0),-1) for o in loc],1)
            face_conf = torch.cat( [o[:,:,:,:2*mbox_num].contiguous().view(o.size(0),-1) for o in conf],1)
            head_loc = torch.cat( [o[:,:,:,4*mbox_num:8*mbox_num].contiguous().view(o.size(0),-1) for o in loc[1:]],1)
            head_conf = torch.cat( [o[:,:,:,2*mbox_num:4*mbox_num].contiguous().view(o.size(0),-1) for o in conf[1:]],1)
            body_loc = torch.cat( [o[:,:,:,8*mbox_num:].contiguous().view(o.size(0),-1) for o in loc[2:]],1)
            body_conf = torch.cat( [o[:,:,:,4*mbox_num:].contiguous().view(o.size(0),-1) for o in conf[2:]],1)
        else:
            face_loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            face_conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            self.cfg['feature_maps'] = featuremap_size
            self.cfg['min_dim'] = image_size
            self.priors = self.init_priors(self.cfg)
            if refine:
                output = self.detect(
                  face_loc.view(face_loc.size(0), -1, 4),         # loc preds
                  self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes)), # conf preds
                  self.priors.type(type(x.data)),                  # default boxes
                  arm_loc.view(arm_loc.size(0), -1, 4),
                  self.softmax(arm_conf.view(arm_conf.size(0), -1, self.num_classes)),
                )
            else:
                output = self.detect(
                  face_loc.view(face_loc.size(0), -1, 4),         # loc preds
                  self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes)), # conf preds
                  self.priors.type(type(x.data))                  # default boxes
                )
        else:
            self.cfg['feature_maps'] = featuremap_size
            self.cfg['min_dim'] = image_size
            if pa: 
              self.face_priors = self.init_priors(self.cfg)
              self.head_priors = self.init_priors(self.cfg , min_size=cfg['min_sizes'][:-1], max_size=cfg['max_sizes'][:-1])
              self.body_priors = self.init_priors(self.cfg , min_size=cfg['min_sizes'][:-2], max_size=cfg['max_sizes'][:-2])
              output = (
                face_loc.view(face_loc.size(0), -1, 4),
                face_conf.view(face_conf.size(0), -1, self.num_classes),
                self.face_priors,
 
                head_loc.view(head_loc.size(0), -1, 4),
                head_conf.view(head_conf.size(0), -1, self.num_classes),
                self.head_priors,

                body_loc.view(body_loc.size(0), -1, 4),
                body_conf.view(body_conf.size(0), -1, self.num_classes),
                self.body_priors
              )
            else:
              self.priors = self.init_priors(self.cfg)
              output = (
                face_loc.view(face_loc.size(0), -1, 4),
                face_conf.view(face_conf.size(0), -1, self.num_classes),
                self.priors
              )
            if refine:
                output = output + tuple((arm_loc.view(arm_loc.size(0), -1, 4), arm_conf.view(arm_conf.size(0), -1, self.num_classes) ))
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or ext == '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def mio_module(self, each_mmbox, len_conf):
        chunk = torch.chunk(each_mmbox, each_mmbox.shape[1], 1)
        bmax  = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls = ( torch.cat([bmax,chunk[3]], dim=1) if len_conf==0 else torch.cat([chunk[3],bmax],dim=1) )
        if len(chunk)==6:
            cls = torch.cat([cls, chunk[4], chunk[5]], dim=1) 
        elif len(chunk)==8:
            cls = torch.cat([cls, chunk[4], chunk[5], chunk[6], chunk[7]], dim=1) 
        return cls 

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def _upsample_product(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') * y

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(vgg_cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in vgg_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # add conv6, conv7
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3 , padding=1)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [ pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True) ]
    return layers


def add_extras(extras_cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(extras_cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, extras_cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]) ]
            flag = not flag
        in_channels = v
    return layers

def multibox(output_channels, mbox_cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = (512 if fem else v)
        loc_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * 4, kernel_size=3, padding=1)]
        if mo:
            if k==0:
                conf_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * 4, kernel_size=3, padding=1)]
            else:
                conf_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * num_classes, kernel_size=3, padding=1)]
        elif mio:
            conf_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * 4, kernel_size=3, padding=1)]
        else:
            conf_layers += [nn.Conv2d(input_channels,  mbox_cfg[k] * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers

def arm_multibox(output_channels, mbox_cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels =  v
        loc_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(input_channels,  mbox_cfg[k] * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers

class DeepHeadModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule , self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)
        #print(self._mid_channels)
        self.conv1 = nn.Conv2d( self._input_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d( self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d( self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d( self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1, padding=0)
    def forward(self, x):
        return self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)), inplace=True))
        #return self.conv4(self.conv3(self.conv2(self.conv1(x))))

def pa_multibox(output_channels, mbox_cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = (512 if fem else v)
        if k ==0:
            loc_output = 4
            conf_output = 2
        elif k==1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * loc_output)]
        if mio:
            conf_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * (2+conf_output))]
        else:
            conf_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * conf_output)]
    return (loc_layers, conf_layers)


'''
def pa_multibox(output_channels, mbox_cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = (512 if cpm else v)
        if k ==0:
            loc_output = 4
            conf_output = 2
        elif k==1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * loc_output, kernel_size=3, padding=1)]
        if mio:
            conf_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * (2+conf_output), kernel_size=3, padding=1)]
        else:  
            conf_layers += [nn.Conv2d(input_channels, mbox_cfg[k] * conf_output, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)
'''

def build_ssd(phase, size=640, num_classes=2):
    if phase != 'test' and phase != 'train':
        print('ERROR: Phase: ' + phase + ' not recognized')
        return
    if size!=640:
        print('ERROR: You specified size ' + repr(size) + '. However, ' +
              'currently only SSD640 (size=640) is supported!')
    return SSD(phase, size, num_classes)
