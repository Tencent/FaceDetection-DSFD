# -*- coding: utf-8 -*-
from __future__ import division , print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import refine_match, pa_sfd_match, sfd_match, match, log_sum_exp
import pdb

from data import  widerface_640
cfg = widerface_640
pa = cfg['progressive_anchor']
ac = cfg['anchor_compensation']
refine = cfg['refinedet']
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, odm=True, use_pa=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.part = ''
        self.odm = odm
        self.use_pa = use_pa
        #self.tmp = [0,0]
    def forward(self, predictions, targets):
        if pa and self.odm and self.use_pa: 
            self.part = 'face'
            if refine:
                face_loss_l , face_loss_c = self.part_forward( (predictions[0],predictions[1],predictions[2]), targets ,(predictions[-2],predictions[-1]), True)
            else:
                face_loss_l , face_loss_c = self.part_forward( (predictions[0],predictions[1],predictions[2]), targets)
            self.part = 'head'
            head_loss_l , head_loss_c = self.part_forward( (predictions[3],predictions[4],predictions[5]), targets)
            self.part = 'body'
            body_loss_l , body_loss_c = self.part_forward( (predictions[6],predictions[7],predictions[8]), targets)
            loss_l = (face_loss_l , head_loss_l , body_loss_l)
            loss_c = (face_loss_c , head_loss_c , body_loss_c)
        else:
            self.part = 'face'
            loss_l , loss_c = self.part_forward( predictions, targets)
        return loss_l , loss_c

    def part_forward(self, predictions, targets , arm_data=None , filter_negative=False ):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        if arm_data:
          arm_loc_data ,arm_conf_data = arm_data

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # sft match strategy , swordli
            if ac: 
                sfd_match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            else:
                if arm_data:
                    refine_match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx, arm_loc_data[idx].data)
                else:
                    match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
                
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
       
        # compute matched anchor number for each gt
        '''
        for i in targets:
            self.tmp[0] = self.tmp[0]+i.shape[0]
        self.tmp[1] = self.tmp[1] + conf_t.sum()
        print(self.tmp[1]/self.tmp[0])
        '''
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        '''
        numpos1 = sum(sum(pos[:,:160*160].data))
        numpos2 = sum(sum(pos[:,160*160: 160*160+80*80].data))
        numpos3 = sum(sum(pos[:,160*160+80*80: 160*160+80*80+40*40].data))
        numpos4 = sum(sum(pos[:,160*160+80*80+40*40: 160*160+80*80+40*40+20*20].data))
        numpos5 = sum(sum(pos[:,-10*10-5*5:-5*5:].data))
        numpos6 = sum(sum(pos[:,-5*5:].data))
        numposall = sum(sum(pos.data))
        print(numpos1, numpos2, numpos3, numpos4, numpos5, numpos6 , numposall)
        '''
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        
        # Compute max conf across batch for hard negative mining
        ignore = conf_t < 0
        #print(sum(conf_t[0].data.cpu().numpy()==1) , sum(conf_t[0].data.cpu().numpy()==-1))
        conf_t[ignore] = 0
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c[ignore] = 0  #filter out ignore

        loss_c = loss_c.view(num, -1)
        #loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        #pdb.set_trace()
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        #pdb.set_trace()
        return loss_l, loss_c

class focalLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_mining, encode_target,  use_gpu=True, gamma = 2, alpha = 0.25, use_pa=True):
        """
            focusing is parameter that can adjust the rate at which easy
            examples are down-weighted.
            alpha may be set by inverse class frequency or treated as a hyper-param
            If you don't want to balance factor, set alpha to 1
            If you don't want to focusing factor, set gamma to 1 
            which is same as normal cross entropy loss
        """
        super(focalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.threshold = overlap_thresh
        self.num_classes = num_classes
        self.encode_target = encode_target
        self.do_neg_mining = neg_mining
        self.variance = cfg['variance']
        self.part = ''
        self.use_pa = use_pa

    def forward(self, predictions, targets):
        if pa and self.use_pa:
            self.part = 'face'
            face_loss_l , face_loss_c = self.part_forward( (predictions[0],predictions[1],predictions[2]), targets)
            self.part = 'head'
            head_loss_l , head_loss_c = self.part_forward( (predictions[3],predictions[4],predictions[5]), targets)
            self.part = 'body'
            body_loss_l , body_loss_c = self.part_forward( (predictions[6],predictions[7],predictions[8]), targets)
            loss_l = (face_loss_l , head_loss_l , body_loss_l)
            loss_c = (face_loss_c , head_loss_c , body_loss_c)
        else:
            self.part = 'face'
            loss_l , loss_c = self.part_forward( predictions, targets)
        return loss_l , loss_c

    def part_forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
    
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # sft match strategy , swordli
            if ac: 
                sfd_match(self.threshold, truths, defaults, self.variance, labels,
                      loc_t, conf_t, idx)
            else:
                match(self.threshold, truths, defaults, self.variance, labels,
                      loc_t, conf_t, idx)
           
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_targets = Variable(loc_t, requires_grad=False)
        conf_targets = Variable(conf_t, requires_grad=False)

        ############# Localization Loss part ##############
        pos = conf_targets > 0 # ignore background
        num_pos = pos.long().sum(1, keepdim = True)
        
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_targets[pos_idx].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_p, loc_t, size_average = False)

       ############### Confiden Loss part ###############
        """
        #focal loss implementation(1)
        pos_cls = conf_targets > -1 # exclude ignored anchors
        mask = pos_cls.unsqueeze(2).expand_as(conf_preds)
        conf_p = conf_preds[mask].view(-1, conf_preds.size(2)).clone()
        conf_t = conf_targets[pos_cls].view(-1).clone()
        p = F.softmax(conf_p, 1)
        p = p.clamp(1e-7, 1. - 1e-7) # to avoid loss going to inf
        c_mask = conf_p.data.new(conf_p.size(0), conf_p.size(1)).fill_(0)
        c_mask = Variable(c_mask)
        ids = conf_t.view(-1, 1)
        c_mask.scatter_(1, ids, 1.)
        p_t = (p*c_mask).sum(1).view(-1, 1)
        p_t_log = p_t.log()
        # This is focal loss presented in ther paper eq(5)
        conf_loss = -self.alpha * ((1 - p_t)**self.gamma * p_t_log)
        conf_loss = conf_loss.sum()
        """
        # focal loss implementation(2)
        pos_cls = conf_targets >-1
        mask = pos_cls.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[mask].view(-1, conf_data.size(2)).clone()
        p_t_log = -F.cross_entropy(conf_p, conf_targets[pos_cls], size_average = False)
        p_t = torch.exp(p_t_log)
        # This is focal loss presented in the paper eq(5)
        conf_loss = -self.alpha * ((1 - p_t)**self.gamma * p_t_log)

        N = max(1 , num_pos.data.sum()) # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes 
        conf_loss /= N # exclude number of background?
        loc_loss /= N
        return conf_loss, loc_loss
