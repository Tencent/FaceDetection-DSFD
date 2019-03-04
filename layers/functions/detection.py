from __future__ import division
import torch
from torch.autograd import Function
from ..box_utils import decode, nms, center_size
from data import widerface_640 as cfg
import pdb

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data, arm_loc_data=None , arm_conf_data=None):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        
        #swordli
        #num_priors = loc_data.size(1)
       
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,  self.num_classes).transpose(2, 1)
        if cfg['refinedet']:
            conf_preds_arm = arm_conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        
        # Decode predictions into bboxes.
        for i in range(num):
           if cfg['refinedet']:
              #default  = center_size(decode(arm_loc_data[i] , prior_data , self.variance))
              decoded_boxes_arm = decode(arm_loc_data[i] , prior_data , self.variance)
              default = center_size(decoded_boxes_arm)
              decoded_boxes_odm = decode(loc_data[i], default, self.variance)
              decoded_boxes = torch.cat((decoded_boxes_odm , decoded_boxes_arm),dim=0)
              conf_scores = torch.cat((conf_preds[i].clone(),conf_preds_arm[i].clone()),dim=1)
           else:
                default = prior_data
                decoded_boxes = decode(loc_data[i], default, self.variance)
                # For each class, perform nms
                conf_scores = conf_preds[i].clone()

           for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
