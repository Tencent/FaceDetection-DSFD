from __future__ import print_function

import argparse
import math
import os
#from resnet50_ssd import build_sfd
import pdb
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from data import *
from data import BaseTransform, TestBaseTransform
from data import WIDERFace_CLASSES as labelmap
from data import (WIDERFace_ROOT, WIDERFaceAnnotationTransform,
                  WIDERFaceDetection)
from face_ssd import build_ssd
from widerface_val import bbox_vote

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='DSFD:Dual Shot Face Detector')
parser.add_argument('--trained_model', default='weights/WIDERFace_DSFD_RES152.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval_tools/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.1, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--img_root', default='./data/worlds-largest-selfie.jpg', help='Location of test images directory')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of WIDERFACE root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def write_to_txt(f, det , event , im_name):
    f.write('{:s}\n'.format(event + '/' + im_name))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4] 
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))

def infer(net , img , transform , thresh , cuda , shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0) , volatile=True)
    if cuda:
        x = x.cuda()
    #print (shrink , x.shape)
    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([ img.shape[1]/shrink, img.shape[0]/shrink,
                         img.shape[1]/shrink, img.shape[0]/shrink] )
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            #label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3]) 
            det.append([pt[0], pt[1], pt[2], pt[3], score])
            j += 1
    if (len(det)) == 0:
        det = [ [0.1,0.1,0.2,0.2,0.01] ]
    det = np.array(det)

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det

def infer_flip(net , img , transform , thresh , cuda , shrink):
    img = cv2.flip(img, 1)
    det = infer(net , img , transform , thresh , cuda , shrink)
    det_t = np.zeros(det.shape)
    det_t[:, 0] = img.shape[1] - det[:, 2]
    det_t[:, 1] = det[:, 1]
    det_t[:, 2] = img.shape[1] - det[:, 0]
    det_t[:, 3] = det[:, 3]
    det_t[:, 4] = det[:, 4]
    return det_t


def infer_multi_scale_sfd(net , img , transform , thresh , cuda ,  max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = infer(net , img , transform , thresh , cuda , st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = infer(net , img , transform , thresh , cuda , bt)
    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , bt)))
            bt *= 2
        det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , max_im_shrink) ))
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b


def vis_detections(im,  dets, image_name , thresh=0.5):
    '''Draw detected bounding boxes.'''
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    print (len(inds))
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2.5)
            )
        '''
        ax.text(bbox[0], bbox[1] - 5,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=10, color='white')
        '''
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(args.save_folder+image_name, dpi=fig.dpi)

def test_oneimage():
    # load net
    cfg = widerface_640
    num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
    net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.cuda()
    net.eval()
    print('Finished loading model!')

    # evaluation
    cuda = args.cuda
    transform = TestBaseTransform((104, 117, 123))
    thresh=cfg['conf_thresh']
    #save_path = args.save_folder
    #num_images = len(testset)
 
    # load data
    path = args.img_root
    img_id = 'face'
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    max_im_shrink = ( (2000.0*2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    det0 = infer(net , img , transform , thresh , cuda , shrink)
    det1 = infer_flip(net , img , transform , thresh , cuda , shrink)
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = infer(net , img , transform , thresh , cuda , st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    factor = 2
    bt = min(factor, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = infer(net , img , transform , thresh , cuda , bt)
    # enlarge small iamge x times for small face
    if max_im_shrink > factor:
        bt *= factor
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , bt)))
            bt *= factor
        det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , max_im_shrink) ))
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    det = np.row_stack((det0, det1, det_s, det_b))
    det = bbox_vote(det)
    vis_detections(img , det , img_id, args.visual_threshold)


if __name__ == '__main__':
    test_oneimage()
