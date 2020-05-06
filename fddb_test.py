from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import pdb
import sys
#import matplotlib.pyplot as plt
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.misc import imread, imresize, imsave, imshow
from torch.autograd import Variable

from data import *
from data import BaseTransform, TestBaseTransform
from data import WIDERFace_CLASSES as labelmap
from data import (WIDERFace_ROOT, WIDERFaceAnnotationTransform,
                  WIDERFaceDetection)
from face_ssd import build_ssd
from utils import draw_toolbox
from widerface_val import (bbox_vote, detect_face, multi_scale_test,
                           multi_scale_test_pyramid)

#plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='DSFD: Dual Shot Face Detector')
parser.add_argument('--trained_model', default='weights/WIDERFace_DSFD_RES152.pth',
                    type=str, help='Trained state_dict file path to open')

parser.add_argument('--split_dir', default='./fddb/FDDB-folds',
                    type=str, help='Dir to folds')
parser.add_argument('--data_dir', default='./fddb/originalPics',
                    type=str, help='Dir to all images')
parser.add_argument('--det_dir', default='./fddb/results1',
                    type=str, help='Dir to save results')

parser.add_argument('--visual_threshold', default=0.01, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


# load net
cfg = widerface_640
num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.cuda()
net.eval()
print('Finished loading model!')


def test_fddbface():
    # evaluation
    cuda = args.cuda
    thresh=cfg['conf_thresh']
    os.makedirs(args.det_dir, exist_ok=True)

    all_splits = sorted([_ for _ in os.listdir(args.split_dir) if 'ellipseList' not in _])
    for folder_ind in range(1, 11):
        with open(os.path.join(args.split_dir, all_splits[folder_ind-1]), 'r') as fp:
            read_lines = fp.readlines()
            all_images = [line.strip() for line in read_lines]
        sys.stdout.write('>> Predicting folder %d/%d\n' % (folder_ind, 10))
        sys.stdout.flush()
        #print(all_images)
        with open(os.path.join(args.det_dir, 'fold-{:02d}-out.txt'.format(folder_ind)), 'wt') as f:
            all_image_length = len(all_images)
            for image_ind, image_name in enumerate(all_images):
                sys.stdout.write('\r>> Predicting image %d/%d' % (image_ind, all_image_length))
                sys.stdout.flush()
                #np_image = imread(os.path.join(data_dir, image_name+'.jpg'))

                np_image = cv2.imread(os.path.join(args.data_dir, image_name+'.jpg'))
                if len(np_image.shape) < 3:
                    np_image = np.stack((np_image,) * 3, -1)
                image = np_image#torch.from_numpy(np_image).permute(2, 0, 1)

                #max_im_shrink = ( (2000.0*2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
                max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
                max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink

                shrink = max_im_shrink if max_im_shrink < 1 else 1

                det0 = detect_face(image, shrink)  # origin test
                det1 = flip_test(image, shrink)    # flip test

                det = np.row_stack((det0, det1))

                dets = bbox_vote(det)
                #dets = bbox_vote2(det0)

                # if not os.path.exists(save_path + event):
                #     os.makedirs(save_path + event)
                # f = open(save_path + event + '/' + img_id.split('.')[0] + '.txt', 'w')
                # #f = open(save_path + str(event[0][0].encode('utf-8'))[2:-1]  + '/' + im_name + '.txt', 'w')
                # write_to_txt(f, dets , event, img_id)


                bbox_xmin = dets[:, 0]
                bbox_ymin = dets[:, 1]
                bbox_xmax = dets[:, 2]
                bbox_ymax = dets[:, 3]
                scores = dets[:, 4]


                bbox_height = bbox_ymax - bbox_ymin + 1
                bbox_width = bbox_xmax - bbox_xmin + 1


                img_to_draw = draw_toolbox.absolute_bboxes_draw_on_img(np_image, scores, dets, thickness=2)
                imsave(os.path.join('./debug/{}.jpg').format(image_ind), img_to_draw)


                valid_mask = np.logical_and(np.logical_and((bbox_height > 1), (bbox_width > 1)), (scores > 0.05))

                f.write('{:s}\n'.format(image_name))
                f.write('{}\n'.format(np.count_nonzero(valid_mask)))
                #print(valid_mask.shape[0], bbox_xmin.shape[0], bboxes.shape[0], bbox_width.shape[0], scores.shape[0])
                for det_ind in range(valid_mask.shape[0]):
                    if not valid_mask[det_ind]:
                        continue

                    f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(np.floor(bbox_xmin[det_ind]), np.floor(bbox_ymin[det_ind]), np.ceil(bbox_width[det_ind]), np.ceil(bbox_height[det_ind]), scores[det_ind]))
        sys.stdout.write('\n')
        sys.stdout.flush()

if __name__=='__main__':
    test_fddbface()
