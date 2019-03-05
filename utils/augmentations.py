from __future__ import division
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import random as random_random
import pdb
import math
from data.config import widerface_640

cfg = widerface_640
das = cfg['data_anchor_sampling']

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size
    def __call__(self, image, boxes=None, labels=None):
        #print (image.shape[0]/self.size)
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels



class RandomBaiduCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self , size):
        
        self.mean = np.array([104, 117, 123],dtype=np.float32)
        self.maxSize = 12000    #max size
        self.infDistance = 9999999
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        random_counter = 0
        boxArea = (boxes[:,2] - boxes[:,0] + 1) * (boxes[:,3] - boxes[:,1] + 1)
        #argsort = np.argsort(boxArea)
        #rand_idx = random.randint(min(len(argsort),6))
        #print('rand idx',rand_idx)
        rand_idx = random.randint(len(boxArea))
        rand_Side = boxArea[rand_idx] ** 0.5
        #rand_Side = min(boxes[rand_idx,2] - boxes[rand_idx,0] + 1, boxes[rand_idx,3] - boxes[rand_idx,1] + 1)
        anchors = [16,32,64,128,256,512]
        distance = self.infDistance
        anchor_idx = 5
        for i,anchor in enumerate(anchors):
            if abs(anchor-rand_Side) < distance:
                distance = abs(anchor-rand_Side)
                anchor_idx = i
        target_anchor = random.choice(anchors[0:min(anchor_idx+1,5)+1])
        ratio = float(target_anchor) / rand_Side
        ratio = ratio * (2**random.uniform(-1,1))
        if int(height * ratio * width * ratio) > self.maxSize*self.maxSize:
            ratio = (self.maxSize*self.maxSize/(height*width))**0.5
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)
        boxes[:,0] *= ratio
        boxes[:,1] *= ratio
        boxes[:,2] *= ratio
        boxes[:,3] *= ratio
        height, width, _ = image.shape
        sample_boxes = []
        xmin = boxes[rand_idx,0]
        ymin = boxes[rand_idx,1]
        bw = (boxes[rand_idx,2] - boxes[rand_idx,0] + 1)
        bh = (boxes[rand_idx,3] - boxes[rand_idx,1] + 1)
        w = h = self.size

        for _ in range(50):
            if w < max(height,width):
                if bw <= w:
                    w_off = random.uniform(xmin + bw - w, xmin)
                else:
                    w_off = random.uniform(xmin, xmin + bw - w)
                if bh <= h:
                    h_off = random.uniform(ymin + bh - h, ymin)
                else:
                    h_off = random.uniform(ymin, ymin + bh -h)
            else:
                w_off = random.uniform(width - w, 0)
                h_off = random.uniform(height - h, 0)
            w_off = math.floor(w_off)
            h_off = math.floor(h_off)
            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(w_off), int(h_off), int(w_off+w), int(h_off+h)])
            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            # mask in all gt boxes that above and to the left of centers
            m1 = (rect[0] <= boxes[:, 0]) * (rect[1] <= boxes[:, 1])
            # mask in all gt boxes that under and to the right of centers
            m2 = (rect[2] >= boxes[:, 2]) * (rect[3] >= boxes[:, 3])
            # mask in that both m1 and m2 are true
            mask = m1 * m2
            overlap = jaccard_numpy(boxes,rect)
            # have any valid boxes? try again if not
            if not mask.any() and not overlap.max() > 0.7:
                continue
            else:
                sample_boxes.append(rect)

        if len(sample_boxes) > 0:
            choice_idx = random.randint(len(sample_boxes))
            choice_box = sample_boxes[choice_idx]
            #print('crop the box :',choice_box)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            m1 = (choice_box[0] < centers[:, 0]) * (choice_box[1] < centers[:, 1])
            m2 = (choice_box[2] > centers[:, 0]) * (choice_box[3] > centers[:, 1])
            mask = m1 * m2
            current_boxes = boxes[mask, :].copy()
            current_labels = labels[mask]
            current_boxes[:, :2] -= choice_box[:2]
            current_boxes[:, 2:] -= choice_box[:2]
            if choice_box[0] < 0 or choice_box[1] < 0:
                new_img_width = width if choice_box[0] >=0 else width-choice_box[0]
                new_img_height = height if choice_box[1] >=0 else height-choice_box[1]
                image_pad = np.zeros((new_img_height,new_img_width,3),dtype=float)
                image_pad[:, :, :] = self.mean
                start_left = 0 if choice_box[0] >=0 else -choice_box[0]
                start_top = 0 if choice_box[1] >=0 else -choice_box[1]
                image_pad[start_top:,start_left:,:] = image

                choice_box_w = choice_box[2] - choice_box[0]
                choice_box_h = choice_box[3] - choice_box[1]

                start_left = choice_box[0] if choice_box[0] >=0 else 0
                start_top = choice_box[1] if choice_box[1] >=0 else 0
                end_right = start_left + choice_box_w
                end_bottom = start_top + choice_box_h
                current_image = image_pad[start_top:end_bottom,start_left:end_right,:].copy()
                return current_image, current_boxes, current_labels
            current_image = image[choice_box[1]:choice_box[3],choice_box[0]:choice_box[2],:].copy()
            return current_image, current_boxes, current_labels
        else:
            return image, boxes, labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(5):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        
        return self.rand_light_noise(im, boxes, labels)


class RandomCrop(object):

    def __init__(self , image_size):
        self.options = [None, 0.3, 0.5, 0.7, 0.9]
        self.small_threshold = 8.0  

    def __call__(self, im, boxes, labels):

        imh, imw, _ = im.shape
        short_size = min(imw, imh)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        while True:
            mode = random_random.choice( self.options )
            for _ in range(50):
                if mode is None or mode < 0.7:
                  if mode is None:
                    w = short_size
                  else:
                    w = random_random.randrange(int(0.3*short_size), int(1*short_size))
                  h = w
                  if imw>w:
                    x = random_random.randrange(imw - w)
                  else:
                    x = 0
                  if imh>h:
                    y = random_random.randrange(imh - h)
                  else:
                    y = 0
                else:  # average sample
                   random_box = random_random.choice(boxes)
                   rbminx = random_box[0]
                   rbminy = random_box[1]
                   rbmaxx = random_box[2]
                   rbmaxy = random_box[3]
                   rbcx = ( rbminx + rbmaxx )/2
                   rbcy = ( rbminy + rbmaxy )/2
                   rbw = rbmaxx - rbminx
                   rbh = rbmaxy - rbminy
                   w = np.sqrt( rbw*rbh )
                   if w > 256:
                       random_scale = 640 / random_random.choice([16,32,64,128,256,512])
                   elif w > 128:
                       random_scale = 640 / random_random.choice([16,32,64,128,256])
                   elif w > 64:
                       random_scale = 640 / random_random.choice([16,32,64,128])
                   elif w > 32:
                       random_scale = 640 / random_random.choice([16,32,64])
                   else:
                       random_scale = 640 / random_random.choice([16,32])

                   w = int( w * random_scale)
                   h = w 
                   _min_x = max(0, rbminx - max(w,rbw+1) + rbw )
                   _min_y = max(0, rbminy - max(h,rbh+1) + rbh )
                   if _min_x == rbminx:
                     x = _min_x
                   else:
                     x = random_random.randrange(_min_x , rbminx)
                   if _min_y == rbminy:
                     y = _min_y
                   else:
                     y = random_random.randrange(_min_y , rbminy)
                   roi_max_x = min(imw , x+w )
                   roi_max_y = min(imh , y+h )
                   w = roi_max_x - x
                   h = roi_max_y - y

                roi = torch.FloatTensor([ [x, y, x+w, y+h] ])
                center = (boxes[:,:2] + boxes[:,2:]) / 2
                roi2 = roi.expand(len(center), 4)

                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])
                mask = mask[:,0] & mask[:,1]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                img = im[y:y+h , x:x+w , :]
                
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                
                boxes_uniform = selected_boxes / torch.FloatTensor([w,h,w,h]).expand_as(selected_boxes)
                boxwh = boxes_uniform[:,2:] - boxes_uniform[:,:2]
                #mask = (boxwh[:,0] > self.small_threshold) & (boxwh[:,1] > self.small_threshold) 
                mask = (boxwh[:,0]*w*boxwh[:,1]*h > self.small_threshold*self.small_threshold)
                if not mask.any():
                    continue
                
                selected_boxes_selected = selected_boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
                #print ( (selected_boxes_selected[:,2]-selected_boxes_selected[:,0])* (selected_boxes_selected[:,3]-selected_boxes_selected[:,1])   ) 
                return img, selected_boxes_selected.numpy(), selected_labels.numpy()

            self.small_threshold=self.small_threshold/2

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        if das:
            self.augment = Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                #RandomSampleCrop(),
                RandomBaiduCrop(self.size),
                #Expand(self.mean),
                RandomMirror(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean)
            ])
        else:
            self.augment = Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                #Expand(self.mean),
                RandomCrop(self.size),
                RandomMirror(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean)
            ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
