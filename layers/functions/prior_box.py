from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
import pdb

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg , min_size, max_size):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.min_sizes = min_size
        self.max_sizes = max_size
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):

        mean = []

        if len(self.min_sizes) == 5:
            self.feature_maps = self.feature_maps[1:]
            self.steps = self.steps[1:]
        if len(self.min_sizes) == 4:
            self.feature_maps = self.feature_maps[2:]
            self.steps = self.steps[2:]

        for k, f in enumerate(self.feature_maps):
            #for i, j in product(range(f), repeat=2):
            for i in range(f[0]):
              for j in range(f[1]):
                #f_k = self.image_size / self.steps[k]
                f_k_i = self.image_size[0] / self.steps[k]
                f_k_j = self.image_size[1] / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k_j
                cy = (i + 0.5) / f_k_i
                # aspect_ratio: 1
                # rel size: min_size
                s_k_i = self.min_sizes[k]/self.image_size[1]
                s_k_j = self.min_sizes[k]/self.image_size[0]
                # swordli@tencent
                if len(self.aspect_ratios[0]) == 0:
                    mean += [cx, cy, s_k_i, s_k_j]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                #s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                if len(self.max_sizes) == len(self.min_sizes):
                    s_k_prime_i = sqrt(s_k_i * (self.max_sizes[k]/self.image_size[1]))
                    s_k_prime_j = sqrt(s_k_j * (self.max_sizes[k]/self.image_size[0]))    
                    mean += [cx, cy, s_k_prime_i, s_k_prime_j]
                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    if len(self.max_sizes) == len(self.min_sizes):
                        mean += [cx, cy, s_k_prime_i/sqrt(ar), s_k_prime_j*sqrt(ar)]
                    mean += [cx, cy, s_k_i/sqrt(ar), s_k_j*sqrt(ar)]
                
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
