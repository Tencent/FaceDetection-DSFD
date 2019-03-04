## DSFD: Dual Shot Face Detector

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Jian Li](https://lijiannuist.github.io/)

### Introduction
We propose a novel face detection network, named DSFD, with superior performance on accuracy. You can use the code to evaluate the DSFD method for face detection. For more details, please refer to our paper [DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220)!

<p align="left">
<img src="https://github.com/TencentYoutuResearch/FaceDetection-DSFD/imgs/DSFD_framework.jpg" alt="FaceBoxes Framework" width="777px">
</p>

Our DSFD face detector achieves state-of-the-art performance on [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html) and [FDDB](http://vis-www.cs.umass.edu/fddb/results.html) benchmark.

<p align="left">
<img src="https://github.com/TencentYoutuResearch/FaceDetection-DSFD/imgs/DSFD_widerface.jpg" alt="FaceBoxes Framework" width="777px">
</p>

<p align="left">
<img src="https://github.com/TencentYoutuResearch/FaceDetection-DSFD/imgs/DSFD_fddb.jpg" alt="FaceBoxes Performance" width="770px">
</p>

### Demo
<p align='center'>
  <img src='https://github.com/TencentYoutuResearch/FaceDetection-DSFD/imgs/DSFD_demo.jpg' width='1280'/>
</p>

## Prerequisites
- Torch == 0.3.1
- Linux
- NVIDIA GPU = Tesla P40 
- CUDA CuDNN 

## Getting Started
### Setup

Clone the github repository:

```bash
git  clone https://github.com/TencentYoutuResearch/FaceDetection-DSFD.git
cd FaceDetection-DSFD
```

Please cite DSFD in your publications if it helps your research:

	@article{li2018dsfd,
	  title={DSFD: Dual Shot Face Detector},
	  author={Li, Jian and Wang, Yabiao and Wang, Changan and Tai, Ying and Qian, Jianjun and Yang, Jian and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
	  journal={arXiv preprint arXiv:1810.10220},
	  year={2018}
	}
