## DSFD: Dual Shot Face Detector

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Jian Li](https://lijiannuist.github.io/), Yabiao Wang, Changan Wang, Ying Tai 

### Introduction
We propose a novel face detection network, named DSFD, with superior performance on accuracy. You can use the code to evaluate the DSFD method for face detection. For more details, please refer to our paper [DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220)!

<p align="left">
<img src="https://github.com/TencentYoutuResearch/FaceDetection-DSFD/blob/master/imgs/DSFD_framework.PNG" alt="DSFD Framework" width="777px">
</p>

Our DSFD face detector achieves state-of-the-art performance on [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html) and [FDDB](http://vis-www.cs.umass.edu/fddb/results.html) benchmark.

<p align="left">
<img src="https://github.com/TencentYoutuResearch/FaceDetection-DSFD/blob/master/imgs/DSFD_widerface.PNG" alt="DSFD Widerface Performance" width="777px">
</p>

<p align="left">
<img src="https://github.com/TencentYoutuResearch/FaceDetection-DSFD/blob/master/imgs/DSFD_fddb.PNG" alt="DSFD FDDB Performance" width="777px">
</p>

### Demo
<p align='center'>
  <img src='https://github.com/TencentYoutuResearch/FaceDetection-DSFD/blob/master/imgs/DSFD_demo3.PNG' width='1280'/>
</p>

<p align='center'>
  <img src='https://github.com/TencentYoutuResearch/FaceDetection-DSFD/blob/master/imgs/DSFD_demo3.PNG' width='1280'/>
</p>

## Requirements
- Torch == 0.3.1
- Torchvision == 0.2.1
- Python == 3.6
- NVIDIA GPU == Tesla P40 
- Linux CUDA CuDNN

## Getting Started

### Installation
Clone the github repository. We will call the cloned directory as `$DSFD_ROOT`.
```bash
git clone https://github.com/TencentYoutuResearch/FaceDetection-DSFD.git
cd FaceDetection-DSFD
```


### Evaluation
1. Download the images of [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to `$DSFD_ROOT/data/`.

2. Download our [DSFD model](https://drive.google.com/open?id=1eyqFViMoBlN8JokGRHxbnJ8D4o0pTWac) trained on WIDER FACE training set to `$DSFD_ROOT/weights/`.


3. Check out [`tools/demo.py`](https://github.com/TencentYoutuResearch/FaceDetection-DSFD/blob/master/test/demo.py) on how to detect faces using the DSFD model and how to plot detection results.

4. Evaluate the trained model via [`./widerface_test.py`](https://github.com/TencentYoutuResearch/FaceDetection-DSFD/blob/master/test/widerface_test.py) on WIDER FACE.
```
export CUDA_VISIBLE_DEVICES=0
python widerface_test.py [--trained_model [TRAINED_MODEL]] [--save_folder [SAVE_FOLDER]] 
                         [--widerface_root [WIDERFACE_ROOT]]
    --trained_model      Path to the saved model
    --save_folder        Path of output widerface resutls
    --widerface_root     Path of widerface dataset
```

5. Evaluate the trained model via [`./fddb_test.py`](https://github.com/sTencentYoutuResearch/FaceDetection-DSFD/blob/master/test/fddb_test.py) on FDDB.

6. Download the [eval_tool](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip) to show the WIDERFACE performance.


### Citing DSFD
Please cite DSFD in your publications if it helps your research:

	@article{li2018dsfd,
	  title={DSFD: Dual Shot Face Detector},
	  author={Li, Jian and Wang, Yabiao and Wang, Changan and Tai, Ying and Qian, Jianjun and Yang, Jian and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
	  journal={arXiv preprint arXiv:1810.10220},
	  year={2018}
	}
