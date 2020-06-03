
<img src="imgs/DSFD_logo.PNG" title="Logo" width="300" /> 

## Update

* 2019.04: Release pytorch-version DSFD inference code.
* 2019.03: DSFD is accepted by CVPR2019.
* 2018.10: Our DSFD ranks No.1 on [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html) and [FDDB](http://vis-www.cs.umass.edu/fddb/results.html)


## Introduction
<p align='center'>
  <img src='./imgs/dsfd_video.gif' width=1000'/>
</p>

In this repo, we propose a novel face detection network, named DSFD, with superior performance over the state-of-the-art face detectors. You can use the code to evaluate our DSFD for face detection. 

For more details, please refer to our paper [DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220)! or poster [slide](./imgs/DSFD_CVPR2019_poster.pdf)!

<p align='center'>
<img src='./imgs/DSFD_framework.PNG' alt='DSFD Framework' width='1000px'>
</p>

Our DSFD face detector achieves state-of-the-art performance on [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html) and [FDDB](http://vis-www.cs.umass.edu/fddb/results.html) benchmark.

### WIDER FACE
<p align='center'>
<img src='./imgs/DSFD_widerface.PNG' alt='DSFD Widerface Performance' width='1000px'>
</p>

### FDDB
<p align='center'>
<img src='./imgs/DSFD_fddb.PNG' alt='DSFD FDDB Performance' width='1000px'>
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
git clone xxxxxx/FaceDetection-DSFD.git
cd FaceDetection-DSFD
export CUDA_VISIBLE_DEVICES=0
```


### Evaluation
1. Download the images of [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to `$DSFD_ROOT/data/`.

2. Download our DSFD model [微云](https://share.weiyun.com/567x0xQ) [google drive](https://drive.google.com/file/d/1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-/view?usp=sharing) trained on WIDER FACE training set to `$DSFD_ROOT/weights/`.

  
3. Check out `./demo.py` on how to detect faces using the DSFD model and how to plot detection results.
```
python demo.py [--trained_model [TRAINED_MODEL]] [--img_root  [IMG_ROOT]] 
               [--save_folder [SAVE_FOLDER]] [--visual_threshold [VISUAL_THRESHOLD]] 
    --trained_model      Path to the saved model
    --img_root           Path of test images
    --save_folder        Path of output detection resutls
    --visual_threshold   Confidence thresh
```

4. Evaluate the trained model via `./widerface_val.py` on WIDER FACE.
```
python widerface_val.py [--trained_model [TRAINED_MODEL]] [--save_folder [SAVE_FOLDER]] 
                         [--widerface_root [WIDERFACE_ROOT]]
    --trained_model      Path to the saved model
    --save_folder        Path of output widerface resutls
    --widerface_root     Path of widerface dataset
```

5. Download the [eval_tool](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip) to show the WIDERFACE performance.

6. Evaluate the trained model via `./fddb_test.py` on FDDB.
```
python widerface_test.py [--trained_model [TRAINED_MODEL]] [--split_dir [SPLIT_DIR]] 
                         [--data_dir [DATA_DIR]] [--det_dir [DET_DIR]]
    --trained_model      Path of the saved model
    --split_dir          Path of fddb folds
    --data_dir           Path of fddb all images
    --det_dir            Path to save fddb results
```

7. Download the [evaluation](http://vis-www.cs.umass.edu/fddb/evaluation.tgz) to show the FDDB performance.
8. Lightweight DSFD is [here](https://github.com/lijiannuist/lightDSFD).

## Qualitative Results
<p align='center'>
  <img src='./imgs/DSFD_demo1.PNG' width='1000'/>
</p>

<p align='center'>
  <img src='./imgs/DSFD_demo2.PNG' width='1000'/>
</p>


### Citation
If you find DSFD useful in your research, please consider citing: 
```
@inproceedings{li2018dsfd,
  title={DSFD: Dual Shot Face Detector},
  author={Li, Jian and Wang, Yabiao and Wang, Changan and Tai, Ying and Qian, Jianjun and Yang, Jian and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
## Contact
For any question, please file an issue or contact
```
Jian Li: swordli@tencent.com
```
