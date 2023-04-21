---
layout: post
title: Fully Convolutional Models for Semantic Segmentation
category: 科研
tags: 深度学习
keywords: 应用
description: 
---

# Fully Convolutional Models for Semantic Segmentation

利用FCN训练一个分割网络，代码地址https://github.com/shelhamer/fcn.berkeleyvision.org

下面以pascal/VOC为例voc-fcn-alexnet网络说明训练过程： 首先在data/pascal中将voc的数据放入，写一个训练脚本,我们以alexnet来finetune这个分割网络，由于alexnet的fc6和fc7是全连接网络，我们需要将fc6和fc7转化为conv层，实践过程中，用了alexnet的全连接层转化conv层，网络的收敛速度将得到极大的提升。

```python
import sys
sys.path.insert(0, '/home/dlg/fcn.berkeleyvision.org/caffe-master/python')
sys.path.insert(0, '/home/dlg/fcn.berkeleyvision.org')
import caffe
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '/home/dlg/fcn.berkeleyvision.org/voc-fcn-alexnet/bvlc_reference_caffenet.caffemodel' #alexnet的分类网络

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('/home/dlg/fcn.berkeleyvision.org/voc-fcn-alexnet/deploy.prototxt',
                weights,
                caffe.TEST)
# 将fc6和fc7的全连接层转化为卷积层
params = ['fc6', 'fc7']
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

solver = caffe.SGDSolver('/home/dlg/fcn.berkeleyvision.org/voc-fcn-alexnet/solver.prototxt')
solver.net.copy_from(weights)

params_full_conv = ['fc6_my', 'fc7_my']
conv_params = {pr: (solver.net.params[pr][0].data, solver.net.params[pr][1].data) for pr in params_full_conv}

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]


# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/home/dlg/fcn.berkeleyvision.org/data/pascal/VOCdevkit2007/VOC2007/ImageSets/Segmentation/val.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
```

test的过程如下

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Make sure that caffe is on the python path:
import sys
sys.path.insert(0, '/home/dlg/fcn.berkeleyvision.org/caffe-master/python')
sys.path.insert(0, '/home/dlg/fcn.berkeleyvision.org')

import caffe

caffe.set_mode_gpu()
net = caffe.Net('/home/dlg/fcn.berkeleyvision.org/voc-fcn-alexnet/test.prototxt',
               '/home/dlg/fcn.berkeleyvision.org/voc-fcn-alexnet/snapshot/train_iter_32000.caffemodel', caffe.TEST)

#net = caffe.Net('/home/dlg/fcn.berkeleyvision.org/pascalcontext-fcn16s/deploy.prototxt',
#                '/home/dlg/fcn.berkeleyvision.org/pascalcontext-fcn16s/pascalcontext-fcn16s-heavy.caffemodel', caffe.TEST)
# load image and prepare as a single input batch for Caffe
im = cv2.imread('/home/dlg/fcn.berkeleyvision.org/data/pascal/VOCdevkit2007/VOC2007/JPEGImages/000018.jpg')
pixel_means = np.load('/home/dlg/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)

in_ = im.astype(np.float32, copy=True)
in_ -= pixel_means
norm_img = in_.transpose((2,0,1))
blob = norm_img[np.newaxis, :, :, :]
net.blobs['data'].reshape(*blob.shape)
net.blobs['data'].data[...] = blob

# make classification map by forward
net.forward()
output = net.blobs['score'].data.reshape(60,norm_img.shape[1],norm_img.shape[2])
plt.imshow(output.argmax(axis=0))
plt.show()
```

下面贴出测试的结果:

![1](/public/img/posts/FCN/000018.jpg)

![2](/public/img/posts/FCN/result.jpg)

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)