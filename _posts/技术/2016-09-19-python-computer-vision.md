---
layout: post
title: python computer vision
category: 技术
tags: 编程语言
keywords: Python
description: 库使用
---

# python computer vision

## 图像处理类库

### 示例

```python
from PIL import Image
pil_im = Image.open('a.jpg') #读图

pil_im = Image.open('a.jpg').convert('L') #读图并转化为灰度图

Image.open(infile).save(outfile) #存图

box = (100,100,400,400)
region = pil_im.crop(box) #裁剪图片

out = pil_im.resize((128,128)) #调整尺寸

import numpy as np
im = np.array(pil_im) #从PIL转化为Numpy格式

pil_im2 = Image.fromarray(im) #从Numpy转化为PIL格式

import pickle
with open('a.pkl','wb') as f: #pickle模块存储
    pickle.dump(v,f) #存储变量v
	
with open('a.pkl','rb') as f:
    v = pickle.load(f) #读取变量v
	
from scipy.ndimage import filters
im2 = filters.gaussian_filter(im, 5) #高斯模糊

import scipy
data = scipy.io.loadmat('test.mat') #读取matlab的.mat文件

data = {}
data['x'] = x
scipy.io.savemat('test.mat',data) #存储为.mat文件

from scipy.misc import imsave
imsave('test.jpg',im2) #通过scipy.misc保存图片
```

## 机器学习类

### 示例

```python
from scipy.cluster.vq import *
...

centroids,variance = vq(features,k)

code,distance = vq(features,centroids)

#kmeans
```

使用LibSVM

```python
import pickle
from svmutil import *

with open('数据.pkl','r') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)
	
class_1 = map(list,class_1)#数组转换成列表，因为libsvm不支持数组对象作为输入，即对class_1中每个对象都进行list转换
class_2 = map(list,class_2)
labels = list(labels)
samples = class_1+class_2

#创建svm
prob = svm_problem(label,samples)
param = svm_parameter('-t 2')

#参数-t表示核函数类型
#0 线性函数
#1 多项式函数
#2 径向基函数（默认）
#3 sigmoid函数

#在数据上训练svm

m = svm_train(prob, param)

#在训练数据上测试分类效果
res = svm_predict(labels,samples,m)
```

## OpenCV Python接口

```python
import cv2
#图像
im = cv2.imread('图片.jpg') #读 BGR存储
cv2.imwrite('图片.png',im) #写

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #彩色图像转灰度图
#cv2.COLOR_BGR2GRAY
#cv2.COLOR_BGR2RGB
#cv2.COLOR_GRAY2BGR

cv2.imshow('fig',gray)
cv2.waitKey()

#视频
cap = cv2.VideoCapture(0) #0为摄像头id， cap = cv2.VideoCapture('filename')表示从文件中读入

while True:
    ret,im = cap.read()
    cv2.imshow('video',im)
    key = cv2.waitKey(10)
    if key == 27:
        break
    if key == ord(' '):
        cv2.imwrite('vid_result.jpg',im)
```
