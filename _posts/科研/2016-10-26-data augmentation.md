---
layout: post
title: Data Augmentation
category: 科研
tags: 深度学习
keywords: 数据增强
description: 
---

# 深度学习中的数据增强实现（Data Augmentation）

深度学习中，为了避免数据过拟合，通常需要输入海量的数据，然后通过算法对图像数据进行几何变换，改变图像像素的位置并保证特征不变。主要的图像数据增强变换主要有以下几种：

旋转/反射变换(Rotation/reflection): 随机旋转图像一定角度; 改变图像内容的朝向;

翻转变换(flip): 沿着水平或者垂直方向翻转图像;

缩放变换(zoom): 按照一定的比例放大或者缩小图像;

平移变换(shift): 在图像平面上对图像以一定方式进行平移，可以采用随机或人为定义的方式指定平移范围和平移步长, 沿水平或竖直方向进行平移. 改变图像内容的位置;

尺度变换(scale): 对图像按照指定的尺度因子, 进行放大或缩小; 或者参照SIFT特征提取思想, 利用指定的尺度因子对图像滤波构造尺度空间. 改变图像内容的大小或模糊程度;

对比度变换(contrast): 在图像的HSV颜色空间，改变饱和度S和V亮度分量，保持色调H不变. 对每个像素的S和V分量进行指数运算(指数因子在0.25到4之间), 增加光照变化;

噪声扰动(noise): 对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声;

颜色变换(color): 在训练集像素值的RGB颜色空间进行PCA, 得到RGB空间的3个主方向向量,3个特征值,$p1$, $p2$, $p3$, $\lambda 1$, $\lambda 2$, $\lambda 3$. 对每幅图像的每个像素$$I_{xy}=\left \{ I^{R_{ xy }} ,I^{G_{ xy }} ,I^{B_{ xy }} \right \}^{T}$$进行加上如下的变化:

$$\left [ p1,p2,p3 \right ]\left [ \alpha 1\lambda 1 ,\alpha 2\lambda 2 ,\alpha 3\lambda 3 \right ]^{T}$$

其中:$\alpha_{i}\$是满足均值为0,方差为0.1的随机变量.

在已有开源代码库Keras中有数据增强的代码实现，首先安装Keras.

```
sudo pip install keras
```

若import keras时出现"TypeError: <method 'max' of 'numpy.ndarray' objects> is not a Python function"需要更新pandas >= 0.16.0

Data Augmentation的代码：

```python
#!/usr/bin/env python
#-*- coding: utf-8 -*-
# import packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
	rotation_range=0.2,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')
img = load_img('lena.jpg') # this is a PIL image, please replace to your own file path
x = img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape) # this is a Numpy array with shape (1, 3, 150, 150)
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x,
	batch_size=1,
	save_to_dir='data/preview',
	save_prefix='lena',
	save_format='jpg'):
    i += 1
    if i > 20:
        break # otherwise the generator would loop indefinitely
```

主要函数：ImageDataGenerator　实现了大多数上文中提到的图像几何变换方法．函数原型如下：

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())
```

参数

```
featurewise_center：布尔值，使输入数据集去中心化（均值为0）
samplewise_center：布尔值，使输入数据的每个样本均值为0
featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化
samplewise_std_normalization：布尔值，将输入的每个样本除以其自身的标准差
zca_whitening：布尔值，对输入数据施加ZCA白化
rotation_range：整数，数据提升时图片随机转动的角度
width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）
zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
channel_shift_range：浮点数，随机通道偏移的幅度
fill_mode：；‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
horizontal_flip：布尔值，进行随机水平翻转
vertical_flip：布尔值，进行随机竖直翻转
rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
dim_ordering：‘tf’和‘th’之一，规定数据的维度顺序。‘tf’模式下数据的形状为samples, width, height, channels，‘th’下形状为(samples, channels, width, height).该参数的默认值是Keras配置文件~/.keras/keras.json的image_dim_ordering值,如果你从未设置过的话,就是'th'
```

# tensorflow中的部分数据增强

```python
import tensorflow as tf
import cv2
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('random_flip_up_down', True, 'If uses flip')
flags.DEFINE_boolean('random_flip_left_right', True, 'If uses flip')
flags.DEFINE_boolean('random_brightness', True, 'If uses brightness')
flags.DEFINE_boolean('random_contrast', True, 'If uses contrast')
flags.DEFINE_boolean('random_saturation', True, 'If uses saturation')
flags.DEFINE_integer('image_size', 224, 'image size.')

"""
#flags examples
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
"""
def pre_process(images): 
    if FLAGS.random_flip_up_down: 
	images = tf.image.random_flip_up_down(images) 
    if FLAGS.random_flip_left_right: 
	images = tf.image.random_flip_left_right(images) 
    if FLAGS.random_brightness: 
        images = tf.image.random_brightness(images, max_delta=0.3) 
    if FLAGS.random_contrast: 
        images = tf.image.random_contrast(images, 0.8, 1.2)
    if FLAGS.random_saturation:
	tf.image.random_saturation(images, 0.3, 0.5)
    new_size = tf.constant([FLAGS.image_size,FLAGS.image_size],dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images

raw_image = cv2.imread("004545.jpg")
#image = tf.Variable(raw_image)
image = tf.placeholder("uint8",[None,None,3])
images = pre_process(image)
with tf.Session() as session:
    result = session.run(images, feed_dict={image: raw_image})
cv2.imshow("image",result.astype(np.uint8))
cv2.waitKey(1000)
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)
