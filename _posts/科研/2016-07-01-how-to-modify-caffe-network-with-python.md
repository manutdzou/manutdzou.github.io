---
layout: post
title: 用python写caffe网络配置
category: 科研
tags: 深度学习
keywords: source code
description: 
---

# Caffe提供了用python来写网络配置的接口net_spec.py

需要的库

```Python
import caffe
from caffe import layers as L
from caffe import params as P
```

使用pycaffe定义的net:

```Python
n = caffe.NetSpec()
```

定义DataLayer

```Python
n.data, n.label = L.Data(batch_size=batch_size,
                         backend=P.Data.LMDB, source=lmdb,
                         transform_param=dict(scale=1. / 255), ntop=2)
# 效果如下：
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    scale: 0.00392156862745
  }
  data_param {
    source: "mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
```

定义ConvolutionLayer

```Python
n.conv1 = L.Convolution(n.data, kernel_size=5,
                        num_output=20, weight_filler=dict(type='xavier'))

# 效果如下：

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 20
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
  }
}
```

定义PoolingLayer

```Python
n.pool1 = L.Pooling(n.conv1, kernel_size=2,
                    stride=2, pool=P.Pooling.MAX)

# 效果如下：

layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
```

定义InnerProductLayer

```Python
n.ip1 = L.InnerProduct(n.pool2, num_output=500,
                       weight_filler=dict(type='xavier'))

# 效果如下：

layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
  }
}
```

定义ReluLayer

```Python
n.relu1 = L.ReLU(n.ip1, in_place=True)

# 效果如下：

layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
```

定义SoftmaxWithLossLayer

```Python
n.loss = L.SoftmaxWithLoss(n.ip2, n.label)

# 效果如下：

layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```

下面是一个cifar的小网络示例

```Python
#coding:utf-8
#!/usr/bin/env python

import sys

caffe_root ='/home/dlg/caffe-master/'
sys.path.insert(0, caffe_root+'python') #添加pycaffe的路径
import caffe
from caffe import layers as L, params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom,ks,nout,stride=1,pad=0,
             param=learned_param,
             weight_filler=dict(type='gaussian',std=0.01),
             bias_filler=dict(type='constant',value=0.1)):
    conv = L.Convolution(bottom,kernel_size=ks,stride=stride,
                         num_output=nout,pad=pad,
                         param=param,weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom,nout,param=learned_param,
            weight_filler=dict(type='gaussian',std=0.01),
            bias_filler=dict(type='constant',value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom,ks,stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def ave_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)

def cifar_quicknet(data, num_classes=10,
                 batch_size=128,learn_all=True):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size,backend=P.Data.LMDB, source=data,
                             transform_param=dict(scale=1./255), ntop=2)
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 5, 32, param=param)
    n.pool1 = max_pool(n.conv1, 3, stride=2)
    n.conv2, n.relu2 = conv_relu(n.pool1, 5, 32, param=param)
    n.pool2 = ave_pool(n.conv2, 3, stride=2)
    n.conv3, n.relu3 = conv_relu(n.pool2, 5, 64, param=param)
    n.pool3 = ave_pool(n.conv3, 3, stride=2)
    n.fc1, n.relu4 = fc_relu(n.pool3, 64, param=param, weight_filler=dict(type='gaussian', std=0.1))
    n.fc2, n.relu5 = fc_relu(n.fc1, num_classes, param=param, weight_filler=dict(type='gaussian', std=0.1))
    n.loss = L.SoftmaxWithLoss(n.relu5, n.label)

    return n.to_proto()

def make_net():
   with open('/home/dlg/train.prototxt', 'w') as f:
      f.write(str(cifar_quicknet('mnist/mnist_train_lmdb', 64)))
   with open('/home/dlg/test.prototxt', 'w') as f:
      f.write(str(cifar_quicknet('mnist/mnist_test_lmdb', 100)))

if __name__ == '__main__':
    make_net()
```

lenet net示例

```Python
#!/usr/bin/env python

import sys

caffe_root = '/home/dlg/caffe-master/'
sys.path.insert(0,caffe_root+'python')

import caffe
from caffe import layers as L
from caffe import params as P


def lenet(lmdb, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size,
                             backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5,
                            num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5,
                            num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500,
                           weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10,
                           weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)

    return n.to_proto()


def make_net():
   with open('/home/dlg/train.prototxt', 'w') as f:
      f.write(str(lenet('mnist/mnist_train_lmdb', 64)))
   with open('/home/dlg/test.prototxt', 'w') as f:
      f.write(str(lenet('mnist/mnist_test_lmdb', 100)))

if __name__ =='__main__':
    make_net()
```

alex net示例

```Python
from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(lmdb, batch_size=256, include_acc=False):
    data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True))

    # the net itself
    conv1, relu1 = conv_relu(data, 11, 96, stride=4)
    pool1 = max_pool(relu1, 3, stride=2)
    norm1 = L.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75)
    conv2, relu2 = conv_relu(norm1, 5, 256, pad=2, group=2)
    pool2 = max_pool(relu2, 3, stride=2)
    norm2 = L.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75)
    conv3, relu3 = conv_relu(norm2, 3, 384, pad=1)
    conv4, relu4 = conv_relu(relu3, 3, 384, pad=1, group=2)
    conv5, relu5 = conv_relu(relu4, 3, 256, pad=1, group=2)
    pool5 = max_pool(relu5, 3, stride=2)
    fc6, relu6 = fc_relu(pool5, 4096)
    drop6 = L.Dropout(relu6, in_place=True)
    fc7, relu7 = fc_relu(drop6, 4096)
    drop7 = L.Dropout(relu7, in_place=True)
    fc8 = L.InnerProduct(drop7, num_output=1000)
    loss = L.SoftmaxWithLoss(fc8, label)

    if include_acc:
        acc = L.Accuracy(fc8, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)

def make_net():
    with open('train.prototxt', 'w') as f:
        print(caffenet('/path/to/caffe-train-lmdb'), file=f)

    with open('test.prototxt', 'w') as f:
        print(caffenet('/path/to/caffe-val-lmdb', batch_size=50, include_acc=True), file=f)

if __name__ == '__main__':
    make_net()
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)