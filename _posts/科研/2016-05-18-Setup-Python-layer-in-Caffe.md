---
layout: post
title: 在Caffe中建立Python layer
category: 科研
tags: 深度学习
keywords: 应用
description: 
---

# Caffe python layer

在Caffe中使用Python Layer

首先编译支持Python Layer的Caffe 

如果是首次编译，修改Caffe根目录下的Makefile.config, uncomment

```
WITH_PYTHON_LAYER:=1
```

如果已经编译过

```Shell
make clean
WITH_PYTHON_LAYER=1 make&& make pycaffe
```

# 使用Python Layer

假设要设置一个Euclidean Loss的Python层训练mnist数据，在caffe-master/examples/mnist里新建一个文件夹（也可以建在别处）python_tools，假设所有python定义层和训练文件都在这个文件夹下，首先第一个文件_init_paths.py主要将caffe和python layer导入PYTHONPATH，也可以加到系统环境中。

_init_paths.py内容如下：

```Python
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', '..', '..', 'python')
add_path(caffe_path)

# Add my python layer to PYTHONPATH
lib_path = osp.join(this_dir)
add_path(lib_path)
```

或者可以加入～/.bashrc中

```Shell
# caffe lib
export PYTHONPATH=/home/zou/caffe-master/python:$PYTHONPATH
# python layer lib
export PYTHONPATH=/home/zou/caffe-master/examples/mnist/python_tools:$PYTHONPATH
```

在python_tools中定义两个python层，第一个为one_hot.py，主要功能是将mnist数据中的label编码成one_hot形式，第二个为pyloss.py主要功能是将输出特征和label的one_hot编码计算loss将输出特征强制为label的one_hot形式，也保证该输出和原来的SoftmaxWithLoss 输出一致，方便Accuracy层测试。

lenet_train_test.prototxt如下:

移除原来的 SoftmaxWithLoss层，prototxt前面保持网络结构保持不变，后面添加如下2个python layer

```
layer {
type:"Python"
name:"one_hot"   
bottom: "label"
bottom: "ip2"
top: "one_hot_label"
python_param{
module: "one_hot"
layer: "label2one_hot"
}
}

layer{
type: "Python"
name: "loss"
top: "loss"
bottom: "ip2"
bottom: "one_hot_label"
python_param{
module: "pyloss"
layer: "EuclideanLossLayer"
}
loss_weight: 1
}
```

python层的格式如下：

```
layer{
type:"Python"
name:"XXXX"
top:"XXXX"
bottom:"XXXX"
python_param{
module: "layer"
#module的名字，通常是定义Layer的.py文件的文件名，需要在$PYTHONPATH下
layer: "layer_class"
#layer的名字---module中的类名
}
}
```

one_hot.py内容如下：

```Python
import caffe
import numpy as np

class label2one_hot(caffe.Layer):
    def setup(self,bottom,top):
        pass
    def forward(self,bottom,top):
        top[0].data[...]=np.zeros_like(bottom[1].data,dtype=np.float32)
        for i in range(bottom[0].data.size):
            ind=bottom[0].data[i]
            top[0].data[i][ind]=1
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass
    def reshape(self, bottom, top):
        top[0].reshape(bottom[1].num,bottom[1].channels)
        pass
```

主要定义类成员函数setup, reshape, forward和backward函数，分别由python_layer.hpp中PythonLayer类的LayerSetUp, Reshape, Forward_cpu和Backward_cpu封装调用，所以现有版本python layer只支持CPU计算，需要用GPU计算时还是需要用C++写CUDA, 表示哭晕在厕所啊。不过还有一种策略是利用theano库计算python层的GPU部分，然后将diff传到caffe的Blob中，还好看到了一点希望。

one_hot.py主要实现了将数据中的单个label转换成one_hot的编码形式，比如如果一个图片类别为7，则将该label转换为[0,0,0,0,0,0,1,0,0,0]. 在网络运行之前首先调用setup根据网络参数进行layer初始化，该层输入两个bottom分别为label和ip2（输入ip2主要为了获得特征尺寸信息，类似于Faster_rcnn中的一个python layer，不参与数值计算），输出one_hot_label。 reshape函数在forward之前调用将one_hot_label的尺寸初始化成和ip2即特征一样大小（如果未reshape将导致top[0]尺寸不固定，还有不理解为什么reshape只指定两维就可以申请一个4维的空间），forward函数根据label的值在one_hot_label中对应位置设为1作为输出。注意在每次网络前传过程中都需要将top[0].data重新初始化为全0，否则上一次前传过程中的值不会自动释放（原因不明），该层不需要反馈传播，无需定义backward。


Pyloss.py内容如下：

```Python
import caffe
import numpy as np

class EuclideanLossLayer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) !=2:
            raise exception("Need two inputs to compute distance")
    def reshape(self,bottom,top):
        if bottom[0].count !=bottom[1].count:
            raise exception("Inputs must have the same dimension.")
        self.diff=np.zeros_like(bottom[0].data,dtype=np.float32)
        top[0].reshape(1)
    def forward(self,bottom,top):
        self.diff[...]=bottom[0].data-bottom[1].data
        top[0].data[...]=np.sum(self.diff**2)/bottom[0].num / 2.
    def backward(self,top,propagate_down,bottom):
        for i in range(1):
            if not propagate_down[i]:
                continue
            if i==0:
                sign=1
            else:
                sign=-1
            bottom[i].diff[...]=sign*self.diff/bottom[i].num
```

Pyloss.py主要计算特征和one_hot类别的Euclidean loss。setup函数检查是否输入2个bottom，reshape函数检查两个bottom尺寸是否匹配，并初始化回传的diff。forward函数计算梯度值和loss，backward函数计算反传梯度敏感值（propagate_down这参数在哪里设置）。具体推导如下：

$$loss=\frac{1}{batch}\sum_{i}^{batch}\frac{1}{2}\left \| ip2_{i}-one-hot_{i} \right \|^{2}$$

$$\frac{\partial loss}{\partial ip2_{i}}=\frac{1}{batch}\left ( ip2_{i}-one-hot_{i} \right )$$

$$\frac{\partial loss}{\partial one-hot_{i}}=\frac{-1}{batch}\left ( ip2_{i}-one-hot_{i} \right )$$

由于one-hot是类别标签不需要回传梯度所以只需要计算第一个bottom的梯度，range设为1就行，第二个直接忽略（But why propagate[1]也是true）




训练网络

train.py内容如下：

```Python
import _init_paths
import caffe
import os
ROOT_DIR=os.getcwd()
solver_prototxt=os.path.join(ROOT_DIR,'..','..','..','examples', 'mnist','lenet_solver.prototxt')
output_dir=os.path.join(ROOT_DIR,'..','..','..','examples', 'mnist')
solver = caffe.SGDSolver(solver_prototxt)
max_iters=1000
while solver.iter < max_iters:
# Make one SGD update
    solver.step(1)
net = solver.net
filename = ('final' + '.caffemodel')
filename = os.path.join(output_dir, filename)
net.save(str(filename))
```

![1](/public/img/posts/Python layer/1.png)

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)
