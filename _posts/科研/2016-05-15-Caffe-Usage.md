---
layout: post
title: Caffe使用教程（下）
category: 科研
tags: 深度学习
keywords: Caffe使用教程
description: 
---
# Caffe使用教程

## 运行caffe自带的两个简单例子

为了程序的简洁，在caffe中是不带练习数据的，因此需要自己去下载。但在caffe根目录下的data文件夹里，作者已经为我们编写好了下载数据的脚本文件，我们只需要联网，运行这些脚本文件就行了。

注意：在caffe中运行所有程序，都必须在根目录下进行，否则会出错

### 1、mnist实例

mnist是一个手写数字库，由DL大牛Yan LeCun进行维护。mnist最初用于支票上的手写数字识别, 现在成了DL的入门练习库。征对mnist识别的专门模型是Lenet，算是最早的cnn模型了。

mnist数据训练样本为60000张，测试样本为10000张，每个样本为28*28大小的黑白图片，手写数字为0-9，因此分为10类。

首先下载mnist数据，假设当前路径为caffe根目录

```
sudo sh data/mnist/get_mnist.sh
```

运行成功后，在 data/mnist/目录下有四个文件：

train-images-idx3-ubyte:  训练集样本 (9912422 bytes) 

train-labels-idx1-ubyte:  训练集对应标注 (28881 bytes) 

t10k-images-idx3-ubyte:   测试集图片 (1648877 bytes) 

t10k-labels-idx1-ubyte:   测试集对应标注 (4542 bytes)

这些数据不能在caffe中直接使用，需要转换成LMDB数据

```
sudo sh examples/mnist/create_mnist.sh
```

如果想运行leveldb数据，请运行 examples/siamese/ 文件夹下面的程序。 examples/mnist/ 文件夹是运行lmdb数据

转换成功后，会在 examples/mnist/目录下，生成两个文件夹，分别是mnist_train_lmdb和mnist_test_lmdb，里面存放的data.mdb和lock.mdb，就是我们需要的运行数据。

接下来是修改配置文件，如果你有GPU且已经完全安装好，这一步可以省略，如果没有，则需要修改solver配置文件。

需要的配置文件有两个，一个是lenet_solver.prototxt，另一个是train_lenet.prototxt.

首先打开lenet_solver_prototxt

```
sudo vi examples/mnist/lenet_solver.prototxt
```

根据需要，在max_iter处设置最大迭代次数，以及决定最后一行solver_mode,是否要改成CPU

保存退出后，就可以运行这个例子了

```
sudo time sh examples/mnist/train_lenet.sh
```

CPU运行时候大约13分钟，GPU运行时间大约4分钟，GPU+cudnn运行时候大约40秒，精度都为99%左右

### 2、cifar10实例

cifar10数据训练样本50000张，测试样本10000张，每张为32*32的彩色三通道图片，共分为10类。

下载数据：

```
sudo sh data/cifar10/get_cifar10.sh
```

运行成功后，会在 data/cifar10/文件夹下生成一堆bin文件

转换数据格式为lmdb：

```
sudo sh examples/cifar10/create_cifar10.sh
```

转换成功后，会在 examples/cifar10/文件夹下生成两个文件夹，cifar10_train_lmdb和cifar10_test_lmdb, 里面的文件就是我们需要的文件。

为了节省时间，我们进行快速训练（train_quick)，训练分为两个阶段，第一个阶段（迭代4000次）调用配置文件cifar10_quick_solver.prototxt, 学习率（base_lr)为0.001

第二阶段（迭代1000次）调用配置文件cifar10_quick_solver_lr1.prototxt, 学习率(base_lr)为0.0001

前后两个配置文件就是学习率(base_lr)和最大迭代次数（max_iter)不一样，其它都是一样。如果你对配置文件比较熟悉以后，实际上是可以将两个配置文件合二为一的，设置lr_policy为multistep就可以了。

``` 
base_lr: 0.001
momentum: 0.9
weight_decay: 0.004
lr_policy: "multistep"
gamma: 0.1
stepvalue: 4000
stepvalue: 5000
```
 
运行例子：

```
sudo time sh examples/cifar10/train_quick.sh
```

GPU+cudnn大约45秒左右，精度75%左右。

## 命令行解析


caffe的运行提供三种接口：c++接口（命令行）、python接口和matlab接口。本文先对命令行进行解析，后续会依次介绍其它两个接口。

caffe的c++主程序（caffe.cpp)放在根目录下的tools文件夹内, 当然还有一些其它的功能文件，如：convert_imageset.cpp, train_net.cpp, test_net.cpp等也放在这个文件夹内。经过编译后，这些文件都被编译成了可执行文件，放在了 ./build/tools/ 文件夹内。因此我们要执行caffe程序，都需要加 ./build/tools/ 前缀。

如：

```
sudo sh ./build/tools/caffe train --solver=examples/mnist/train_lenet.sh
```

caffe程序的命令行执行格式如下：

caffe <command> <args>

其中的<command>有这样四种：

•	train

•	test

•	device_query

•	time

对应的功能为：

train----训练或finetune模型（model),

test-----测试模型

device_query---显示gpu信息

time-----显示程序执行时间

其中的<args>参数有：

•	-solver

•	-gpu

•	-snapshot

•	-weights

•	-iteration

•	-model

•	-sighup_effect

•	-sigint_effect

注意前面有个-符号。对应的功能为：

-solver：必选参数。一个protocol buffer类型的文件，即模型的配置文件。如：

``` 
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt
```

-gpu: 可选参数。该参数用来指定用哪一块gpu运行，根据gpu的id进行选择，如果设置为'-gpu all'则使用所有的gpu运行。如使用第二块gpu运行：

```
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2
```

-snapshot:可选参数。该参数用来从快照（snapshot)中恢复训练。可以在solver配置文件设置快照，保存solverstate。如：

```
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -snapshot examples/mnist/lenet_iter_5000.solverstate
```

-weights:可选参数。用预先训练好的权重来fine-tuning模型，需要一个caffemodel，不能和-snapshot同时使用。如：

```
./build/tools/caffe train -solver examples/finetuning_on_flickr_style/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
```

-iterations: 可选参数，迭代次数，默认为50。 如果在配置文件文件中没有设定迭代次数，则默认迭代50次。

-model:可选参数，定义在protocol buffer文件中的模型。也可以在solver配置文件中指定。

-sighup_effect：可选参数。用来设定当程序发生挂起事件时，执行的操作，可以设置为snapshot, stop或none, 默认为snapshot

-sigint_effect: 可选参数。用来设定当程序发生键盘中止事件时（ctrl+c), 执行的操作，可以设置为snapshot, stop或none, 默认为stop
 
刚才举例了一些train参数的例子，现在我们来看看其它三个<command>：

test参数用在测试阶段，用于最终结果的输出，要模型配置文件中我们可以设定需要输入accuracy还是loss. 假设我们要在验证集中验证已经训练好的模型，就可以这样写

```
./build/tools/caffe test -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 100
```

这个例子比较长，不仅用到了test参数，还用到了-model, -weights, -gpu和-iteration四个参数。意思是利用训练好了的权重（-weight)，输入到测试模型中(-model)，用编号为0的gpu(-gpu)测试100次(-iteration)。

time参数用来在屏幕上显示程序运行时间。如：

```
./build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10
```

这个例子用来在屏幕上显示lenet模型迭代10次所使用的时间。包括每次迭代的forward和backward所用的时间，也包括每层forward和backward所用的平均时间。

```
./build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -gpu 0
```

这个例子用来在屏幕上显示lenet模型用gpu迭代50次所使用的时间。

```
./build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 10
```

利用给定的权重，利用第一块gpu，迭代10次lenet模型所用的时间。

device_query参数用来诊断gpu信息。

```
./build/tools/caffe device_query -gpu 0
```

最后，我们来看两个关于gpu的例子

```
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 0,1
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -gpu all
```

这两个例子表示： 用两块或多块GPU来平行运算，这样速度会快很多。但是如果你只有一块或没有gpu, 就不要加-gpu参数了，加了反而慢。

最后，在linux下，本身就有一个time命令，因此可以结合进来使用，因此我们运行mnist例子的最终命令是(一块gpu)：

```
sudo time ./build/toos/caffe train -solver examples/mnist/lenet_solver.prototxt
```

## 图像数据转换成db（leveldb/lmdb)文件

在深度学习的实际应用中，我们经常用到的原始数据是图片文件，如jpg,jpeg,png,tif等格式的，而且有可能图片的大小还不一致。而在caffe中经常使用的数据类型是lmdb或leveldb，因此就产生了这样的一个问题：如何从原始图片文件转换成caffe中能够运行的db（leveldb/lmdb)文件？

在caffe中，作者为我们提供了这样一个文件：convert_imageset.cpp，存放在根目录下的tools文件夹下。编译之后，生成对应的可执行文件放在 buile/tools/ 下面，这个文件的作用就是用于将图片文件转换成caffe框架中能直接使用的db文件。

该文件的使用格式：
 
convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME

需要带四个参数：

FLAGS: 图片参数组，后面详细介绍

ROOTFOLDER/: 图片存放的绝对路径，从linux系统根目录开始

LISTFILE: 图片文件列表清单，一般为一个txt文件，一行一张图片

DB_NAME: 最终生成的db文件存放目录

如果图片已经下载到本地电脑上了，那么我们首先需要创建一个图片列表清单，保存为txt

本文以caffe程序中自带的图片为例，进行讲解，图片目录是  example/images/, 两张图片，一张为cat.jpg, 另一张为fish_bike.jpg，表示两个类别。

我们创建一个sh脚本文件，调用linux命令来生成图片清单：

```
sudo vi examples/images/create_filelist.sh
```

编辑这个文件,输入下面的代码并保存

```Shell
# /usr/bin/env sh
DATA=examples/images
echo "Create train.txt..."
rm -rf $DATA/train.txt
find $DATA -name *cat.jpg | cut -d '/' -f3 | sed "s/$/ 1/">>$DATA/train.txt
find $DATA -name *bike.jpg | cut -d '/' -f3 | sed "s/$/ 2/">>$DATA/tmp.txt
cat $DATA/tmp.txt>>$DATA/train.txt
rm -rf $DATA/tmp.txt
echo "Done.."
```

这个脚本文件中，用到了rm,find, cut, sed,cat等linux命令。

rm: 删除文件

find: 寻找文件

cut: 截取路径

sed: 在每行的最后面加上标注。本例中将找到的*cat.jpg文件加入标注为1，找到的*bike.jpg文件加入标注为2

cat: 将两个类别合并在一个文件里。

最终生成如下的一个train.txt文件：

```
cat.jpg 1
fish-bike.jpg 2
```

当然，图片很少的时候，手动编写这个列表清单文件就行了。但图片很多的情况，就需要用脚本文件来自动生成了。在以后的实际应用中，还需要生成相应的val.txt和test.txt文件，方法是一样的。

生成的这个train.txt文件，就可以作为第三个参数，直接使用了。

接下来，我们来了解一下FLAGS这个参数组，有些什么内容：

-gray: 是否以灰度图的方式打开图片。程序调用opencv库中的imread()函数来打开图片，默认为false

-shuffle: 是否随机打乱图片顺序。默认为false

-backend:需要转换成的db文件格式，可选为leveldb或lmdb,默认为lmdb

-resize_width/resize_height: 改变图片的大小。在运行中，要求所有图片的尺寸一致，因此需要改变图片大小。 程序调用opencv库的resize（）函数来对图片放大缩小，默认为0，不改变

-check_size: 检查所有的数据是否有相同的尺寸。默认为false,不检查

-encoded: 是否将原图片编码放入最终的数据中，默认为false

-encode_type: 与前一个参数对应，将图片编码为哪一个格式：‘png','jpg'......

好了，知道这些参数后，我们就可以调用命令来生成最终的lmdb格式数据了

由于参数比较多，因此我们可以编写一个sh脚本来执行命令：

首先，创建sh脚本文件：

```
sudo vi examples/images/create_lmdb.sh
```

编辑，输入下面的代码并保存

```Shell
#!/usr/bin/en sh
DATA=examples/images
rm -rf $DATA/img_train_lmdb
build/tools/convert_imageset --shuffle \
--resize_height=256 --resize_width=256 \
/home/xxx/caffe/examples/images/ $DATA/train.txt  $DATA/img_train_lmdb
```

设置参数-shuffle,打乱图片顺序。设置参数-resize_height和-resize_width将所有图片尺寸都变为256*256.

/home/xxx/caffe/examples/images/ 为图片保存的绝对路径。

最后，运行这个脚本文件

```
sudo sh examples/images/create_lmdb.sh
```

就会在examples/images/ 目录下生成一个名为 img_train_lmdb的文件夹，里面的文件就是我们需要的db文件了。

## 数据可视化环境（python接口)配置

caffe程序是由c++语言写的，本身是不带数据可视化功能的。只能借助其它的库或接口，如opencv, python或matlab。大部分人使用python接口来进行可视化，因为python出了个比较强大的东西：ipython notebook, 现在的最新版本改名叫jupyter notebook，它能将python代码搬到浏览器上去执行，以富文本方式显示，使得整个工作可以以笔记的形式展现、存储，对于交互编程、学习非常方便。 

### 一、安装python和pip

一般linux系统都自带python，所以不需要安装。如果没有的，安装起来也非常方便。安装完成后，可用version查看版本

```
python --version
```

pip是专门用于安装python各种依赖库的，所以我们这里安装一下pip1.5.6

先用链接下载安装包 https://pypi.python.org/packages/source/p/pip/pip-1.5.6.tar.gz，然后解压，里面有一个setup.py的文件，执行这个文件就可以安装pip了

```
sudo python setup.py install
```

有些电脑可能会提示 no moudle name setuptools 的错误，这是没有安装setuptools的原因。那就需要先安装一下setuptools, 到https://pypi.python.org/packages/source/s/setuptools/setuptools-19.2.tar.gz 下载安装包setuptools-19.2.tar.gz，然后解压执行

```
sudo python setup.py install
```

就要以安装setuptools了，然后再回头去重新安装pip。执行的代码都是一样的，只是在不同的目录下执行。

### 二、安装pyhon接口依赖库

在caffe根目录的python文件夹下，有一个requirements.txt的清单文件，上面列出了需要的依赖库，按照这个清单安装就可以了。

在安装scipy库的时候，需要fortran编译器（gfortran)，如果没有这个编译器就会报错，因此，我们可以先安装一下。

首先回到caffe的根目录，然后执行安装代码：

```
cd ~/caffe
sudo apt-get install gfortran
for req in $(cat requirements.txt); do sudo pip install $req; done
```

安装完成以后，我们可以执行：

```
sudo pip install -r python/requirements.txt
```

就会看到，安装成功的，都会显示Requirement already satisfied, 没有安装成功的，会继续安装。

### 三、编译python接口

首先，将caffe根目录下的python文件夹加入到环境变量

打开配置文件bashrc

```
sudo vi ~/.bashrc
```

在最后面加入

```
export PYTHONPATH=/home/xxx/caffe/python:$PYTHONPATH
```

注意 /home/xxx/caffe/python 是我的路径，所有这个地方每个人都不同，需要修改

保存退出，更新配置文件

```
sudo ldconfig
```

最后进行编译：

```
sudo make pycaffe
```
编译成功后，不能重复编译，否则会提示 Nothing to be done for "pycaffe"的错误。

可以从两个方面查看是否编译成功：

1、查看 python/caffe/ 目录下，除了原先的一堆py后缀文件，现在多出了一堆pyc后缀文件

2、进入python环境，进行import操作

```
python
>>> import caffe
```

如果没有提示错误，则编译成功。

### 四、安装jupyter

学会了python还不行，还得学习一下ipython，后者更加方便快捷，更有自动补全功能。而ipython notebook是ipython的最好展现方式。最新的版本改名为jupyter notebook,我们先来安装一下。

```
sudo pip install jupyter
```

安装成功后，运行notebook

```
jupyter notebook
```

就会在浏览器中打开notebook,  点击右上角的New-python2, 就可以新建一个网页一样的文件，扩展名为ipynb。在这个网页上，我们就可以像在命令行下面一样运行python代码了。输入代码后，按shift+enter运行，更多的快捷键，可点击上方的help-Keyboard shortcuts查看，或者先按esc退出编辑状态，再按h键查看。

## 初识数据可视化

首先将caffe的根目录作为当前目录，然后加载caffe程序自带的小猫图片，并显示。

图片大小为360x480，三通道

```
In [1]:
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import caffe
caffe_root='/home/xxx/caffe/'
import os,sys
os.chdir(caffe_root)
sys.path.insert(0,caffe_root+'python')
im = caffe.io.load_image('examples/images/cat.jpg')
print im.shape
plt.imshow(im)
plt.axis('off')
 
(360, 480, 3)
Out[1]:
(-0.5, 479.5, 359.5, -0.5)
```

 
打开examples/net_surgery/conv.prototxt文件，修改两个地方

一是将input_shape由原来的是（1，1，100，100）修改为(1,3,100,100),即由单通道灰度图变为三通道角色图。

二是将过滤器个数(num_output)修改为16，当然保持原来的数不变也行。

其它地方不变，修改后的prototxt如下：只有一个卷积层

```
In [2]:
! cat examples/net_surgery/conv.prototxt
 
# Simple single-layer network to showcase editing model parameters.
name: "convolution"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 100
  dim: 100
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 16
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
```

加载单一卷积层为训练网络，并将输入的原始图片数据变为blob数据。

根据加载的配置文件，在这里，我们也可以反过来从blob中提取出原始数据，并进行显示。

显示的时候要注意各维的顺序，如blobs的顺序是(1,3,360,480),从前往后分别表示1张图片，3三个通道，

图片大小为360x480，需要调用transpose改变为(360,480,3)才能正常显示。

其中用data[0]表示第一张图片，下标从0开始，此例只有一张图片，因此只能是data[0].

分别用data[0,0],data[0,1]和data[0,3]表示该图片的三个通道。

```
In [3]:
net = caffe.Net('examples/net_surgery/conv.prototxt', caffe.TEST)
im_input=im[np.newaxis,:,:,:].transpose(0,3,1,2)
print "data-blobs:",im_input.shape
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input
plt.imshow(net.blobs['data'].data[0].transpose(1,2,0))
plt.axis('off')
 
data-blobs: (1, 3, 360, 480)
Out[3]:
(-0.5, 479.5, 359.5, -0.5)
```


编写一个show_data函数来显示数据

```
In [4]:
plt.rcParams['image.cmap'] = 'gray'

def show_data(data,head,padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.title(head)
    plt.imshow(data)
    plt.axis('off')
```

从blobs数据中将原始图片提取出来，并分别显示不同的通道图

```
In [5]:
print "data-blobs:",net.blobs['data'].data.shape
show_data(net.blobs['data'].data[0],'origin images')
 
data-blobs: (1, 3, 360, 480)
```


调用forward()执行卷积操作，blobs数据发生改变。由原来的(1,3,360,480)变为（1，16，356，476）。

并初始化生成了相应的权值，权值数据为(16,3,5,5)。

最后调用两次show_data来分别显示权值和卷积过滤后的16通道图片。

```
In [6]:
net.forward()
print "data-blobs:",net.blobs['data'].data.shape
print "conv-blobs:",net.blobs['conv'].data.shape
print "weight-blobs:",net.params['conv'][0span>].data.shape
show_data(net.params['conv'][0].data[:,0],'conv weights(filter)')
show_data(net.blobs['conv'].data[0],'post-conv images')
 
data-blobs: (1, 3, 360, 480)
conv-blobs: (1, 16, 356, 476)
weight-blobs: (16, 3, 5, 5)
```

## 计算图片数据的均值

图片减去均值后，再进行训练和测试，会提高速度和精度。因此，一般在各种模型中都会有这个操作。

那么这个均值怎么来的呢，实际上就是计算所有训练样本的平均值，计算出来后，保存为一个均值文件，在以后的测试中，就可以直接使用这个均值来相减，而不需要对测试图片重新计算。

### 一、二进制格式的均值计算

caffe中使用的均值数据格式是binaryproto, 作者为我们提供了一个计算均值的文件compute_image_mean.cpp，放在caffe根目录下的tools文件夹里面。编译后的可执行体放在 build/tools/ 下面，我们直接调用就可以了

```
sudo build/tools/compute_image_mean examples/mnist/mnist_train_lmdb examples/mnist/mean.binaryproto
```

带两个参数：
第一个参数：examples/mnist/mnist_train_lmdb， 表示需要计算均值的数据，格式为lmdb的训练数据。
第二个参数：examples/mnist/mean.binaryproto， 计算出来的结果保存文件。

### 二、python格式的均值计算

如果我们要使用python接口，或者我们要进行特征可视化，可能就要用到python格式的均值文件了。首先，我们用lmdb格式的数据，计算出二进制格式的均值，然后，再转换成python格式的均值。

我们可以编写一个python脚本来实现：

``` 
#!/usr/bin/env python
import numpy as np
import sys,caffe

if len(sys.argv)!=3:
    print "Usage: python convert_mean.py mean.binaryproto mean.npy"
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
bin_mean = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(bin_mean)
arr = np.array( caffe.io.blobproto_to_array(blob) )
npy_mean = arr[0]
np.save( sys.argv[2] , npy_mean )
```

将这个脚本保存为convert_mean.py

调用格式为：

```
sudo python convert_mean.py mean.binaryproto mean.npy
```

其中的 mean.binaryproto 就是经过前面步骤计算出来的二进制均值。

mean.npy就是我们需要的python格式的均值。

## caffemodel可视化

通过前面的学习，我们已经能够正常训练各种数据了。设置好solver.prototxt后，我们可以把训练好的模型保存起来，如lenet_iter_10000.caffemodel。 训练多少次就自动保存一下，这个是通过snapshot进行设置的，保存文件的路径及文件名前缀是由snapshot_prefix来设定的。这个文件里面存放的就是各层的参数，即net.params，里面没有数据(net.blobs)。顺带还生成了一个相应的solverstate文件，这个和caffemodel差不多，但它多了一些数据，如模型名称、当前迭代次数等。两者的功能不一样，训练完后保存起来的caffemodel，是在测试阶段用来分类的，而solverstate是用来恢复训练的，防止意外终止而保存的快照（有点像断点续传的感觉)。

既然我们知道了caffemodel里面保存的就是模型各层的参数，因此我们可以把这些参数提取出来，进行可视化，看一看究竟长什么样。
 
我们先训练cifar10数据（mnist也可以），迭代10000次，然后将训练好的 model保存起来，名称为my_iter_10000.caffemodel，然后使用jupyter notebook 来进行可视化。

```
In [1]:
import numpy as np
import matplotlib.pyplot as plt
import os,sys,caffe
%matplotlib inline
In [2]:
caffe_root='/home/lee/caffe/'
os.chdir(caffe_root)
sys.path.insert(0,caffe_root+'python')
In [3]:
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
 　　　　　　设置网络模型，并显示该模型中各层名称和参数的规模（注意此处是net.params, 而不是net.blobs)
In [4]:
net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_full.prototxt',
                caffe_root + 'examples/cifar10/my_iter_10000.caffemodel',
                caffe.TEST)
[(k, v[0].data.shape) for k, v in net.params.items()]
Out[4]:
[('conv1', (32, 3, 5, 5)),
 ('conv2', (32, 32, 5, 5)),
 ('conv3', (64, 32, 5, 5)),
 ('ip1', (10, 1024))]
```

cifar10训练的模型配置在文件cifar10_full.prototxt里面，共有三个卷积层和一个全连接层，参数规模如上所示。

```
In [5]:
#编写一个函数，用于显示各层的参数
def show_feature(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
In [6]:
# 第一个卷积层，参数规模为(32,3,5,5)，即32个5*5的3通道filter
weight = net.params["conv1"][0].data
print weight.shape
show_feature(weight.transpose(0, 2, 3, 1))
(32, 3, 5, 5)
```

参数有两种类型：权值参数和偏置项。分别用params["conv1"][0] 和params["conv1"][1] 表示 。

我们只显示权值参数，因此用params["conv1"][0] 

``` 
In [7]:
# 第二个卷积层的权值参数，共有32*32个filter,每个filter大小为5*5
weight = net.params["conv2"][0].data
print weight.shape
show_feature(weight.reshape(32**2, 5, 5))
 
(32, 32, 5, 5)

In [8]:
# 第三个卷积层的权值，共有64*32个filter,每个filter大小为5*5，取其前1024个进行可视化
weight = net.params["conv3"][0].data print weight.shape show_feature(weight.reshape(64*32, 5, 5)[:1024])
 
(64, 32, 5, 5)
``` 

## 绘制网络模型

python/draw_net.py, 这个文件，就是用来绘制网络模型的。也就是将网络模型由prototxt变成一张图片。

在绘制之前，需要先安装两个库

１、安装ＧraphViz

```
sudo apt-get install GraphViz
```

注意，这里用的是apt-get来安装，而不是pip.

2 、安装pydot

```
sudo pip install pydot
```

用的是pip来安装，而不是apt-get

安装好了，就可以调用脚本来绘制图片了

draw_net.py执行的时候带三个参数

第一个参数：网络模型的prototxt文件

第二个参数：保存的图片路径及名字

第二个参数：--rankdir=x , x 有四种选项，分别是LR, RL, TB, BT 。用来表示网络的方向，分别是从左到右，从右到左，从上到小，从下到上。默认为ＬＲ。

例：绘制Lenet模型

```
sudo python python/draw_net.py examples/mnist/lenet_train_test.prototxt netImage/lenet.png --rankdir=BT
```

例：绘制cifar10的模型

```
sudo python python/draw_net.py examples/cifar10/cifar10_full_train_test.prototxt netImage/cifar10.png --rankdir=BT
```

## 绘制loss和accuracy曲线

如同前几篇的可视化，这里采用的也是jupyter notebook来进行曲线绘制。

``` 
In [1]:
#加载必要的库
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import sys,os,caffe
#设置当前目录
caffe_root = '/home/bnu/caffe/' 
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
#设置求解器，和c++/caffe一样，需要一个solver配置文件。
In [2]:
# set the solver prototxt
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('examples/cifar10/cifar10_quick_solver.prototxt')
```

如果不需要绘制曲线，只需要训练出一个caffemodel, 直接调用solver.solve()就可以了。如果要绘制曲线，就需要把迭代过程中的值

保存下来，因此不能直接调用solver.solve(), 需要迭代。在迭代过程中，每迭代200次测试一次

```
In [5]:
%%time
niter =4000
test_interval = 200
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')
    
    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc
 
Iteration 0 testing... accuracy: 0.10000000149
Iteration 200 testing... accuracy: 0.419999986887
Iteration 400 testing... accuracy: 0.479999989271
Iteration 600 testing... accuracy: 0.540000021458
Iteration 800 testing... accuracy: 0.620000004768
Iteration 1000 testing... accuracy: 0.629999995232
Iteration 1200 testing... accuracy: 0.649999976158
Iteration 1400 testing... accuracy: 0.660000026226
Iteration 1600 testing... accuracy: 0.660000026226
Iteration 1800 testing... accuracy: 0.670000016689
Iteration 2000 testing... accuracy: 0.709999978542
Iteration 2200 testing... accuracy: 0.699999988079
Iteration 2400 testing... accuracy: 0.75
Iteration 2600 testing... accuracy: 0.740000009537
Iteration 2800 testing... accuracy: 0.769999980927
Iteration 3000 testing... accuracy: 0.75
Iteration 3200 testing... accuracy: 0.699999988079
Iteration 3400 testing... accuracy: 0.740000009537
Iteration 3600 testing... accuracy: 0.72000002861
Iteration 3800 testing... accuracy: 0.769999980927
CPU times: user 41.7 s, sys: 54.2 s, total: 1min 35s
Wall time: 1min 18s
```

绘制train过程中的loss曲线，和测试过程中的accuracy曲线。

```
In [6]:
print test_acc
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
 
[ 0.1         0.41999999  0.47999999  0.54000002  0.62        0.63
  0.64999998  0.66000003  0.66000003  0.67000002  0.70999998  0.69999999
  0.75        0.74000001  0.76999998  0.75        0.69999999  0.74000001
  0.72000003  0.76999998]
Out[6]:
<matplotlib.text.Text at 0x7fd1297bfcd0>
``` 

## 用训练好的caffemodel来进行分类

caffe程序自带有一张小猫图片，存放路径为caffe根目录下的 examples/images/cat.jpg, 如果我们想用一个训练好的caffemodel来对这张图片进行分类，那该怎么办呢？ 如果不用这张小猫图片，换一张别的图片，又该怎么办呢？如果学会了小猫图片的分类，那么换成其它图片，程序实际上是一样的。

开发caffe的贾大牛团队，利用imagenet图片和caffenet模型训练好了一个caffemodel,  供大家下载。要进行图片的分类，这个caffemodel是最好不过的了。所以，不管是用c++来进行分类，还是用python接口来分类，我们都应该准备这样三个文件：

### 1、caffemodel文件。 

可以直接在浏览器里输入地址下载，也可以运行脚本文件下载。下载地址为：http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

文件名称为：bvlc_reference_caffenet.caffemodel，文件大小为230M左右，为了代码的统一，将这个caffemodel文件下载到caffe根目录下的 models/bvlc_reference_caffenet/ 文件夹下面。也可以运行脚本文件进行下载：

```
sudo ./scripts/download_model_binary.py models/bvlc_reference_caffenet
```

### 2、均值文件。

有了caffemodel文件，就需要对应的均值文件，在测试阶段，需要把测试数据减去均值。这个文件我们用脚本来下载，在caffe根目录下执行：

```
sudo sh ./data/ilsvrc12/get_ilsvrc_aux.sh
```

执行并下载后，均值文件放在 data/ilsvrc12/ 文件夹里。

### 3、synset_words.txt文件

在调用脚本文件下载均值的时候，这个文件也一并下载好了。里面放的是1000个类的名称。

数据准备好了，我们就可以开始分类了，我们给大家提供两个版本的分类方法：

#### 一、c++方法

在caffe根目录下的 examples/cpp-classification/ 文件夹下面，有个classification.cpp文件，就是用来分类的。当然编译后，放在/build/examples/cpp_classification/ 下面

我们就直接运行命令：

```
sudo ./build/examples/cpp_classification/classification.bin \
  models/bvlc_reference_caffenet/deploy.prototxt \
  models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  data/ilsvrc12/imagenet_mean.binaryproto \
  data/ilsvrc12/synset_words.txt \
  examples/images/cat.jpg
```

命令很长，用了很多的\符号来换行。可以看出，从第二行开始就是参数，每行一个，共需要4个参数

运行成功后，输出top-5结果：

```
---------- Prediction for examples/images/cat.jpg ----------
0.3134 - "n02123045 tabby, tabby cat"
0.2380 - "n02123159 tiger cat"
0.1235 - "n02124075 Egyptian cat"
0.1003 - "n02119022 red fox, Vulpes vulpes"
0.0715 - "n02127052 lynx, catamount"
```

即有0.3134的概率为tabby cat, 有0.2380的概率为tiger cat ......

#### 二、python方法

python接口可以使用jupyter notebook来进行可视化操作，因此推荐使用这种方法。

在这里我就不用可视化了，编写一个py文件，命名为py-classify.py

```Python 
#coding=utf-8
#加载必要的库
import numpy as np

import sys,os

#设置当前目录
caffe_root = '/home/xxx/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)

net_file=caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
caffe_model=caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))

im=caffe.io.load_image(caffe_root+'examples/images/cat.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data',im)
out = net.forward()


imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
for i in np.arange(top_k.size):
    print top_k[i], labels[top_k[i]]
```

执行这个文件，输出：

```
281 n02123045 tabby, tabby cat
282 n02123159 tiger cat
285 n02124075 Egyptian cat
277 n02119022 red fox, Vulpes vulpes
287 n02127052 lynx, catamount
```

caffe开发团队实际上也编写了一个python版本的分类文件，路径为 python/classify.py
运行这个文件必需两个参数，一个输入图片文件，一个输出结果文件。而且运行必须在python目录下。假设当前目录是caffe根目录，则运行：

```
cd python
sudo python classify.py ../examples/images/cat.jpg result.npy
```

分类的结果保存为当前目录下的result.npy文件里面，是看不见的。而且这个文件有错误，运行的时候，会提示

Mean shape incompatible with input shape的错误。因此，要使用这个文件，我们还得进行修改：

1、修改均值计算：

定位到 

```
mean = np.load(args.mean_file)
```

这一行，在下面加上一行：

```
mean=mean.mean(1).mean(1)
```

则可以解决报错的问题。

2、修改文件，使得结果显示在命令行下：

定位到

```
# Classify.
    start = time.time()
    predictions = classifier.predict(inputs, not args.center_only)
    print("Done in %.2f s." % (time.time() - start))
```

这个地方，在后面加上几行，如下所示：
 
```
# Classify.
    start = time.time()
    predictions = classifier.predict(inputs, not args.center_only)
    print("Done in %.2f s." % (time.time() - start))
    imagenet_labels_filename = '../data/ilsvrc12/synset_words.txt'
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    top_k = predictions.flatten().argsort()[-1:-6:-1]
    for i in np.arange(top_k.size):
        print top_k[i], labels[top_k[i]]
```

就样就可以了。运行不会报错，而且结果会显示在命令行下面。

## 如何将别人训练好的model用到自己的数据上

caffe团队用imagenet图片进行训练，迭代30多万次，训练出来一个model。这个model将图片分为1000类，应该是目前为止最好的图片分类model了。

假设我现在有一些自己的图片想进行分类，但样本量太小，可能只有几百张，而一般深度学习都要求样本量在1万以上，因此训练出来的model精度太低，根本用不上，那怎么办呢？

那就用caffe团队提供给我们的model吧。

因为训练好的model里面存放的就是一些参数，因此我们实际上就是把别人预先训练好的参数，拿来作为我们的初始化参数，而不需要再去随机初始化了。图片的整个训练过程，说白了就是将初始化参数不断更新到最优的参数的一个过程，既然这个过程别人已经帮我们做了，而且比我们做得更好，那为什么不用他们的成果呢？

使用别人训练好的参数，必须有一个前提，那就是必须和别人用同一个network，因为参数是根据network而来的。当然，最后一层，我们是可以修改的，因为我们的数据可能并没有1000类，而只有几类。我们把最后一层的输出类别改一下，然后把层的名称改一下就可以了。最后用别人的参数、修改后的network和我们自己的数据，再进行训练，使得参数适应我们的数据，这样一个过程，通常称之为微调（fine tuning).

讲解整个微调训练过程。

### 一、下载model参数

可以直接在浏览器里输入地址下载，也可以运行脚本文件下载。下载地址为：http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

文件名称为：bvlc_reference_caffenet.caffemodel，文件大小为230M左右，为了代码的统一，将这个caffemodel文件下载到caffe根目录下的 models/bvlc_reference_caffenet/ 文件夹下面。也可以运行脚本文件进行下载：

```
sudo ./scripts/download_model_binary.py models/bvlc_reference_caffenet
```

### 二、准备数据

如果有自己的数据最好，如果没有，可以下载数据：http://pan.baidu.com/s/1MotUe

这些数据共有500张图片，分为大巴车、恐龙、大象、鲜花和马五个类，每个类100张。编号分别以0,1,2,3,4开头，各为一类。我从其中每类选出20张作为测试，其余80张作为训练。因此最终训练图片400张（放在train文件夹内，每个类一个子文件夹），测试图片100张（放在test文件夹内，每个类一个子文件夹）。

将图片下载下来后解压，放在一个文件夹内。比如我在当前用户根目录下创建了一个data文件夹，专门用来存放数据，因此我的训练图片路径为：/home/xxx/data/finetune/train

caffenet的网络配置文件，放在 caffe/models/bvlc_reference_caffenet/ 这个文件夹里面，名字叫train_val.prototxt。打开这个文件，将里面的内容复制到上图的Custom Network文本框里，然后进行修改，主要修改这几个地方：

#### 1、修改train阶段的data层为：

```
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "examples/finetune/mean.binaryproto"
    mirror: true
    crop_size: 227
  }
  data_param {
    source: "examples/finetune/finetune_train_lmdb"
    batch_size: 100
    backend: LMDB
  }

}
```

把均值文件（mean_file)、数据源文件(source)、批次大小(batch_size)和数据源格式（backend)这四项作相应的修改

2、修改test阶段的data层：

``` 
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_file: "examples/finetune/mean.binaryproto"
    mirror: false
    crop_size: 227
  }
   data_param {
    source: "examples/finetune/finetune_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
```

3、修改最后一个全连接层（fc8)：

```
layer {
  name: "fc8-my"               #原来为"fc8"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }
  inner_product_param {
    num_output: 5        #原来为"1000"
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
```

看注释的地方，就只有两个地方修改，其它不变。

设置好后，就可以开始微调了(fine tuning).

训练结果就是一个新的model，可以用来单张图片和多张图片测试。具体测试方法前一篇文章已讲过，在此就不重复了。

在此，将别人训练好的model用到我们自己的图片分类上，整个微调过程就是这样了。

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)
