---
layout: post
title: tensorflow onnx tensort
category: 科研
tags: 
keywords: 
description:
---

# 配置环境

首先我的环境是ubuntu1804+cuda10.0+tensort7.0.0.11+tensorflow1.15.0

## 安装onnx

```
sudo apt-get install protobuf-compiler libprotoc-dev

git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
python setup.py install

pip install pytest nbval
pytest
```

## 安装tensorflow2onnx

```
git clone https://github.com/onnx/tensorflow-onnx.git
python setup.py install

pip install onnxruntime-gpu
```

首先将tensorflow的模型固化成pb,注意对于模型的input要预先设置好shape(否则转onnx时候可能会出错)，记录下input和output的name

然后将pb转成onnx，这里需要注意版本问题，有些tensorflow的op只有高版本的tf2onnx的和高opset的才支持

这里我使用：

tf2onnx.tfonnx: Using tensorflow=1.15.0, onnx=1.6.0, tf2onnx=1.6.0/342270
tf2onnx.tfonnx: Using opset <onnx, 7>
onnxruntime-gpu为1.1.0
protobuf为3.11.1

```
python -m tf2onnx.convert\
    --input yolov3_tiny.pb\
    --inputs input/input_data:0\
    --outputs pred_mbbox/concat_3:0,pred_lbbox/concat_3:0\
    --opset 7\
    --output yolov3_tiny.onnx\
    --verbose --fold_const
```

## 安装TensorRT

首先下载TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz

```
tar xzvf TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz

#to profile
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/dfsdata2/jinyi_data/Model_Deploy/TensorRT-7.0.0.11/lib

cd TensorRT-7.0.0.11/python
pip install tensorrt-7.0.0.11-cp36-none-linux_x86_64.whl

cd TensorRT-7.0.0.11/uff
pip install uff-0.6.5-py2.py3-none-any.whl

cd TensorRT-7.0.0.11/graphsurgeon
pip install graphsurgeon-0.4.1-py2.py3-none-any.whl
```

## 安装onnx2tensort

```
git clone https://github.com/onnx/onnx-tensorrt
cd onnx-tensorrt
git submodule init
git submodule update
cmake . -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include -DTENSORRT_ROOT=/dfsdata2/jinyi_data/Model_Deploy/TensorRT-7.0.0.11 -DGPU_ARCHS="53"
make -j8
sudo make instal
```

将onnx转化为trt，注意，这里onnx使用opset 7 有些可能会失败

```
onnx2trt yolov3_tiny.onnx -o yolov3_tiny.trt
```



看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)