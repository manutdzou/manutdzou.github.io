---
layout: post
title: ssd开发
category: 科研
tags: 深度学习
keywords: source code
description: 
---

# SSD开发

在我的[github](https://github.com/manutdzou/KITTI_SSD)使用SSD检测算法在KITTI数据集上训练了一个车辆行人的模型，下面列出一些修改的小模块。

## 训练好的模型如何测试

SSD给出了一个VideoData的层用于读取摄像头视频进行检测，现在修改这个layer使它支持本地视频检测，首先需要修改的是caffe.proto里的layer定义

```
message VideoDataParameter{
  enum VideoType {
    WEBCAM = 0;
    LOCAL_SOURCE = 1;
  }
  optional VideoType video_type = 1 [default = WEBCAM];
  optional int32 device_id = 2 [default = 0];
  optional string source = 3;
}
```

定义一个本地视频读取VideoType的LOCAL_SOURCE和对应的路径optional string source = 3，这样test时候VideoData使用video_tye为LOCAL_SOURCE并指定source路径并修改部分源码就能支持本地视频检测。

比如

```
layer {
  name: "data"
  type: "VideoData"
  top: "data"
  transform_param {
    mean_value: 104
    mean_value: 117
    mean_value: 123
    resize_param {
      prob: 1
      resize_mode: WARP
      height: 600
      width: 600
      interp_mode: LINEAR
    }
  }
  data_param {
    batch_size: 1
  }
  video_data_param {
    video_type: WEBCAM
    device_id: 0
  }
}
```

读取摄像头视频

```
layer {
  name: "data"
  type: "VideoData"
  top: "data"
  transform_param {
    mean_value: 104
    mean_value: 117
    mean_value: 123
    resize_param {
      prob: 1
      resize_mode: WARP
      height: 600
      width: 600
      interp_mode: LINEAR
    }
  }
  data_param {
    batch_size: 1
  }
  video_data_param {
    video_type: LOCAL_SOURCE
    #device_id: 0
    source: "/home/bsl/Debug/ssd_caffe/04050833_2639.MP4"
  }
}
```

读取本地视频

下面给出修改后VideoData的源码

```C++
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

#include <stdint.h>
#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
VideoDataLayer<Dtype>::VideoDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param) {
}

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer() {
  this->StopInternalThread();
  if (video_type_ == VideoDataParameter_VideoType_WEBCAM) {
    cap_.release();
  }
//添加一个LOCAL_SOURCE的析构函数
  else if (video_type_ == VideoDataParameter_VideoType_LOCAL_SOURCE) {
    cap_.release();
  }
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const VideoDataParameter& video_data_param =
      this->layer_param_.video_data_param();
  video_type_ = video_data_param.video_type();

  vector<int> top_shape;
  if (video_type_ == VideoDataParameter_VideoType_WEBCAM) {
    const int device_id = video_data_param.device_id();
    if (!cap_.open(device_id)) {
      LOG(FATAL) << "Failed to open webcam: " << device_id;
    }
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img;
    cap_ >> cv_img;
    CHECK(cv_img.data) << "Could not load image from webcam!";
    // Use data_transformer to infer the expected blob shape from a cv_image.
    top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
  }
//添加一个LOCAL_SOURCE的初始化
  else if (video_type_ == VideoDataParameter_VideoType_LOCAL_SOURCE) {
    const string source = video_data_param.source();
    if (!cap_.open(source)) {
      LOG(FATAL) << "Failed to open source: "<<source ;
    }
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img;
    cap_ >> cv_img;
    CHECK(cv_img.data) << "Could not load image from source file!";
    // Use data_transformer to infer the expected blob shape from a cv_image.
    top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
  }
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  vector<int> top_shape;
  if (video_type_ == VideoDataParameter_VideoType_WEBCAM) {
    cv::Mat cv_img;
    cap_ >> cv_img;
    CHECK(cv_img.data) << "Could not load image from webcam!";
    // Use data_transformer to infer the expected blob shape from a cv_img.
    top_shape = this->data_transformer_->InferBlobShape(cv_img);
  }

  else if (video_type_ == VideoDataParameter_VideoType_LOCAL_SOURCE) {
    cv::Mat cv_img;
    cap_ >> cv_img;
    CHECK(cv_img.data) << "Could not load image from local source!";
    // Use data_transformer to infer the expected blob shape from a cv_img.
    top_shape = this->data_transformer_->InferBlobShape(cv_img);
  }

  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    if (video_type_ == VideoDataParameter_VideoType_WEBCAM) {
      cv::Mat cv_img;
      cap_ >> cv_img;
      CHECK(cv_img.data) << "Could not load image from webcam!";
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();
    }
    else if (video_type_ == VideoDataParameter_VideoType_LOCAL_SOURCE) {
      cv::Mat cv_img;
      cap_ >> cv_img;
      CHECK(cv_img.data) << "Could not load image from local source!";
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();
    }
    if (this->output_labels_) {
      top_label[item_id] = 0;
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
```

## 如何训练和测试

### 训练和测试

```Shell
cd /home/bsl/Debug/ssd_caffe
./build/tools/caffe train \
--solver="/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/solver.prototxt" \ 
--weights="/home/bsl/Debug/ssd_caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \#fine-tune model
--iterations=7481 \ #测试7481张图片
--gpu 0 2>&1 | tee /home/bsl/Debug/ssd_caffe/jobs/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150.log
```

solver文件

```
train_net: "/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/train.prototxt"
test_net: "/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/test.prototxt"
test_iter: 7481
test_interval: 10000
base_lr: 0.001
display: 10
max_iter: 60000
lr_policy: "step"
gamma: 0.1
momentum: 0.9
weight_decay: 0.0005
stepsize: 40000
snapshot: 40000
snapshot_prefix: "/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150"
solver_mode: GPU
device_id: 0
debug_info: false
snapshot_after_train: true
test_initialization: false
average_loss: 10
iter_size: 1
type: "SGD"
eval_type: "detection"
ap_version: "11point"
```

### 已有模型直接测试

```Shell
cd /home/bsl/Debug/ssd_caffe
./build/tools/caffe train \
--solver="/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/solver_test.prototxt" \
--weights="/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_iter_60000.caffemodel" \ #训练好的模型
--iterations=7481 \
--gpu 0 2>&1 | tee /home/bsl/Debug/ssd_caffe/jobs/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_TEST.log
```

solver文件

```
train_net: "/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/train.prototxt"
test_net: "/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/test.prototxt"
test_iter: 7481
test_interval: 10000
base_lr: 0.001
display: 10
max_iter: 0 #max_iter设为0后就不进行训练了直接进入测试网络阶段
lr_policy: "step"
gamma: 0.1
momentum: 0.9
weight_decay: 0.0005
stepsize: 40000
snapshot: 40000
snapshot_prefix: "/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150"
solver_mode: GPU
device_id: 0
debug_info: false
snapshot_after_train: true
test_initialization: false
average_loss: 10
iter_size: 1
type: "SGD"
eval_type: "detection"
ap_version: "11point"
```

## 如何导出python接口

由于ssd工程全在c++下编写，在使用时非常不方便，所以可以参考caffe的python重写一个ssd的接口，下面给出python接口：

检测图片

```Python
import sys
sys.path.append('/home/bsl/Debug/ssd_caffe/python/')#添加caffe的python接口
import caffe
import os
import numpy as np
import cv2
import time

#定义一个计时的类用于测试计算时间
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

caffe_root = "/home/bsl/Debug/ssd_caffe/"
if os.path.isfile(caffe_root + 'models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_iter_60000.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'CaffeNet not found'
model_def = caffe_root + 'models/VGGNet/KITTI/SSD_600x150/deploy.prototxt'
model_weights = caffe_root + 'models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_iter_60000.caffemodel'

net = caffe.Net(model_def,model_weights,caffe.TEST)
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
mu = np.array([104, 117, 123])
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
net.blobs['data'].reshape(1,3,150, 600)

test_image_path=caffe_root+'data/KITTI/training/data_object_image_2/training/image_2'
color=[(255,0,0),(0,255,0),(0,0,255)]
visualize_threshold=0.6
for parent, dirnames, filenames in os.walk(test_image_path):
    for filename in filenames:
        timer = Timer()
        img_path=caffe_root + 'data/KITTI/training/data_object_image_2/training/image_2/'+filename
        result_path=caffe_root + 'data/KITTI/results/'+filename
        image = caffe.io.load_image(img_path)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        timer.tic()
        output = net.forward() #detectors 1*1*N*7 N*(image-id, label, confidence, xmin, ymin, xmax, ymax)
        timer.toc()
        shape=output['detection_out'].shape
        detectors=output['detection_out'].reshape(shape[2],shape[3])
        #visualize
        img=cv2.imread(img_path)
        size=img.shape
        for i in xrange(detectors.shape[0]):
            if detectors[i][2]>=visualize_threshold:
                xmin=int(detectors[i][3]*size[1])
                ymin=int(detectors[i][4]*size[0])
                xmax=int(detectors[i][5]*size[1])
                ymax=int(detectors[i][6]*size[0])
                label=detectors[i][1]
                rect_start=(xmin,ymin)
                rect_end=(xmax,ymax)
                cv2.rectangle(img, rect_start, rect_end, color[int(label-1)], 2)
        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        cv2.imwrite(result_path,img)
        print ('Detection took {:.3f}s').format(timer.total_time)
```

检测视频

```Python
import sys
sys.path.append('/home/bsl/Debug/ssd_caffe/python/'）
import caffe
import os
import numpy as np
import cv2
import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
caffe_root = "/home/bsl/Debug/ssd_caffe/"
if os.path.isfile(caffe_root + 'models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_iter_60000.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'CaffeNet not found'
model_def = caffe_root + 'models/VGGNet/KITTI/SSD_600x150/deploy_large.prototxt'
model_weights = caffe_root + 'models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_iter_60000.caffemodel'

net = caffe.Net(model_def,model_weights,caffe.TEST)
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
mu = np.array([104, 117, 123])
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
net.blobs['data'].reshape(1,3,270, 480)#可作适当调整

test_image_path=caffe_root+'data/KITTI/training/data_object_image_2/testing/image_2'
color=[(255,0,0),(0,255,0),(0,0,255)]
visualize_threshold=0.6


dir_name='04041652_2624.MP4'
dir_root=os.path.join(caffe_root,dir_name)
videoCapture = cv2.VideoCapture(dir_root)
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
#fps=25
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))/2,int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))/2)
success, frame = videoCapture.read()


#cv2.cv.CV_FOURCC('I','4','2','0') avi
#cv2.cv.CV_FOURCC('P','I','M','1') avi
#cv2.cv.CV_FOURCC('M','J','P','G') avi
#cv2.cv.CV_FOURCC('T','H','E','O') ogv
#cv2.cv.CV_FOURCC('F','L','V','1') flv
video=cv2.VideoWriter(dir_name, cv2.cv.CV_FOURCC('M','J','P','G'), int(fps),size)

while success:
    timer=Timer()
    image=frame/255.
    #image = caffe.io.load_image(img_path)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    timer.tic()
    output = net.forward() #detectors 1*1*N*7 N*(image-id, label, confidence, xmin, ymin, xmax, ymax)
    timer.toc()
    shape=output['detection_out'].shape
    detectors=output['detection_out'].reshape(shape[2],shape[3])
    #visualize
    img=cv2.resize(frame,(size[1],size[0]))
    for i in xrange(detectors.shape[0]):
        if detectors[i][2]>=visualize_threshold:
            xmin=int(detectors[i][3]*size[1])
            ymin=int(detectors[i][4]*size[0])
            xmax=int(detectors[i][5]*size[1])
            ymax=int(detectors[i][6]*size[0])
            label=detectors[i][1]
            rect_start=(xmin,ymin)
            rect_end=(xmax,ymax)
            cv2.rectangle(img, rect_start, rect_end, color[int(label-1)], 2)
    cv2.imshow('image',img)
    cv2.waitKey(1)
    print ('Detection took {:.3f}s').format(timer.total_time)
    success, frame = videoCapture.read()
```

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)
