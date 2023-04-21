---
layout: post
title: Training Faster RCNN with Online Hard Example Mining
category: 科研
tags: 深度学习
keywords: OHEM
description: 
---

# Training Faster RCNN with Online Hard Example Mining

OHEM的论文描述在https://arxiv.org/abs/1604.03540,主要思想是首先将所有ROI区域输入fast rcnn模块求得每个ROI的loss,对所有ROI的loss从大到小排序,选择前N个最大的loss的ROI样本进行训练网络更新fast rcnn的参数.故而叫做在线困难样本挖掘,因为只有每次迭代过程中才能确定哪些ROI区域作为训练样本,而训练过程主要选择loss最大的那些ROI.算法比较简单.作者主要实现了fast rcnn的OHEM,本文主要参考fast rcnn的OHEM实现faster rcnn的OHEM.

## model

RPN的model保持不变,主要修改fast rcnn部分的model,如下


stage1_fast_rcnn_ohem_train.pt

```
name: "VGG_ILSVRC_16_layers"
layer {
  name: 'data'
  type: 'Python'
  top: 'data'
  top: 'rois'
  top: 'labels'
  top: 'bbox_targets'
  top: 'bbox_inside_weights'
  top: 'bbox_outside_weights'
  python_param {
    module: 'roi_data_layer.layer'
    layer: 'RoIDataLayer'
    param_str: "'num_classes': 21"
  }
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu1_1"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_1"
}
layer {
  name: "conv1_2"
  type: "Convolution"
  bottom: "conv1_1"
  top: "conv1_2"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu1_2"
  type: "ReLU"
  bottom: "conv1_2"
  top: "conv1_2"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1_2"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2_1"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2_1"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu2_1"
  type: "ReLU"
  bottom: "conv2_1"
  top: "conv2_1"
}
layer {
  name: "conv2_2"
  type: "Convolution"
  bottom: "conv2_1"
  top: "conv2_2"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu2_2"
  type: "ReLU"
  bottom: "conv2_2"
  top: "conv2_2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2_2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv3_1"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3_1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3_1"
  type: "ReLU"
  bottom: "conv3_1"
  top: "conv3_1"
}
layer {
  name: "conv3_2"
  type: "Convolution"
  bottom: "conv3_1"
  top: "conv3_2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3_2"
  type: "ReLU"
  bottom: "conv3_2"
  top: "conv3_2"
}
layer {
  name: "conv3_3"
  type: "Convolution"
  bottom: "conv3_2"
  top: "conv3_3"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3_3"
  type: "ReLU"
  bottom: "conv3_3"
  top: "conv3_3"
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3_3"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv4_1"
  type: "Convolution"
  bottom: "pool3"
  top: "conv4_1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu4_1"
  type: "ReLU"
  bottom: "conv4_1"
  top: "conv4_1"
}
layer {
  name: "conv4_2"
  type: "Convolution"
  bottom: "conv4_1"
  top: "conv4_2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu4_2"
  type: "ReLU"
  bottom: "conv4_2"
  top: "conv4_2"
}
layer {
  name: "conv4_3"
  type: "Convolution"
  bottom: "conv4_2"
  top: "conv4_3"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu4_3"
  type: "ReLU"
  bottom: "conv4_3"
  top: "conv4_3"
}
layer {
  name: "pool4"
  type: "Pooling"
  bottom: "conv4_3"
  top: "pool4"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv5_1"
  type: "Convolution"
  bottom: "pool4"
  top: "conv5_1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu5_1"
  type: "ReLU"
  bottom: "conv5_1"
  top: "conv5_1"
}
layer {
  name: "conv5_2"
  type: "Convolution"
  bottom: "conv5_1"
  top: "conv5_2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu5_2"
  type: "ReLU"
  bottom: "conv5_2"
  top: "conv5_2"
}
layer {
  name: "conv5_3"
  type: "Convolution"
  bottom: "conv5_2"
  top: "conv5_3"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu5_3"
  type: "ReLU"
  bottom: "conv5_3"
  top: "conv5_3"
}

##########################
## Readonly RoI Network ##
######### Start ##########
layer {
  name: "roi_pool5_readonly"
  type: "ROIPooling"
  bottom: "conv5_3"
  bottom: "rois"
  top: "pool5_readonly"
  propagate_down: false
  propagate_down: false
  roi_pooling_param {
    pooled_w: 7
    pooled_h: 7
    spatial_scale: 0.0625 # 1/16
  }
}
layer {
  name: "fc6_readonly"
  type: "InnerProduct"
  bottom: "pool5_readonly"
  top: "fc6_readonly"
  propagate_down: false
  param {
    name: "fc6_w"
  }
  param {
    name: "fc6_b"
  }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu6_readonly"
  type: "ReLU"
  bottom: "fc6_readonly"
  top: "fc6_readonly"
  propagate_down: false
}
layer {
  name: "drop6_readonly"
  type: "Dropout"
  bottom: "fc6_readonly"
  top: "fc6_readonly"
  propagate_down: false
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7_readonly"
  type: "InnerProduct"
  bottom: "fc6_readonly"
  top: "fc7_readonly"
  propagate_down: false
  param {
    name: "fc7_w"
  }
  param {
    name: "fc7_b"
  }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu7_readonly"
  type: "ReLU"
  bottom: "fc7_readonly"
  top: "fc7_readonly"
  propagate_down: false
}
layer {
  name: "drop7_readonly"
  type: "Dropout"
  bottom: "fc7_readonly"
  top: "fc7_readonly"
  propagate_down: false
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "cls_score_readonly"
  type: "InnerProduct"
  bottom: "fc7_readonly"
  top: "cls_score_readonly"
  propagate_down: false
  param {
    name: "cls_score_w"
  }
  param {
    name: "cls_score_b"
  }
  inner_product_param {
    num_output: 21
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
layer {
  name: "bbox_pred_readonly"
  type: "InnerProduct"
  bottom: "fc7_readonly"
  top: "bbox_pred_readonly"
  propagate_down: false
  param {
    name: "bbox_pred_w"
  }
  param {
    name: "bbox_pred_b"
  }
  inner_product_param {
    num_output: 84
    weight_filler {
      type: "gaussian"
      std: 0.001
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "cls_prob_readonly"
  type: "Softmax"
  bottom: "cls_score_readonly"
  top: "cls_prob_readonly"
  propagate_down: false
}
layer {
  name: "hard_roi_mining"
  type: "Python"
  bottom: "cls_prob_readonly"
  bottom: "bbox_pred_readonly"
  bottom: "rois"
  bottom: "labels"
  bottom: "bbox_targets"
  bottom: "bbox_inside_weights"
  bottom: "bbox_outside_weights"
  top: "rois_hard"
  top: "labels_hard"
  top: "bbox_targets_hard"
  top: "bbox_inside_weights_hard"
  top: "bbox_outside_weights_hard"
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  python_param {
    module: "roi_data_layer.layer"
    layer: "OHEMDataLayer"
    param_str: "'num_classes': 21"
  }
}
########## End ###########
## Readonly RoI Network ##
##########################

layer {
  name: "roi_pool5"
  type: "ROIPooling"
  bottom: "conv5_3"
  bottom: "rois_hard"
  top: "pool5"
  propagate_down: true
  propagate_down: false
  roi_pooling_param {
    pooled_w: 7
    pooled_h: 7
    spatial_scale: 0.0625 # 1/16
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  param {
    name: "fc6_w"
    lr_mult: 1
  }
  param {
    name: "fc6_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  param {
    name: "fc7_w"
    lr_mult: 1
  }
  param {
    name: "fc7_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "cls_score"
  type: "InnerProduct"
  bottom: "fc7"
  top: "cls_score"
  param {
    name: "cls_score_w"
    lr_mult: 1
  }
  param {
    name: "cls_score_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 21
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
layer {
  name: "bbox_pred"
  type: "InnerProduct"
  bottom: "fc7"
  top: "bbox_pred"
  param {
    name: "bbox_pred_w"
    lr_mult: 1
  }
  param {
    name: "bbox_pred_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 84
    weight_filler {
      type: "gaussian"
      std: 0.001
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "loss_cls"
  type: "SoftmaxWithLoss"
  bottom: "cls_score"
  bottom: "labels_hard"
  top: "loss_cls"
  propagate_down: true
  propagate_down: false
  loss_weight: 1
}
layer {
  name: "loss_bbox"
  type: "SmoothL1Loss"
  bottom: "bbox_pred"
  bottom: "bbox_targets_hard"
  bottom: "bbox_inside_weights_hard"
  bottom: "bbox_outside_weights_hard"
  top: "loss_bbox"
  propagate_down: true
  propagate_down: false
  propagate_down: false
  propagate_down: false
  loss_weight: 1
}

#========= RPN ============
# Dummy layers so that initial parameters are saved into the output net

layer {
  name: "rpn_conv/3x3"
  type: "Convolution"
  bottom: "conv5_3"
  top: "rpn/output"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    kernel_size: 3 pad: 1 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_relu/3x3"
  type: "ReLU"
  bottom: "rpn/output"
  top: "rpn/output"
}

layer {
  name: "rpn_cls_score"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_cls_score"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 18   # 2(bg/fg) * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_bbox_pred"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_bbox_pred"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 36   # 4 * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "silence_rpn_cls_score"
  type: "Silence"
  bottom: "rpn_cls_score"
}
layer {
  name: "silence_rpn_bbox_pred"
  type: "Silence"
  bottom: "rpn_bbox_pred"
}
```

stage2_fast_rcnn_ohem_train.pt

```
name: "VGG_ILSVRC_16_layers"
layer {
  name: 'data'
  type: 'Python'
  top: 'data'
  top: 'rois'
  top: 'labels'
  top: 'bbox_targets'
  top: 'bbox_inside_weights'
  top: 'bbox_outside_weights'
  python_param {
    module: 'roi_data_layer.layer'
    layer: 'RoIDataLayer'
    param_str: "'num_classes': 21"
  }
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu1_1"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_1"
}
layer {
  name: "conv1_2"
  type: "Convolution"
  bottom: "conv1_1"
  top: "conv1_2"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu1_2"
  type: "ReLU"
  bottom: "conv1_2"
  top: "conv1_2"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1_2"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2_1"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2_1"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu2_1"
  type: "ReLU"
  bottom: "conv2_1"
  top: "conv2_1"
}
layer {
  name: "conv2_2"
  type: "Convolution"
  bottom: "conv2_1"
  top: "conv2_2"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu2_2"
  type: "ReLU"
  bottom: "conv2_2"
  top: "conv2_2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2_2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv3_1"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3_1"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3_1"
  type: "ReLU"
  bottom: "conv3_1"
  top: "conv3_1"
}
layer {
  name: "conv3_2"
  type: "Convolution"
  bottom: "conv3_1"
  top: "conv3_2"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3_2"
  type: "ReLU"
  bottom: "conv3_2"
  top: "conv3_2"
}
layer {
  name: "conv3_3"
  type: "Convolution"
  bottom: "conv3_2"
  top: "conv3_3"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3_3"
  type: "ReLU"
  bottom: "conv3_3"
  top: "conv3_3"
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3_3"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv4_1"
  type: "Convolution"
  bottom: "pool3"
  top: "conv4_1"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu4_1"
  type: "ReLU"
  bottom: "conv4_1"
  top: "conv4_1"
}
layer {
  name: "conv4_2"
  type: "Convolution"
  bottom: "conv4_1"
  top: "conv4_2"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu4_2"
  type: "ReLU"
  bottom: "conv4_2"
  top: "conv4_2"
}
layer {
  name: "conv4_3"
  type: "Convolution"
  bottom: "conv4_2"
  top: "conv4_3"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu4_3"
  type: "ReLU"
  bottom: "conv4_3"
  top: "conv4_3"
}
layer {
  name: "pool4"
  type: "Pooling"
  bottom: "conv4_3"
  top: "pool4"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv5_1"
  type: "Convolution"
  bottom: "pool4"
  top: "conv5_1"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu5_1"
  type: "ReLU"
  bottom: "conv5_1"
  top: "conv5_1"
}
layer {
  name: "conv5_2"
  type: "Convolution"
  bottom: "conv5_1"
  top: "conv5_2"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu5_2"
  type: "ReLU"
  bottom: "conv5_2"
  top: "conv5_2"
}
layer {
  name: "conv5_3"
  type: "Convolution"
  bottom: "conv5_2"
  top: "conv5_3"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu5_3"
  type: "ReLU"
  bottom: "conv5_3"
  top: "conv5_3"
}
##########################
## Readonly RoI Network ##
######### Start ##########
layer {
  name: "roi_pool5_readonly"
  type: "ROIPooling"
  bottom: "conv5_3"
  bottom: "rois"
  top: "pool5_readonly"
  propagate_down: false
  propagate_down: false
  roi_pooling_param {
    pooled_w: 7
    pooled_h: 7
    spatial_scale: 0.0625 # 1/16
  }
}
layer {
  name: "fc6_readonly"
  type: "InnerProduct"
  bottom: "pool5_readonly"
  top: "fc6_readonly"
  propagate_down: false
  param {
    name: "fc6_w"
  }
  param {
    name: "fc6_b"
  }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu6_readonly"
  type: "ReLU"
  bottom: "fc6_readonly"
  top: "fc6_readonly"
  propagate_down: false
}
layer {
  name: "drop6_readonly"
  type: "Dropout"
  bottom: "fc6_readonly"
  top: "fc6_readonly"
  propagate_down: false
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7_readonly"
  type: "InnerProduct"
  bottom: "fc6_readonly"
  top: "fc7_readonly"
  propagate_down: false
  param {
    name: "fc7_w"
  }
  param {
    name: "fc7_b"
  }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu7_readonly"
  type: "ReLU"
  bottom: "fc7_readonly"
  top: "fc7_readonly"
  propagate_down: false
}
layer {
  name: "drop7_readonly"
  type: "Dropout"
  bottom: "fc7_readonly"
  top: "fc7_readonly"
  propagate_down: false
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "cls_score_readonly"
  type: "InnerProduct"
  bottom: "fc7_readonly"
  top: "cls_score_readonly"
  propagate_down: false
  param {
    name: "cls_score_w"
  }
  param {
    name: "cls_score_b"
  }
  inner_product_param {
    num_output: 21
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
layer {
  name: "bbox_pred_readonly"
  type: "InnerProduct"
  bottom: "fc7_readonly"
  top: "bbox_pred_readonly"
  propagate_down: false
  param {
    name: "bbox_pred_w"
  }
  param {
    name: "bbox_pred_b"
  }
  inner_product_param {
    num_output: 84
    weight_filler {
      type: "gaussian"
      std: 0.001
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "cls_prob_readonly"
  type: "Softmax"
  bottom: "cls_score_readonly"
  top: "cls_prob_readonly"
  propagate_down: false
}
layer {
  name: "hard_roi_mining"
  type: "Python"
  bottom: "cls_prob_readonly"
  bottom: "bbox_pred_readonly"
  bottom: "rois"
  bottom: "labels"
  bottom: "bbox_targets"
  bottom: "bbox_inside_weights"
  bottom: "bbox_outside_weights"
  top: "rois_hard"
  top: "labels_hard"
  top: "bbox_targets_hard"
  top: "bbox_inside_weights_hard"
  top: "bbox_outside_weights_hard"
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  propagate_down: false
  python_param {
    module: "roi_data_layer.layer"
    layer: "OHEMDataLayer"
    param_str: "'num_classes': 21"
  }
}
########## End ###########
## Readonly RoI Network ##
##########################

layer {
  name: "roi_pool5"
  type: "ROIPooling"
  bottom: "conv5_3"
  bottom: "rois_hard"
  top: "pool5"
  propagate_down: true
  propagate_down: false
  roi_pooling_param {
    pooled_w: 7
    pooled_h: 7
    spatial_scale: 0.0625 # 1/16
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  param { name: "fc6_w" lr_mult: 1 }
  param { name: "fc6_b" lr_mult: 2 }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  param { name: "fc7_w" lr_mult: 1 }
  param { name: "fc7_b" lr_mult: 2 }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "cls_score"
  type: "InnerProduct"
  bottom: "fc7"
  top: "cls_score"
  param { name: "cls_score_w" lr_mult: 1 }
  param { name: "cls_score_b" lr_mult: 2 }
  inner_product_param {
    num_output: 21
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
layer {
  name: "bbox_pred"
  type: "InnerProduct"
  bottom: "fc7"
  top: "bbox_pred"
  param { name: "bbox_pred_w" lr_mult: 1 }
  param { name: "bbox_pred_b" lr_mult: 2 }
  inner_product_param {
    num_output: 84
    weight_filler {
      type: "gaussian"
      std: 0.001
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "loss_cls"
  type: "SoftmaxWithLoss"
  bottom: "cls_score"
  bottom: "labels_hard"
  top: "loss_cls"
  propagate_down: true
  propagate_down: false
  loss_weight: 1
}
layer {
  name: "loss_bbox"
  type: "SmoothL1Loss"
  bottom: "bbox_pred"
  bottom: "bbox_targets_hard"
  bottom: "bbox_inside_weights_hard"
  bottom: "bbox_outside_weights_hard"
  top: "loss_bbox"
  propagate_down: true
  propagate_down: false
  propagate_down: false
  propagate_down: false
  loss_weight: 1
}

#========= RPN ============
# Dummy layers so that initial parameters are saved into the output net

layer {
  name: "rpn_conv/3x3"
  type: "Convolution"
  bottom: "conv5_3"
  top: "rpn/output"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    kernel_size: 3 pad: 1 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_relu/3x3"
  type: "ReLU"
  bottom: "rpn/output"
  top: "rpn/output"
}

layer {
  name: "rpn_cls_score"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_cls_score"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 18   # 2(bg/fg) * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_bbox_pred"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_bbox_pred"
  param { lr_mult: 0 decay_mult: 0 }
  param { lr_mult: 0 decay_mult: 0 }
  convolution_param {
    num_output: 36   # 4 * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "silence_rpn_cls_score"
  type: "Silence"
  bottom: "rpn_cls_score"
}
layer {
  name: "silence_rpn_bbox_pred"
  type: "Silence"
  bottom: "rpn_bbox_pred"
}
```

如上所示,两个过程主要增加了一个read only的模块用于计算RPN输出全部roi的loss,增加一个OHEMDataLayer层用于选择最大loss的roi参与fast rcnn的训练.注意,OHEM时所有的层不更新参数所以所有readony部分layer的bottom都需要关闭反向梯度传播"propagate_down:false",这里顺便提一下,原版的faster rcnn不需要反传的bottom没有设置propagate_down:false也没有出错,这是因为caffe有一个智能分析哪些bottom需要反传梯度的识别机制,当这种机制失效时候就需要人工设置propagate_down:false.同时通过设置相同的类似param { name: "fc6_w"  }来共享层参数.

minibatch.py中需要添加

```
def get_allrois_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        # Doesn't support RPN yet.
        assert False
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)

        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _all_rois(roidb[im_i], num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs


def get_ohem_minibatch(loss, rois, labels, bbox_targets=None,
                       bbox_inside_weights=None, bbox_outside_weights=None):
    """Given rois and their loss, construct a minibatch using OHEM."""
    loss = np.array(loss)

    if cfg.TRAIN.OHEM_USE_NMS:
        # Do NMS using loss for de-dup and diversity
        keep_inds = []
        nms_thresh = cfg.TRAIN.OHEM_NMS_THRESH
        source_img_ids = [roi[0] for roi in rois]
        for img_id in np.unique(source_img_ids):
            for label in np.unique(labels):
                sel_indx = np.where(np.logical_and(labels == label, \
                                    source_img_ids == img_id))[0]
                if not len(sel_indx):
                    continue
                boxes = np.concatenate((rois[sel_indx, 1:],
                        loss[sel_indx][:,np.newaxis]), axis=1).astype(np.float32)
                keep_inds.extend(sel_indx[nms(boxes, nms_thresh)])

        hard_keep_inds = select_hard_examples(loss[keep_inds])
        hard_inds = np.array(keep_inds)[hard_keep_inds]
    else:
        hard_inds = select_hard_examples(loss)

    blobs = {'rois_hard': rois[hard_inds, :].copy(),
             'labels_hard': labels[hard_inds].copy()}
    if bbox_targets is not None:
        assert cfg.TRAIN.BBOX_REG
        blobs['bbox_targets_hard'] = bbox_targets[hard_inds, :].copy()
        blobs['bbox_inside_weights_hard'] = bbox_inside_weights[hard_inds, :].copy()
        blobs['bbox_outside_weights_hard'] = bbox_outside_weights[hard_inds, :].copy()

    return blobs

def select_hard_examples(loss):
    """Select hard rois."""
    # Sort and select top hard examples.
    sorted_indices = np.argsort(loss)[::-1]
    hard_keep_inds = sorted_indices[0:np.minimum(len(loss), cfg.TRAIN.BATCH_SIZE)]
    # (explore more ways of selecting examples in this function; e.g., sampling)

    return hard_keep_inds
```

minibatch.py中修改

```
def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights
```

layer.py中添加一个OHEM层

```
class OHEMDataLayer(caffe.Layer):
    """Online Hard-example Mining Layer."""
    def setup(self, bottom, top):
        """Setup the OHEMDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_bottom_map = {
            'cls_prob_readonly': 0,
            'bbox_pred_readonly': 1,
            'rois': 2,
            'labels': 3}

        if cfg.TRAIN.BBOX_REG:
            self._name_to_bottom_map['bbox_targets'] = 4
            self._name_to_bottom_map['bbox_loss_weights'] = 5

        self._name_to_top_map = {}

        assert cfg.TRAIN.HAS_RPN == False
        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[idx].reshape(1, 5)
        self._name_to_top_map['rois_hard'] = idx
        idx += 1

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[idx].reshape(1)
        self._name_to_top_map['labels_hard'] = idx
        idx += 1

        if cfg.TRAIN.BBOX_REG:
            # bbox_targets blob: R bounding-box regression targets with 4
            # targets per class
            top[idx].reshape(1, self._num_classes * 4)
            self._name_to_top_map['bbox_targets_hard'] = idx
            idx += 1

            # bbox_inside_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top[idx].reshape(1, self._num_classes * 4)
            self._name_to_top_map['bbox_inside_weights_hard'] = idx
            idx += 1

            top[idx].reshape(1, self._num_classes * 4)
            self._name_to_top_map['bbox_outside_weights_hard'] = idx
            idx += 1

        print 'OHEMDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

        cls_prob = bottom[0].data
        bbox_pred = bottom[1].data
        rois = bottom[2].data
        labels = bottom[3].data
        if cfg.TRAIN.BBOX_REG:
            bbox_target = bottom[4].data
            bbox_inside_weights = bottom[5].data
            bbox_outside_weights = bottom[6].data
        else:
            bbox_target = None
            bbox_inside_weights = None
            bbox_outside_weights = None

        flt_min = np.finfo(float).eps
        # classification loss
        loss = [ -1 * np.log(max(x, flt_min)) \
            for x in [cls_prob[i,label] for i, label in enumerate(labels)]]

        if cfg.TRAIN.BBOX_REG:
            # bounding-box regression loss
            # d := w * (b0 - b1)
            # smoothL1(x) = 0.5 * x^2    if |x| < 1
            #               |x| - 0.5    otherwise
            def smoothL1(x):
                if abs(x) < 1:
                    return 0.5 * x * x
                else:
                    return abs(x) - 0.5

            bbox_loss = np.zeros(labels.shape[0])
            for i in np.where(labels > 0 )[0]:
                indices = np.where(bbox_inside_weights[i,:] != 0)[0]
                bbox_loss[i] = sum(bbox_outside_weights[i,indices] * [smoothL1(x) \
                    for x in bbox_inside_weights[i,indices] * (bbox_pred[i,indices] - bbox_target[i,indices])])
            loss += bbox_loss

        blobs = get_ohem_minibatch(loss, rois, labels, bbox_target, \
            bbox_inside_weights, bbox_outside_weights)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
```

修改layer.py中的_get_next_minibatch,run函数

```
    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            if cfg.TRAIN.USE_OHEM:
                blobs = get_allrois_minibatch(minibatch_db, self._num_classes)
            else:
                blobs = get_minibatch(minibatch_db, self._num_classes)

            return blobs

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            if cfg.TRAIN.USE_OHEM:
                blobs = get_allrois_minibatch(minibatch_db, self._num_classes)
            else:
                blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)
```

在train_faster_rcnn_alt_opt.py中修改

```
def get_solvers(net_name):
    # Faster R-CNN Alternating Optimization
    n = 'faster_rcnn_alt_opt'
    # Solver for each training stage
    solvers = [[net_name, n, 'stage1_rpn_solver60k80k.pt'],
               [net_name, n, 'stage1_fast_rcnn_ohem_solver30k40k.pt'],
               [net_name, n, 'stage2_rpn_solver60k80k.pt'],
               [net_name, n, 'stage2_fast_rcnn_ohem_solver30k40k.pt']]
    solvers = [os.path.join(cfg.ROOT_DIR, 'models', *s) for s in solvers]
    # Iterations for each training stage
    max_iters = [80000, 40000, 80000, 40000]
    #max_iters = [50, 50, 50, 50]
    # Test prototxt for the RPN
    rpn_test_prototxt = os.path.join(
        cfg.ROOT_DIR, 'models', net_name, n, 'rpn_test.pt')
    return solvers, max_iters, rpn_test_prototxt

def train_rpn(queue=None, imdb_name=None, init_model=None, solver=None,
              max_iters=None, cfg=None):
    """Train a Region Proposal Network in a separate training process.
    """

    # Not using any proposals, just ground-truth boxes
    cfg.TRAIN.USE_OHEM = False

    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.BBOX_REG = False  # applies only to Fast R-CNN bbox regression
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name)
    print 'roidb len: {}'.format(len(roidb))
    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    rpn_model_path = model_paths[-1]
    # Send final model path through the multiprocessing queue
    queue.put({'model_path': rpn_model_path})

def train_fast_rcnn(queue=None, imdb_name=None, init_model=None, solver=None,
                    max_iters=None, cfg=None, rpn_file=None):
    """Train a Fast R-CNN using proposals generated by an RPN.
    """

    cfg.TRAIN.USE_OHEM = True
    cfg.TRAIN.BG_THRESH_LO = 0.0
    cfg.ASPECT_GROUPING = False
    cfg.TRAIN.OHEM_USE_NMS = False
    
    cfg.TRAIN.HAS_RPN = False           # not generating prosals on-the-fly
    cfg.TRAIN.PROPOSAL_METHOD = 'rpn'   # use pre-computed RPN proposals instead
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print 'RPN proposals: {}'.format(rpn_file)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name, rpn_file=rpn_file)
    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Train Fast R-CNN
    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    fast_rcnn_model_path = model_paths[-1]
    # Send Fast R-CNN model path over the multiprocessing queue
    queue.put({'model_path': fast_rcnn_model_path})
```

项目地址:https://github.com/manutdzou/py-faster-rcnn-OHEM

看我写的辛苦求打赏啊！！！有学术讨论和指点请加微信manutdzou,注明

![20](/public/img/pay.jpg)